#!/usr/bin/env python3
"""
Automate SfM reconstruction and ground-truth benchmarking across multiple scenes.

Example:
    python -m matching_framework.testing.run_colmap_suite \
        --images-root ImageInputs \
        --output-root results/colmap_suite \
        --device cuda \
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from SfMFramework.extractors import ExtractorFactory
from SfMFramework.pair_selectors import PairSelectorFactory

logger = logging.getLogger("colmap_suite")


def decide_matcher(extractor: str) -> str:
    """Auto-select matcher based on extractor type.

    Args:
        extractor: Extractor name

    Returns:
        Matcher name ('nn' or 'lightglue')
    """
    # Learned extractors use LightGlue
    learned_extractors = {"superpoint", "aliked", "disk"}
    if extractor in learned_extractors:
        return "lightglue"

    # Traditional extractors (SIFT, ORB, etc.) use NN
    # Note: NN matcher auto-detects Hamming vs L2 based on descriptor type
    return "nn"


def should_use_colmap_cli(extractor: str, pair_selector: str) -> bool:
    """Check if we can use COLMAP CLI for this configuration.

    COLMAP CLI is 10-50x faster but only supports:
    - SIFT extractor
    - exhaustive or vocab_tree matchers

    Args:
        extractor: Feature extractor name
        pair_selector: Pair selector name

    Returns:
        True if COLMAP CLI should be used
    """
    return (
        extractor == "sift" and
        pair_selector in ["exhaustive", "vocab_tree"]
    )


def _contains_images(directory: Path) -> bool:
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
    for pattern in patterns:
        if next(directory.glob(pattern), None) is not None:
            return True
    return False


def resolve_images_dir(scene_root: Path, preferred: Optional[str]) -> Path:
    if preferred:
        candidate = scene_root / preferred
        if candidate.exists() and _contains_images(candidate):
            return candidate

    candidate_names = [
        "images_2",
        "images",
        "Images_2",
        "Images",
        "rgb",
        "RGB",
    ]
    for name in candidate_names:
        candidate = scene_root / name
        if candidate.exists() and _contains_images(candidate):
            return candidate

    for child in scene_root.iterdir():
        if child.is_dir() and _contains_images(child):
            return child

    raise FileNotFoundError(f"Could not locate image directory under {scene_root}")


def run_command(cmd: List[str], dry_run: bool, cwd: Path, env: dict = None) -> int:
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return 0

    import os
    # Merge custom env with current environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    completed = subprocess.run(cmd, cwd=str(cwd), env=run_env)
    if completed.returncode != 0:
        logger.error("Command failed with return code %s", completed.returncode)
    return completed.returncode


def run_colmap_cli_pipeline(
    images_dir: Path,
    output_dir: Path,
    pair_selector: str,
    use_gpu: bool,
    max_num_features: int,
    dry_run: bool,
    colmap_executable: str = "colmap",
) -> int:
    """Run COLMAP CLI pipeline (feature extraction + matching + reconstruction).

    This is 10-50x faster than Python implementation for SIFT.
    Based on convert.py from Gaussian Splatting.

    Args:
        images_dir: Directory containing input images
        output_dir: Output directory for COLMAP database and reconstruction
        pair_selector: "exhaustive" or "vocab_tree"
        use_gpu: Whether to use GPU acceleration
        max_num_features: Maximum number of features per image
        dry_run: If True, only print commands
        colmap_executable: Path to COLMAP executable

    Returns:
        Exit code (0 = success)
    """
    db_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir_0 = sparse_dir / "0"

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

    use_gpu_flag = "1" if use_gpu else "0"

    # Set Qt to headless mode for WSL/SSH environments
    colmap_env = {
        "QT_QPA_PLATFORM": "offscreen",
    }

    # 1. Feature extraction
    logger.info("COLMAP CLI: Feature extraction (SIFT, GPU=%s)", use_gpu_flag)
    feat_cmd = [
        colmap_executable, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", use_gpu_flag,
        "--SiftExtraction.max_num_features", str(max_num_features),
    ]
    rc = run_command(feat_cmd, dry_run, cwd=Path.cwd(), env=colmap_env)
    if rc != 0:
        return rc

    # 2. Feature matching
    if pair_selector == "exhaustive":
        logger.info("COLMAP CLI: Exhaustive matching (GPU=%s)", use_gpu_flag)
        match_cmd = [
            colmap_executable, "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", use_gpu_flag,
        ]
    elif pair_selector == "vocab_tree":
        logger.info("COLMAP CLI: Vocabulary tree matching (GPU=%s)", use_gpu_flag)
        # Note: vocab_tree_matcher requires a pre-built vocabulary tree
        # For now, fallback to sequential matcher as a fast alternative
        match_cmd = [
            colmap_executable, "sequential_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", use_gpu_flag,
            "--SequentialMatching.overlap", "10",
        ]
        logger.warning(
            "vocab_tree requires pre-built tree; using sequential_matcher as fast alternative"
        )
    else:
        logger.error("Unsupported pair_selector for COLMAP CLI: %s", pair_selector)
        return 1

    rc = run_command(match_cmd, dry_run, cwd=Path.cwd(), env=colmap_env)
    if rc != 0:
        return rc

    # 3. Sparse reconstruction (mapper)
    logger.info("COLMAP CLI: Sparse reconstruction (mapper)")
    mapper_cmd = [
        colmap_executable, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_function_tolerance", "0.000001",
    ]
    rc = run_command(mapper_cmd, dry_run, cwd=Path.cwd(), env=colmap_env)
    if rc != 0:
        return rc

    # 4. Export statistics (if reconstruction succeeded)
    if not dry_run and sparse_dir_0.exists():
        logger.info("COLMAP CLI: Reconstruction successful")
        # Export basic statistics to match Python pipeline format
        stats_file = output_dir / "statistics.json"
        stats = {
            "method": "colmap_cli",
            "extractor": "sift",
            "pair_selector": pair_selector,
            "note": "Generated by COLMAP CLI (10-50x faster than Python)",
        }
        with stats_file.open("w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved statistics to %s", stats_file)

    return 0


@dataclass
class RunRecord:
    scene: str
    extractor: str
    matcher: str
    pair_selector: str
    pipeline_status: str
    benchmark_status: str
    pipeline_output: str
    benchmark_output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SfM pipeline and benchmark across scenes and configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("ImageInputs"),
        help="Root folder containing scene sub-directories",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["bonsai", "garden", "kitchen", "room", "stump"],
        help="Scene folder names relative to --images-root",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images_2",
        help="Preferred sub-folder that stores RGB images (auto-fallback if missing)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/colmap_suite"),
        help="Where to store pipeline and benchmark results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device argument forwarded to both pipeline and benchmark",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=4096,
        help="Max keypoints per image for SfM pipeline",
    )
    parser.add_argument(
        "--resize-max",
        type=int,
        default=None,
        help="Optional resize_max forwarded to SfM pipeline",
    )
    parser.add_argument(
        "--colmap-min-shared-points",
        type=int,
        default=30,
        help="Minimum shared 3D points when building benchmark pairs",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.8,
        help="Distance threshold for NN matcher (0.8 recommended for SIFT/ORB, 0.7 for learned)",
    )
    parser.add_argument(
        "--max-pairs-per-image",
        type=int,
        default=20,
        help="Max pairs per image for pair selector (forwarded to benchmark)",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=None,
        help="Subset of extractors to evaluate (defaults to all registered)",
    )
    parser.add_argument(
        "--pair-selectors",
        nargs="+",
        default=None,
        help="Subset of pair selectors to evaluate (defaults to all registered)",
    )
    parser.add_argument(
        "--pipeline-use-magsac",
        action="store_true",
        help="Enable MAGSAC filtering inside SfM pipeline",
    )
    parser.add_argument(
        "--benchmark-use-magsac",
        action="store_true",
        help="Enable MAGSAC filtering during benchmark evaluation",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps where outputs already exist (statistics.json / summary.json)",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Do not run SfM pipeline, only benchmark existing reconstructions",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Run SfM pipeline only, skip benchmark evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--suite-name",
        type=str,
        default=None,
        help="Optional identifier stored in summary metadata",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    extractors = args.extractors or ExtractorFactory.list_extractors()
    pair_selectors = args.pair_selectors or PairSelectorFactory.list_selectors()

    logger.info("Extractors: %s", extractors)
    logger.info("Pair selectors: %s", pair_selectors)
    logger.info("Matcher selection: auto (learned→lightglue, traditional→nn)")

    output_root = args.output_root
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    suite_records: List[RunRecord] = []

    for scene in args.scenes:
        scene_root = args.images_root / scene
        if not scene_root.exists():
            logger.warning("Skipping scene '%s' (missing directory %s)", scene, scene_root)
            continue

        try:
            images_dir = resolve_images_dir(scene_root, args.images_subdir)
        except FileNotFoundError as exc:
            logger.error("Skipping scene '%s': %s", scene, exc)
            continue

        logger.info("Scene '%s' images: %s", scene, images_dir)

        for extractor, pair_selector in itertools.product(extractors, pair_selectors):
            # Auto-select matcher based on extractor type
            matcher = decide_matcher(extractor)
            combo_name = f"{extractor}_{matcher}_{pair_selector}"
            logger.info("==== Scene %s | %s ====", scene, combo_name)

            combo_base = output_root / scene / combo_name
            pipeline_dir = combo_base / "sfm"
            benchmark_dir = combo_base / "benchmark"

            if not args.dry_run:
                pipeline_dir.mkdir(parents=True, exist_ok=True)
                benchmark_dir.mkdir(parents=True, exist_ok=True)

            pipeline_status = "skipped" if args.skip_pipeline else "pending"
            benchmark_status = "skipped" if args.skip_benchmark else "pending"

            if not args.skip_pipeline:
                stats_file = pipeline_dir / "statistics.json"
                if args.skip_existing and stats_file.exists():
                    logger.info("Skipping pipeline (existing %s)", stats_file)
                    pipeline_status = "cached"
                else:
                    # Check if we can use COLMAP CLI (10-50x faster for SIFT)
                    use_cli = should_use_colmap_cli(extractor, pair_selector)

                    if use_cli:
                        logger.info("Using COLMAP CLI (10-50x faster than Python)")
                        use_gpu = args.device == "cuda"
                        rc = run_colmap_cli_pipeline(
                            images_dir=images_dir,
                            output_dir=pipeline_dir,
                            pair_selector=pair_selector,
                            use_gpu=use_gpu,
                            max_num_features=args.max_keypoints,
                            dry_run=args.dry_run,
                        )
                    else:
                        logger.info("Using Python pipeline")
                        cmd = [
                            sys.executable,
                            "-m",
                            "matching_framework.sfm_pipeline",
                            "--images",
                            str(images_dir),
                            "--output",
                            str(pipeline_dir),
                            "--extractor",
                            extractor,
                            "--matcher",
                            matcher,
                            "--pair_selector",
                            pair_selector,
                            "--device",
                            args.device,
                            "--max_keypoints",
                            str(args.max_keypoints),
                        ]
                        if args.resize_max:
                            cmd.extend(["--resize_max", str(args.resize_max)])
                        if args.pipeline_use_magsac:
                            cmd.append("--use_magsac_filtering")

                        rc = run_command(cmd, args.dry_run, cwd=Path.cwd())

                    pipeline_status = "ok" if rc == 0 else f"failed ({rc})"

            if not args.skip_benchmark:
                summary_file = benchmark_dir / "summary.json"
                if args.skip_existing and summary_file.exists():
                    logger.info("Skipping benchmark (existing %s)", summary_file)
                    benchmark_status = "cached"
                else:
                    cmd = [
                        sys.executable,
                        "-m",
                        "matching_framework.testing.benchmark",
                        "--dataset",
                        "colmap_scene",
                        "--dataset_path",
                        str(scene_root),
                        "--extractor",
                        extractor,
                        "--matcher",
                        matcher,
                        "--pair_selector",
                        pair_selector,
                        "--output",
                        str(benchmark_dir),
                        "--device",
                        args.device,
                        "--colmap_min_shared_points",
                        str(args.colmap_min_shared_points),
                        "--distance_threshold",
                        str(args.distance_threshold),
                        "--max_pairs_per_image",
                        str(args.max_pairs_per_image),
                    ]
                    if args.max_keypoints:
                        cmd.extend(["--max_keypoints", str(args.max_keypoints)])
                    if args.resize_max:
                        cmd.extend(["--resize_max", str(args.resize_max)])
                    if args.benchmark_use_magsac:
                        cmd.append("--use_magsac_filtering")

                    rc = run_command(cmd, args.dry_run, cwd=Path.cwd())
                    benchmark_status = "ok" if rc == 0 else f"failed ({rc})"

            suite_records.append(
                RunRecord(
                    scene=scene,
                    extractor=extractor,
                    matcher=matcher,
                    pair_selector=pair_selector,
                    pipeline_status=pipeline_status,
                    benchmark_status=benchmark_status,
                    pipeline_output=str(pipeline_dir),
                    benchmark_output=str(benchmark_dir),
                )
            )

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "suite_name": args.suite_name,
        "images_root": str(args.images_root),
        "device": args.device,
        "resize_max": args.resize_max,
        "max_keypoints": args.max_keypoints,
        "colmap_min_shared_points": args.colmap_min_shared_points,
        "distance_threshold": args.distance_threshold,
        "max_pairs_per_image": args.max_pairs_per_image,
        "pipeline_use_magsac": args.pipeline_use_magsac,
        "benchmark_use_magsac": args.benchmark_use_magsac,
        "records": [asdict(record) for record in suite_records],
    }

    summary_file = output_root / "suite_summary.json"
    if args.dry_run:
        logger.info("Dry-run summary (not written):\n%s", json.dumps(summary, indent=2))
    else:
        with summary_file.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Saved suite summary to %s", summary_file)


if __name__ == "__main__":
    main()
