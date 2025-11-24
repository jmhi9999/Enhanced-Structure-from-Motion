#!/usr/bin/env python3
"""
Sequentially run ground-truth benchmarks across multiple scenes and extractor/pair-selector combos.

Example:
    python -m SfMFramework.testing.run_benchmark_suite \
        --images-root ImageInputs \
        --output-root results/benchmark_suite \
        --colmap-min-shared-points 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from SfMFramework.extractors import ExtractorFactory
from SfMFramework.pair_selectors import PairSelectorFactory

logger = logging.getLogger("benchmark_suite")


def run_command(cmd: List[str], dry_run: bool) -> int:
    """Run a shell command and return the exit code."""
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return 0

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        logger.error("Command failed with return code %s", completed.returncode)
    return completed.returncode


@dataclass
class BenchmarkRecord:
    scene: str
    extractor: str
    matcher: str
    pair_selector: str
    status: str
    output: str


def decide_matcher(extractor: str, learned_extractors: Set[str]) -> str:
    """Select matcher based on extractor type."""
    return "lightglue" if extractor in learned_extractors else "nn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate SfMFramework.testing.benchmark over multiple scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("ImageInputs"),
        help="Root directory containing scene folders.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["bonsai", "garden", "kitchen", "room", "stump"],
        help="Scene folder names to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/benchmark_suite"),
        help="Directory where benchmark outputs will be stored.",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=None,
        help="Optional subset of extractors to evaluate (defaults to all registered).",
    )
    parser.add_argument(
        "--pair-selectors",
        nargs="+",
        default=None,
        help="Optional subset of pair selectors (for bookkeeping/paths). Defaults to all registered.",
    )
    parser.add_argument(
        "--learned-extractors",
        nargs="+",
        default=None,
        help="Explicit list of learning-based extractors (otherwise auto-detected).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device forwarded to the benchmark script.",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=4096,
        help="Max keypoints forwarded to the benchmark script.",
    )
    parser.add_argument(
        "--resize-max",
        type=int,
        default=None,
        help="Optional resize_max forwarded to the benchmark script.",
    )
    parser.add_argument(
        "--colmap-min-shared-points",
        type=int,
        default=30,
        help="Minimum shared 3D points per image pair when building the dataset.",
    )
    parser.add_argument(
        "--use-magsac",
        action="store_true",
        help="Enable MAGSAC filtering during evaluation.",
    )
    parser.add_argument(
        "--magsac-threshold",
        type=float,
        default=2.0,
        help="MAGSAC reprojection threshold in pixels.",
    )
    parser.add_argument(
        "--magsac-confidence",
        type=float,
        default=0.999,
        help="MAGSAC confidence level.",
    )
    parser.add_argument(
        "--magsac-max-iters",
        type=int,
        default=1000,
        help="MAGSAC maximum iterations.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.8,
        help="Distance threshold for NN matcher (0.8 for SIFT/ORB, 0.7 for learned).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--suite-name",
        type=str,
        default=None,
        help="Optional label stored in the suite summary.",
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

    if args.learned_extractors is not None:
        learned_extractors = set(args.learned_extractors)
    else:
        candidate_learned = {"superpoint", "aliked", "disk"}
        learned_extractors = {name for name in extractors if name in candidate_learned}

    logger.info("Extractors (all): %s", extractors)
    logger.info("Pair selectors: %s", pair_selectors)
    if learned_extractors:
        logger.info("Learning-based extractor subset (LightGlue): %s", sorted(learned_extractors))
    else:
        logger.info("Learning-based extractor subset is empty; all extractors use NN matcher.")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    records: List[BenchmarkRecord] = []

    for scene in args.scenes:
        scene_root = args.images_root / scene
        if not scene_root.exists():
            logger.warning("Skipping scene '%s' (missing directory %s)", scene, scene_root)
            continue

        for extractor in extractors:
            matcher = decide_matcher(extractor, learned_extractors)

            for pair_selector in pair_selectors:
                combo_name = f"{extractor}_{matcher}_{pair_selector}"
                output_dir = args.output_root / scene / combo_name
                summary_file = output_dir / "summary.json"
                status = "pending"

                if args.skip_existing and summary_file.exists():
                    logger.info("Skipping %s/%s (cached summary)", scene, combo_name)
                    status = "cached"
                    records.append(
                        BenchmarkRecord(
                            scene=scene,
                            extractor=extractor,
                            matcher=matcher,
                            pair_selector=pair_selector,
                            status=status,
                            output=str(output_dir),
                        )
                    )
                    continue

                if not args.dry_run:
                    output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    "-m",
                    "SfMFramework.testing.benchmark",
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
                    str(output_dir),
                    "--device",
                    args.device,
                    "--max_keypoints",
                    str(args.max_keypoints),
                    "--colmap_min_shared_points",
                    str(args.colmap_min_shared_points),
                    "--distance_threshold",
                    str(args.distance_threshold),
                ]

                if args.resize_max:
                    cmd.extend(["--resize_max", str(args.resize_max)])
                if args.use_magsac:
                    cmd.extend([
                        "--use_magsac_filtering",
                        "--magsac_threshold", str(args.magsac_threshold),
                        "--magsac_confidence", str(args.magsac_confidence),
                        "--magsac_max_iters", str(args.magsac_max_iters),
                    ])

                exit_code = run_command(cmd, args.dry_run)

                status = "ok" if exit_code == 0 else f"failed ({exit_code})"
                records.append(
                    BenchmarkRecord(
                        scene=scene,
                        extractor=extractor,
                        matcher=matcher,
                        pair_selector=pair_selector,
                        status=status,
                        output=str(output_dir),
                    )
                )

                if exit_code == 0 and not args.dry_run:
                    meta = {
                        "scene": scene,
                        "extractor": extractor,
                        "matcher": matcher,
                        "pair_selector": pair_selector,
                        "device": args.device,
                        "max_keypoints": args.max_keypoints,
                        "resize_max": args.resize_max,
                        "colmap_min_shared_points": args.colmap_min_shared_points,
                        "distance_threshold": args.distance_threshold,
                        "use_magsac": args.use_magsac,
                        "magsac_threshold": args.magsac_threshold if args.use_magsac else None,
                        "magsac_confidence": args.magsac_confidence if args.use_magsac else None,
                        "magsac_max_iters": args.magsac_max_iters if args.use_magsac else None,
                    }
                    with (output_dir / "benchmark_meta.json").open("w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=2)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "suite_name": args.suite_name,
        "images_root": str(args.images_root),
        "output_root": str(args.output_root),
        "device": args.device,
        "max_keypoints": args.max_keypoints,
        "resize_max": args.resize_max,
        "colmap_min_shared_points": args.colmap_min_shared_points,
        "distance_threshold": args.distance_threshold,
        "use_magsac": args.use_magsac,
        "magsac_threshold": args.magsac_threshold if args.use_magsac else None,
        "magsac_confidence": args.magsac_confidence if args.use_magsac else None,
        "magsac_max_iters": args.magsac_max_iters if args.use_magsac else None,
        "records": [asdict(record) for record in records],
    }

    if args.dry_run:
        logger.info("Dry-run summary (not written):\n%s", json.dumps(summary, indent=2))
    else:
        summary_file = args.output_root / "benchmark_suite_summary.json"
        with summary_file.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Saved suite summary to %s", summary_file)


if __name__ == "__main__":
    main()
