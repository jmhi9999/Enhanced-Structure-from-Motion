#!/usr/bin/env python3
"""
Benchmark script for evaluating feature matching on standard datasets.

Usage:
    python -m matching_framework.testing.benchmark \
        --dataset eth3d \
        --dataset_path data/eth3d \
        --extractor superpoint \
        --matcher lightglue \
        --output results/eth3d
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
import sys
import cv2

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors import ExtractorFactory, ExtractorConfig
from matchers import MatcherFactory, MatcherConfig
from pair_selectors import PairSelectorFactory, PairSelectorConfig
from testing.datasets import DatasetFactory
from testing.metrics import evaluate_matches, evaluate_pose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def filter_matches_with_magsac(
    features0_kpts: np.ndarray,
    features1_kpts: np.ndarray,
    match_indices: np.ndarray,
    threshold: float = 2.0,
    confidence: float = 0.999,
    max_iters: int = 1000,
) -> np.ndarray:
    """Filter matches using MAGSAC (same as sfm_pipeline for realistic evaluation).

    Args:
        features0_kpts: Keypoints from image 0 [N, 2]
        features1_kpts: Keypoints from image 1 [M, 2]
        match_indices: Match indices [K, 2] (indices into features0 and features1)
        threshold: MAGSAC reprojection threshold in pixels
        confidence: MAGSAC confidence level
        max_iters: Maximum RANSAC iterations

    Returns:
        Filtered match indices [K', 2] after MAGSAC
    """
    if len(match_indices) < 8:
        return np.array([]).reshape(0, 2)

    # Get matched keypoints
    matched_kpts0 = features0_kpts[match_indices[:, 0]]
    matched_kpts1 = features1_kpts[match_indices[:, 1]]

    # Run MAGSAC
    F_matrix, inlier_mask = cv2.findFundamentalMat(
        matched_kpts0.astype(np.float32),
        matched_kpts1.astype(np.float32),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=threshold,
        confidence=confidence,
        maxIters=max_iters,
    )

    if F_matrix is None or inlier_mask is None:
        return np.array([]).reshape(0, 2)

    # Keep only inlier matches
    inlier_mask = inlier_mask.ravel().astype(bool)

    if inlier_mask.sum() < 8:
        return np.array([]).reshape(0, 2)

    return match_indices[inlier_mask]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark feature matching on datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DatasetFactory.list_datasets(),
        help="Dataset name"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument(
        "--colmap_min_shared_points",
        type=int,
        default=30,
        help="Minimum number of shared 3D tracks when using 'colmap_scene' dataset",
    )

    # Feature extractor
    parser.add_argument(
        "--extractor",
        type=str,
        default="superpoint",
        choices=ExtractorFactory.list_extractors(),
        help="Feature extractor"
    )
    parser.add_argument("--max_keypoints", type=int, default=4096, help="Max keypoints")
    parser.add_argument("--resize_max", type=int, default=None, help="Resize max dimension")

    # Matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="lightglue",
        choices=MatcherFactory.list_matchers(),
        help="Feature matcher"
    )
    parser.add_argument("--distance_threshold", type=float, default=0.8, help="Distance threshold (0.8 for SIFT/ORB, 0.7 for learned)")

    # Pair selector (optional - if not specified, uses all dataset pairs)
    parser.add_argument(
        "--pair_selector",
        type=str,
        default=None,
        help="Pair selector (if specified, overrides dataset pairs)"
    )
    parser.add_argument("--max_pairs_per_image", type=int, default=20, help="Max pairs per image (for pair selector)")

    # Evaluation
    parser.add_argument("--epipolar_threshold", type=float, default=1.0, help="Epipolar threshold (px)")
    parser.add_argument("--ransac_threshold", type=float, default=1.0, help="RANSAC threshold (px)")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples")

    # MAGSAC Filtering (same as sfm_pipeline for realistic evaluation)
    parser.add_argument(
        "--use_magsac_filtering",
        action="store_true",
        help="Apply MAGSAC filtering (same as sfm_pipeline) for realistic SfM evaluation"
    )
    parser.add_argument(
        "--magsac_threshold",
        type=float,
        default=2.0,
        help="MAGSAC reprojection threshold in pixels (default: 2.0, strict: 0.8)"
    )
    parser.add_argument(
        "--magsac_confidence",
        type=float,
        default=0.999,
        help="MAGSAC confidence level (default: 0.999, strict: 0.9999)"
    )
    parser.add_argument(
        "--magsac_max_iters",
        type=int,
        default=1000,
        help="MAGSAC maximum iterations (default: 1000, strict: 5000)"
    )

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    return parser.parse_args()


def run_benchmark(args) -> Dict:
    """Run benchmark evaluation.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary of aggregated results
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Load Dataset ==========
    logger.info(f"Loading {args.dataset} dataset from {args.dataset_path}...")
    dataset_kwargs = {"split": args.split}
    if args.dataset == "colmap_scene":
        dataset_kwargs["min_shared_points"] = args.colmap_min_shared_points

    dataset = DatasetFactory.create(args.dataset, args.dataset_path, **dataset_kwargs)

    if args.num_samples:
        dataset.samples = dataset.samples[:args.num_samples]

    # Build ground truth lookup: (path0, path1) -> DatasetSample
    ground_truth_lookup = {}
    for sample in dataset.samples:
        key = (sample.image0, sample.image1)
        ground_truth_lookup[key] = sample
        # Also add reverse pair for symmetric lookup
        key_reverse = (sample.image1, sample.image0)
        if key_reverse not in ground_truth_lookup:
            # Create reversed sample with inverted transformation
            T_1to0 = np.linalg.inv(sample.T_0to1)
            from testing.datasets import DatasetSample
            reversed_sample = DatasetSample(
                image0=sample.image1,
                image1=sample.image0,
                K0=sample.K1.copy(),
                K1=sample.K0.copy(),
                T_0to1=T_1to0,
                depth0=sample.depth1,
                depth1=sample.depth0,
                metadata=sample.metadata,
            )
            ground_truth_lookup[key_reverse] = reversed_sample

    # If pair_selector is specified, use it to select pairs
    if args.pair_selector:
        logger.info(f"Using pair selector: {args.pair_selector}")
        # Get unique images from dataset
        unique_images = sorted(
            {sample.image0 for sample in dataset.samples}.union({sample.image1 for sample in dataset.samples}),
            key=lambda p: str(p)
        )
        logger.info(f"Found {len(unique_images)} unique images in dataset")

        # We'll extract features first, then use pair selector
        # (features will be extracted in the next section)
        selected_pairs_mode = True
    else:
        selected_pairs_mode = False

    samples = dataset.samples
    total_samples_dataset = len(samples)
    logger.info(f"Loaded {total_samples_dataset} ground truth pairs from dataset")

    # ========== Setup Extractor ==========
    logger.info(f"Setting up {args.extractor} extractor...")
    extractor_config = ExtractorConfig(
        max_keypoints=args.max_keypoints,
        resize_max=args.resize_max,
        device=args.device,
    )
    extractor = ExtractorFactory.create(args.extractor, extractor_config)

    # Cache features once per unique image to avoid redundant extraction
    if selected_pairs_mode:
        unique_image_paths = unique_images
    else:
        unique_image_paths = sorted(
            {sample.image0 for sample in samples}.union({sample.image1 for sample in samples}),
            key=lambda p: str(p)
        )
    num_unique_images = len(unique_image_paths)

    logger.info(
        "Extracting features for %d unique images (feature cache)...",
        num_unique_images,
    )
    features_cache = extractor.extract_batch(unique_image_paths) if num_unique_images else {}

    missing_feature_images = {
        path for path in unique_image_paths if path not in features_cache
    }
    if missing_feature_images:
        logger.warning(
            "Feature cache missing %d images; pairs referencing them will be skipped.",
            len(missing_feature_images),
        )

    # ========== Pair Selection (if specified) ==========
    if selected_pairs_mode:
        logger.info(f"Selecting pairs with {args.pair_selector}...")
        selector_config = PairSelectorConfig(
            max_pairs_per_image=args.max_pairs_per_image,
            device=args.device,
        )
        pair_selector = PairSelectorFactory.create(args.pair_selector, selector_config)

        # Select pairs using pair selector
        selected_pairs = pair_selector.select(unique_image_paths, features=features_cache)
        logger.info(f"Pair selector chose {len(selected_pairs)} pairs")

        # Filter to only pairs with ground truth
        samples = []
        for img0, img1 in selected_pairs:
            key = (img0, img1)
            if key in ground_truth_lookup:
                samples.append(ground_truth_lookup[key])
            else:
                # Try reverse
                key_reverse = (img1, img0)
                if key_reverse in ground_truth_lookup:
                    samples.append(ground_truth_lookup[key_reverse])

        logger.info(f"Filtered to {len(samples)} pairs with ground truth (out of {len(selected_pairs)} selected)")
        logger.info(f"Coverage: {len(samples)}/{total_samples_dataset} dataset pairs ({100*len(samples)/total_samples_dataset:.1f}%)")

    total_samples = len(samples)
    logger.info(f"Evaluating {total_samples} pairs")

    # ========== Setup Matcher ==========
    logger.info(f"Setting up {args.matcher} matcher...")
    matcher_config = MatcherConfig(
        distance_threshold=args.distance_threshold,
        mutual_check=True,
        device=args.device,
    )
    matcher = MatcherFactory.create(
        args.matcher,
        matcher_config,
        extractor.get_config_dict()
    )

    # ========== Run Evaluation ==========
    logger.info("Running evaluation...")

    results = []
    match_metrics_all = []
    pose_metrics_all = []

    processing_stats = {
        "skipped_missing_features": 0,
        "skipped_empty_features": 0,
        "matching_failures": 0,
    }
    warned_missing_images = set()

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating", total=total_samples)):
        try:
            features0 = features_cache.get(sample.image0)
            features1 = features_cache.get(sample.image1)

            if features0 is None or features1 is None:
                processing_stats["skipped_missing_features"] += 1
                missing_paths = [
                    path
                    for path in (sample.image0, sample.image1)
                    if path not in features_cache and path not in warned_missing_images
                ]
                for path in missing_paths:
                    logger.warning(
                        "Missing cached features for %s; skipping related pairs.",
                        path.name,
                    )
                    warned_missing_images.add(path)
                continue

            if len(features0) == 0 or len(features1) == 0:
                processing_stats["skipped_empty_features"] += 1
                continue

            # Match
            matches = matcher.match(features0, features1)

            if len(matches.matches0) == 0:
                continue

            # Convert to numpy arrays
            match_indices = np.stack([matches.matches0, matches.matches1], axis=1)

            # Apply MAGSAC filtering if requested (same as sfm_pipeline)
            num_raw_matches = len(match_indices)
            if args.use_magsac_filtering and num_raw_matches > 0:
                match_indices = filter_matches_with_magsac(
                    features0.keypoints,
                    features1.keypoints,
                    match_indices,
                    threshold=args.magsac_threshold,
                    confidence=args.magsac_confidence,
                    max_iters=args.magsac_max_iters,
                )
                num_filtered_matches = len(match_indices)
                logger.debug(
                    f"MAGSAC filtering: {num_raw_matches} -> {num_filtered_matches} matches "
                    f"({num_filtered_matches / num_raw_matches * 100:.1f}% retained)"
                )

                if len(match_indices) == 0:
                    logger.warning(
                        f"No matches after MAGSAC filtering for sample {idx}: "
                        f"{sample.image0.name} <-> {sample.image1.name}"
                    )
                    continue

            # Load depth maps if available
            depth0 = None
            depth1 = None
            if sample.depth0 and sample.depth1:
                import cv2
                depth0 = cv2.imread(str(sample.depth0), cv2.IMREAD_UNCHANGED)
                depth1 = cv2.imread(str(sample.depth1), cv2.IMREAD_UNCHANGED)

            # Evaluate matches
            match_metrics = evaluate_matches(
                features0.keypoints,
                features1.keypoints,
                match_indices,
                sample.K0,
                sample.K1,
                sample.T_0to1,
                depth0=depth0,
                depth1=depth1,
                epipolar_threshold=args.epipolar_threshold,
            )

            # Evaluate pose
            pose_metrics = evaluate_pose(
                features0.keypoints,
                features1.keypoints,
                match_indices,
                sample.K0,
                sample.K1,
                sample.T_0to1,
                ransac_threshold=args.ransac_threshold,
            )

            # Store results
            result = {
                "sample_idx": idx,
                "image0": str(sample.image0.name),
                "image1": str(sample.image1.name),
                "metadata": sample.metadata,
                "match_metrics": match_metrics,
                "pose_metrics": pose_metrics,
            }
            results.append(result)

            match_metrics_all.append(match_metrics)
            pose_metrics_all.append(pose_metrics)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            processing_stats["matching_failures"] += 1
            continue

    # ========== Aggregate Results ==========
    logger.info("Aggregating results...")

    aggregated = aggregate_results(match_metrics_all, pose_metrics_all)

    # ========== Save Results ==========
    # Save detailed results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {results_file}")

    # Save summary
    summary_file = output_dir / "summary.json"
    summary = {
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "num_samples": len(results),
        "total_samples": total_samples,
        "total_samples_dataset": total_samples_dataset,
        "extractor": args.extractor,
        "matcher": args.matcher,
        "pair_selector": args.pair_selector,
        "config": {
            "max_keypoints": args.max_keypoints,
            "distance_threshold": args.distance_threshold,
            "max_pairs_per_image": args.max_pairs_per_image if args.pair_selector else None,
            "epipolar_threshold": args.epipolar_threshold,
            "ransac_threshold": args.ransac_threshold,
            "use_magsac_filtering": args.use_magsac_filtering,
            "magsac_threshold": args.magsac_threshold if args.use_magsac_filtering else None,
            "magsac_confidence": args.magsac_confidence if args.use_magsac_filtering else None,
            "magsac_max_iters": args.magsac_max_iters if args.use_magsac_filtering else None,
        },
        "cache": {
            "unique_images": num_unique_images,
            "cached_images": len(features_cache),
            "missing_images": len(missing_feature_images),
        },
        "processing": processing_stats,
        "metrics": aggregated,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    # Save summary CSV
    summary_csv_file = output_dir / "summary.csv"
    write_summary_csv(summary, summary_csv_file)
    logger.info(f"Saved summary CSV to {summary_csv_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Extractor: {args.extractor}")
    print(f"Matcher: {args.matcher}")
    if args.pair_selector:
        print(f"Pair Selector: {args.pair_selector} (max_pairs_per_image={args.max_pairs_per_image})")
        print(f"Coverage: {len(results)}/{total_samples_dataset} dataset pairs ({100*len(results)/total_samples_dataset:.1f}%)")
    else:
        print(f"Pair Selector: None (using all dataset pairs)")
    if args.use_magsac_filtering:
        print(f"MAGSAC Filtering: ENABLED (threshold={args.magsac_threshold}px, conf={args.magsac_confidence})")
    else:
        print(f"MAGSAC Filtering: DISABLED (raw matching evaluation)")
    print(f"Samples: {len(results)} / {total_samples} (processed / total)")
    print("-" * 60)
    print("Match Metrics:")
    print(f"  Avg Matches: {aggregated['avg_num_matches']:.1f}")
    print(f"  Avg Epipolar Error: {aggregated['avg_epipolar_error']:.2f} px")
    print(f"  Median Epipolar Error: {aggregated['median_epipolar_error']:.2f} px")
    print(f"  Inlier Ratio: {aggregated['avg_inlier_ratio']:.2%}")
    print("-" * 60)
    print("Pose Metrics:")
    print(f"  Success Rate: {aggregated['pose_success_rate']:.2%}")
    print(f"  Avg Rotation Error: {aggregated['avg_rotation_error']:.2f}°")
    print(f"  Avg Translation Error: {aggregated['avg_translation_error']:.2f}°")
    print(f"  Median Rotation Error: {aggregated['median_rotation_error']:.2f}°")
    print(f"  Median Translation Error: {aggregated['median_translation_error']:.2f}°")
    print(f"  AUC@5°: {aggregated['auc_5deg']:.2%}")
    print(f"  AUC@10°: {aggregated['auc_10deg']:.2%}")
    print(f"  AUC@20°: {aggregated['auc_20deg']:.2%}")
    print("=" * 60 + "\n")

    return summary


def aggregate_results(
    match_metrics_all: List[Dict],
    pose_metrics_all: List[Dict]
) -> Dict:
    """Aggregate metrics across all samples.

    Args:
        match_metrics_all: List of match metrics for each sample
        pose_metrics_all: List of pose metrics for each sample

    Returns:
        Dictionary of aggregated metrics
    """
    aggregated = {}

    # Match metrics
    if match_metrics_all:
        aggregated["avg_num_matches"] = np.mean([m["num_matches"] for m in match_metrics_all])

        epi_errors = [m["epipolar_error"] for m in match_metrics_all if m["epipolar_error"] != float("inf")]
        aggregated["avg_epipolar_error"] = np.mean(epi_errors) if epi_errors else float("inf")

        med_epi_errors = [m["median_epipolar_error"] for m in match_metrics_all if "median_epipolar_error" in m]
        aggregated["median_epipolar_error"] = np.median(med_epi_errors) if med_epi_errors else float("inf")

        aggregated["avg_inlier_ratio"] = np.mean([m["inlier_ratio"] for m in match_metrics_all])

    # Pose metrics
    if pose_metrics_all:
        successful = [m for m in pose_metrics_all if m["pose_estimated"]]
        aggregated["pose_success_rate"] = len(successful) / len(pose_metrics_all)

        if successful:
            rot_errors = [m["rotation_error"] for m in successful]
            trans_errors = [m["translation_error"] for m in successful]

            aggregated["avg_rotation_error"] = np.mean(rot_errors)
            aggregated["avg_translation_error"] = np.mean(trans_errors)
            aggregated["median_rotation_error"] = np.median(rot_errors)
            aggregated["median_translation_error"] = np.median(trans_errors)

            # AUC metrics (area under curve for pose error thresholds)
            aggregated["auc_5deg"] = compute_auc(rot_errors, trans_errors, threshold=5)
            aggregated["auc_10deg"] = compute_auc(rot_errors, trans_errors, threshold=10)
            aggregated["auc_20deg"] = compute_auc(rot_errors, trans_errors, threshold=20)
        else:
            aggregated["avg_rotation_error"] = float("inf")
            aggregated["avg_translation_error"] = float("inf")
            aggregated["median_rotation_error"] = float("inf")
            aggregated["median_translation_error"] = float("inf")
            aggregated["auc_5deg"] = 0.0
            aggregated["auc_10deg"] = 0.0
            aggregated["auc_20deg"] = 0.0

    return aggregated


def compute_auc(
    rot_errors: List[float],
    trans_errors: List[float],
    threshold: float
) -> float:
    """Compute AUC (area under curve) for pose error.

    A sample is considered correct if both rotation and translation
    errors are below the threshold.

    Args:
        rot_errors: List of rotation errors in degrees
        trans_errors: List of translation errors in degrees
        threshold: Error threshold in degrees

    Returns:
        AUC (fraction of samples below threshold)
    """
    correct = [
        (r < threshold and t < threshold)
        for r, t in zip(rot_errors, trans_errors)
    ]
    return np.mean(correct)


def write_summary_csv(summary: Dict, csv_path: Path) -> None:
    """Write aggregated summary to a single-row CSV."""
    import csv

    header = [
        "dataset",
        "dataset_path",
        "num_samples",
        "total_samples",
        "total_samples_dataset",
        "extractor",
        "matcher",
        "pair_selector",
        "max_keypoints",
        "max_pairs_per_image",
        "distance_threshold",
        "epipolar_threshold",
        "ransac_threshold",
        "use_magsac_filtering",
        "magsac_threshold",
        "magsac_confidence",
        "magsac_max_iters",
    ]

    metric_fields = sorted(summary["metrics"].keys())
    header.extend(metric_fields)

    row = [
        summary["dataset"],
        summary["dataset_path"],
        summary["num_samples"],
        summary.get("total_samples", summary["num_samples"]),
        summary.get("total_samples_dataset", summary.get("total_samples", summary["num_samples"])),
        summary["extractor"],
        summary["matcher"],
        summary.get("pair_selector", None),
        summary["config"]["max_keypoints"],
        summary["config"].get("max_pairs_per_image", None),
        summary["config"]["distance_threshold"],
        summary["config"]["epipolar_threshold"],
        summary["config"]["ransac_threshold"],
        summary["config"]["use_magsac_filtering"],
        summary["config"]["magsac_threshold"],
        summary["config"]["magsac_confidence"],
        summary["config"]["magsac_max_iters"],
    ]
    for key in metric_fields:
        value = summary["metrics"][key]
        if isinstance(value, float) and np.isinf(value):
            value = "inf"
        row.append(value)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(row)


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
