#!/usr/bin/env python3
"""
SfM Reconstruction Pipeline using matching_framework components.

This script performs complete Structure-from-Motion reconstruction on a folder
of images using modular extractors, matchers, and pair selectors from
matching_framework. It collects reconstruction statistics (even without ground truth)
and exports results to CSV for comparison.

Usage:
    python -m matching_framework.sfm_pipeline \
        --images data/images \
        --output results/reconstruction \
        --extractor superpoint \
        --matcher lightglue \
        --pair_selector mpa \
        --device cuda

Compare different configurations:
    # Run 1: SuperPoint + LightGlue + MPA
    python -m matching_framework.sfm_pipeline \
        --images data/temple \
        --output results/temple_sp_lg \
        --extractor superpoint --matcher lightglue --pair_selector mpa

    # Run 2: SIFT + NN + Exhaustive
    python -m matching_framework.sfm_pipeline \
        --images data/temple \
        --output results/temple_sift_nn \
        --extractor sift --matcher nn --pair_selector exhaustive

    # Compare results
    python -m matching_framework.testing.compare_reconstructions \
        results/temple_sp_lg/statistics.csv \
        results/temple_sift_nn/statistics.csv
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For sfm module

from extractors import ExtractorFactory, ExtractorConfig
from matchers import MatcherFactory, MatcherConfig
from pair_selectors import PairSelectorFactory, PairSelectorConfig

# Import COLMAP reconstruction functions from existing sfm module
from sfm.core.colmap_binary import (
    colmap_binary_reconstruction,
    read_colmap_binary_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SfM reconstruction pipeline using matching_framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    parser.add_argument("--images", type=str, required=True, help="Input images directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    # Feature extractor
    parser.add_argument(
        "--extractor",
        type=str,
        default="superpoint",
        choices=ExtractorFactory.list_extractors(),
        help="Feature extractor"
    )
    parser.add_argument("--max_keypoints", type=int, default=4096, help="Max keypoints per image")
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

    # Pair selector
    parser.add_argument(
        "--pair_selector",
        type=str,
        default="exhaustive",
        choices=PairSelectorFactory.list_selectors(),
        help="Pair selector"
    )
    parser.add_argument("--max_pairs_per_image", type=int, default=20, help="Max pairs per image")

    # MAGSAC filtering (optional)
    parser.add_argument(
        "--use_magsac_filtering",
        action="store_true",
        help="Apply MAGSAC filtering before reconstruction"
    )
    parser.add_argument("--magsac_threshold", type=float, default=2.0, help="MAGSAC threshold (px)")
    parser.add_argument("--magsac_confidence", type=float, default=0.999, help="MAGSAC confidence")
    parser.add_argument("--magsac_max_iters", type=int, default=1000, help="MAGSAC max iterations")

    # COLMAP reconstruction parameters
    parser.add_argument(
        "--colmap_executable",
        type=str,
        default="colmap",
        help="COLMAP executable path"
    )
    parser.add_argument(
        "--skip_reconstruction",
        action="store_true",
        help="Skip COLMAP reconstruction (only extract features and match)"
    )
    parser.add_argument(
        "--colmap_timeout",
        type=int,
        default=1200,
        help="COLMAP mapper timeout in seconds (default: 1200 = 20 minutes)"
    )
    parser.add_argument(
        "--colmap_min_matches",
        type=int,
        default=None,
        help="COLMAP minimum number of matches (default: auto-select based on matcher)"
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


def run_colmap_command(cmd: List[str], description: str) -> Tuple[bool, float]:
    """Run a COLMAP command and measure time.

    Args:
        cmd: Command to run
        description: Description for logging

    Returns:
        (success, elapsed_time)
    """
    logger.info(f"Running COLMAP: {description}...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        elapsed = time.time() - start_time
        logger.info(f"✓ {description} completed in {elapsed:.2f}s")
        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} failed: {e.stderr}")
        return False, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} timed out after {elapsed:.2f}s")
        return False, elapsed


def filter_matches_with_magsac(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    match_indices: np.ndarray,
    threshold: float = 2.0,
    confidence: float = 0.999,
    max_iters: int = 1000,
) -> np.ndarray:
    """Filter matches using MAGSAC."""
    if len(match_indices) < 8:
        return np.array([]).reshape(0, 2)

    matched_kpts0 = kpts0[match_indices[:, 0]]
    matched_kpts1 = kpts1[match_indices[:, 1]]

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

    inlier_mask = inlier_mask.ravel().astype(bool)

    if inlier_mask.sum() < 8:
        return np.array([]).reshape(0, 2)

    return match_indices[inlier_mask]


def convert_to_colmap_format(
    features_dict: Dict,
    matches_dict: Dict,
) -> Tuple[Dict, Dict]:
    """Convert matching_framework format to colmap_binary.py format.

    Args:
        features_dict: Dict[Path, FeatureData] from matching_framework
        matches_dict: Dict[(Path, Path), MatchData] from matching_framework

    Returns:
        (features, matches) in colmap_binary.py format
    """
    # Convert features
    features_colmap = {}
    for img_path, feat_data in features_dict.items():
        features_colmap[str(img_path)] = {
            'keypoints': feat_data.keypoints,
            'descriptors': feat_data.descriptors,
            'image_shape': feat_data.image_shape,
        }

    # Convert matches
    matches_colmap = {}
    for (img0, img1), match_data in matches_dict.items():
        if len(match_data.matches0) == 0:
            continue

        matches_colmap[(str(img0), str(img1))] = {
            'matches0': match_data.matches0,
            'matches1': match_data.matches1,
            'mscores0': match_data.scores,
            'mscores1': match_data.scores,
        }

    return features_colmap, matches_colmap


def parse_colmap_statistics(sparse_dir: Path) -> Dict:
    """Parse COLMAP reconstruction output and extract statistics.

    Reads cameras.txt, images.txt, points3D.txt and computes:
    - Number of registered images
    - Number of 3D points
    - Number of observations
    - Reprojection error statistics
    - Track length statistics

    Args:
        sparse_dir: COLMAP sparse reconstruction directory

    Returns:
        Dictionary of statistics
    """
    stats = {
        "num_registered_images": 0,
        "num_3d_points": 0,
        "num_observations": 0,
        "mean_reprojection_error": None,
        "median_reprojection_error": None,
        "mean_track_length": None,
        "median_track_length": None,
    }

    cameras_file = sparse_dir / "cameras.txt"
    images_file = sparse_dir / "images.txt"
    points3d_file = sparse_dir / "points3D.txt"

    if not images_file.exists() or not points3d_file.exists():
        logger.warning(f"COLMAP output files not found in {sparse_dir}")
        return stats

    # Parse images.txt
    num_registered = 0
    with open(images_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # Followed by: POINTS2D[] as (X, Y, POINT3D_ID)
            if not line.startswith(" "):  # Image line (not points2D line)
                num_registered += 1

    stats["num_registered_images"] = num_registered // 2  # Each image has 2 lines

    # Parse points3D.txt
    track_lengths = []
    reproj_errors = []

    with open(points3d_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            point3d_id = int(parts[0])
            error = float(parts[7])
            track = parts[8:]  # Remaining are image_id, point2d_idx pairs

            track_length = len(track) // 2
            track_lengths.append(track_length)
            reproj_errors.append(error)

    stats["num_3d_points"] = len(track_lengths)
    stats["num_observations"] = sum(track_lengths)

    if reproj_errors:
        stats["mean_reprojection_error"] = float(np.mean(reproj_errors))
        stats["median_reprojection_error"] = float(np.median(reproj_errors))

    if track_lengths:
        stats["mean_track_length"] = float(np.mean(track_lengths))
        stats["median_track_length"] = float(np.median(track_lengths))

    return stats


def save_statistics_csv(
    stats: Dict,
    output_file: Path,
) -> None:
    """Save statistics to CSV file.

    Args:
        stats: Dictionary of statistics
        output_file: Output CSV file path
    """
    import csv

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Metric", "Value"])

        # Configuration
        writer.writerow(["Configuration", ""])
        for key in ["extractor", "matcher", "pair_selector", "device"]:
            if key in stats:
                writer.writerow([key, stats[key]])

        writer.writerow(["", ""])

        # Feature/Matching statistics
        writer.writerow(["Feature Extraction & Matching", ""])
        for key in [
            "num_images",
            "num_pairs",
            "num_matches_total",
            "avg_matches_per_pair",
            "feature_extraction_time",
            "matching_time",
        ]:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                writer.writerow([key, value])

        writer.writerow(["", ""])

        # Reconstruction statistics
        writer.writerow(["Reconstruction Statistics", ""])
        for key in [
            "num_registered_images",
            "registration_rate",
            "num_3d_points",
            "num_observations",
            "mean_reprojection_error",
            "median_reprojection_error",
            "mean_track_length",
            "median_track_length",
            "reconstruction_time",
            "total_time",
        ]:
            if key in stats:
                value = stats[key]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                writer.writerow([key, value])

    logger.info(f"Statistics saved to {output_file}")


def run_reconstruction_pipeline(args) -> Dict:
    """Run complete SfM reconstruction pipeline.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary of statistics
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images)
    if not images_dir.exists():
        raise ValueError(f"Images directory does not exist: {images_dir}")

    # Get image list
    image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    image_list = sorted([
        p for p in images_dir.iterdir()
        if p.suffix in image_extensions
    ])

    if len(image_list) == 0:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_list)} images")

    # Initialize statistics
    stats = {
        "extractor": args.extractor,
        "matcher": args.matcher,
        "pair_selector": args.pair_selector,
        "device": args.device,
        "num_images": len(image_list),
    }

    total_start_time = time.time()

    # ========== 1. Feature Extraction ==========
    logger.info(f"Extracting features with {args.extractor}...")
    extractor_config = ExtractorConfig(
        max_keypoints=args.max_keypoints,
        resize_max=args.resize_max,
        device=args.device,
    )
    extractor = ExtractorFactory.create(args.extractor, extractor_config)

    feature_start_time = time.time()
    features_dict = extractor.extract_batch(image_list)
    stats["feature_extraction_time"] = time.time() - feature_start_time

    # Filter out images with too few features
    min_features = 10  # Minimum features for reconstruction
    filtered_features_dict = {}
    removed_count = 0

    for img_path, features in features_dict.items():
        if len(features) >= min_features:
            filtered_features_dict[img_path] = features
        else:
            logger.warning(f"Removing {img_path.name}: only {len(features)} features (min: {min_features})")
            removed_count += 1

    features_dict = filtered_features_dict
    stats["num_images_with_features"] = len(features_dict)
    stats["num_images_removed"] = removed_count

    if len(features_dict) == 0:
        logger.error("No images with sufficient features. Aborting.")
        return stats

    logger.info(f"✓ Features extracted in {stats['feature_extraction_time']:.2f}s")
    logger.info(f"  {len(features_dict)}/{len(image_list)} images have sufficient features (>={min_features})")

    # ========== 2. Pair Selection ==========
    logger.info(f"Selecting pairs with {args.pair_selector}...")
    selector_config = PairSelectorConfig(
        max_pairs_per_image=args.max_pairs_per_image,
        device=args.device,
    )
    pair_selector = PairSelectorFactory.create(args.pair_selector, selector_config)

    # Use only images with features for pair selection
    valid_image_list = list(features_dict.keys())
    pairs = pair_selector.select(valid_image_list, features=features_dict)
    stats["num_pairs"] = len(pairs)

    logger.info(f"✓ Selected {len(pairs)} pairs")

    # ========== 3. Feature Matching ==========
    logger.info(f"Matching features with {args.matcher}...")
    matcher_config = MatcherConfig(
        distance_threshold=args.distance_threshold,
        mutual_check=True,
        device=args.device,
    )

    learning_extractors = {"superpoint", "aliked", "disk"}
    use_gpu_bruteforce = (
        args.pair_selector == "exhaustive"
        and args.extractor in learning_extractors
        and args.matcher == "lightglue"
        and args.device == "cuda"
    )

    matches_dict = {}
    matching_start_time = time.time()

    if use_gpu_bruteforce:
        try:
            import torch
            from sfm.core.gpu_brute_force_matcher import GPUBruteForceMatcher
            from matchers.base import MatchData

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device not available for GPU brute-force matching")

            gpu_device = torch.device("cuda")
            logger.info("Using GPU brute-force LightGlue matcher for exhaustive pairs")

            # Prepare features for GPU matcher
            features_for_gpu = {}
            for img_path, feat in features_dict.items():
                features_for_gpu[str(img_path)] = {
                    "keypoints": np.asarray(feat.keypoints, dtype=np.float32),
                    "descriptors": np.asarray(feat.descriptors, dtype=np.float32),
                    "scores": np.asarray(feat.scores, dtype=np.float32),
                    "image_shape": feat.image_shape,
                }

            gpu_matcher = GPUBruteForceMatcher(
                device=gpu_device,
                feature_type=args.extractor,
                config={
                    "batch_size": 32,
                    "confidence_threshold": matcher_config.extra.get("confidence_threshold", 0.1)
                    if matcher_config.extra
                    else 0.1,
                },
            )

            gpu_matcher.load_features(features_for_gpu)
            pair_names = [(str(p0), str(p1)) for (p0, p1) in pairs]
            raw_matches = gpu_matcher.match_specific_pairs(pair_names)

            for (img0, img1) in pairs:
                key = (str(img0), str(img1))
                data = raw_matches.get(key)
                if data is None:
                    matches0 = np.array([], dtype=np.int32)
                    matches1 = np.array([], dtype=np.int32)
                    scores = np.array([], dtype=np.float32)
                else:
                    matches0 = data["matches0"].astype(np.int32)
                    matches1 = data["matches1"].astype(np.int32)
                    mscores = data.get("mscores0")
                    if mscores is None:
                        scores = np.ones(len(matches0), dtype=np.float32)
                    else:
                        scores = np.asarray(mscores, dtype=np.float32)

                matches_dict[(img0, img1)] = MatchData(
                    matches0=matches0,
                    matches1=matches1,
                    scores=scores,
                )

            gpu_matcher.clear_memory()
        except Exception as e:
            logger.warning(f"GPU brute-force matcher unavailable ({e}); falling back to standard matcher.")
            use_gpu_bruteforce = False

    if not use_gpu_bruteforce:
        matcher = MatcherFactory.create(
            args.matcher,
            matcher_config,
            extractor.get_config_dict()
        )
        matches_dict = matcher.match_pairs(features_dict, pairs)

    stats["matching_time"] = time.time() - matching_start_time

    logger.info(f"✓ Matching completed in {stats['matching_time']:.2f}s")

    # ========== 4. Optional MAGSAC Filtering ==========
    if args.use_magsac_filtering:
        logger.info("Applying MAGSAC filtering...")
        filtered_matches_dict = {}

        for (img0, img1), matches in matches_dict.items():
            if len(matches.matches0) == 0:
                continue

            match_indices = np.stack([matches.matches0, matches.matches1], axis=1)

            filtered_indices = filter_matches_with_magsac(
                features_dict[img0].keypoints,
                features_dict[img1].keypoints,
                match_indices,
                threshold=args.magsac_threshold,
                confidence=args.magsac_confidence,
                max_iters=args.magsac_max_iters,
            )

            if len(filtered_indices) > 0:
                from matchers.base import MatchData
                filtered_matches = MatchData(
                    matches0=filtered_indices[:, 0],
                    matches1=filtered_indices[:, 1],
                    scores=matches.scores[:len(filtered_indices)],
                )
                filtered_matches_dict[(img0, img1)] = filtered_matches

        logger.info(f"✓ MAGSAC filtering: {len(matches_dict)} → {len(filtered_matches_dict)} pairs")
        matches_dict = filtered_matches_dict

    # Compute match statistics
    num_matches_total = sum(len(m.matches0) for m in matches_dict.values())
    stats["num_matches_total"] = num_matches_total
    stats["avg_matches_per_pair"] = num_matches_total / len(matches_dict) if matches_dict else 0

    logger.info(f"Total matches: {num_matches_total}")
    logger.info(f"Avg matches per pair: {stats['avg_matches_per_pair']:.1f}")

    # ========== 5. Convert to COLMAP Format ==========
    logger.info("Converting to COLMAP format...")
    features_colmap, matches_colmap = convert_to_colmap_format(features_dict, matches_dict)

    # ========== 6. Run COLMAP Reconstruction ==========
    if not args.skip_reconstruction:
        logger.info("Running COLMAP reconstruction...")

        reconstruction_start_time = time.time()

        try:
            # Determine COLMAP parameters based on matcher/pair_selector
            # Exhaustive: speed-focused (many pairs, need fast processing)
            # MPA + learning: strict parameters (high quality matches)
            # Others: lenient parameters
            is_exhaustive = args.pair_selector == 'exhaustive'
            is_learning_based = args.extractor in ['superpoint', 'aliked', 'disk']
            is_mpa = args.pair_selector == 'mpa'

            if is_exhaustive:
                # Speed-focused parameters for exhaustive matching (many pairs!)
                colmap_params = {
                    'min_num_matches': args.colmap_min_matches if args.colmap_min_matches else 8,
                    'speed_mode': True,  # Use speed-focused BA iterations
                }
                logger.info("Using speed-focused COLMAP parameters (exhaustive matching)")
            elif is_learning_based and is_mpa:
                # Strict parameters for high-quality matches
                colmap_params = {
                    'min_num_matches': args.colmap_min_matches if args.colmap_min_matches else 15,
                    'abs_pose_min_inliers': 20,
                    'abs_pose_min_inlier_ratio': 0.20,
                    'init_min_tri_angle': 3.0,
                    'tri_min_angle': 1.5,
                }
                logger.info("Using strict COLMAP parameters (learning-based + MPA)")
            else:
                # Lenient parameters for other cases
                colmap_params = {
                    'min_num_matches': args.colmap_min_matches if args.colmap_min_matches else 10,
                    'abs_pose_min_inliers': 15,
                    'abs_pose_min_inlier_ratio': 0.15,
                    'init_min_tri_angle': 2.0,
                    'tri_min_angle': 1.0,
                }
                logger.info("Using lenient COLMAP parameters")

            # Run COLMAP reconstruction with appropriate parameters
            # Note: This function applies MAGSAC filtering internally
            logger.info(f"COLMAP parameters: {colmap_params}")
            logger.info(f"COLMAP timeout: {args.colmap_timeout}s")

            points3d, cameras, images = colmap_binary_reconstruction(
                features_colmap,
                matches_colmap,
                output_dir,
                images_dir,
                mapper_params=colmap_params,
                timeout=args.colmap_timeout
            )

            stats["reconstruction_time"] = time.time() - reconstruction_start_time

            # Extract reconstruction statistics
            stats["num_registered_images"] = len(images)
            stats["registration_rate"] = len(images) / len(image_list) if image_list else 0
            stats["num_3d_points"] = len(points3d)

            # Compute observations and track lengths
            if points3d:
                num_observations = 0
                track_lengths = []
                reproj_errors = []

                for point_id, point_data in points3d.items():
                    if isinstance(point_data, dict):
                        # Handle dict format from read_colmap_text_results
                        # track is a list of (img_id, point2d_idx) tuples
                        track = point_data.get('track', [])
                        track_len = len(track)
                        error = point_data.get('error', 0.0)
                    else:
                        # Handle namedtuple format (if used)
                        track_len = len(point_data.image_ids) if hasattr(point_data, 'image_ids') else 0
                        error = point_data.error if hasattr(point_data, 'error') else 0.0

                    if track_len > 0:  # Only count valid points
                        track_lengths.append(track_len)
                        num_observations += track_len
                        reproj_errors.append(error)

                stats["num_observations"] = num_observations
                if track_lengths:
                    stats["mean_track_length"] = float(np.mean(track_lengths))
                    stats["median_track_length"] = float(np.median(track_lengths))
                else:
                    stats["mean_track_length"] = 0.0
                    stats["median_track_length"] = 0.0

                if reproj_errors:
                    stats["mean_reprojection_error"] = float(np.mean(reproj_errors))
                    stats["median_reprojection_error"] = float(np.median(reproj_errors))
                else:
                    stats["mean_reprojection_error"] = 0.0
                    stats["median_reprojection_error"] = 0.0

            logger.info(f"✓ Reconstruction completed in {stats['reconstruction_time']:.2f}s")
            logger.info(f"  Registered images: {stats['num_registered_images']}/{len(image_list)}")
            logger.info(f"  3D points: {stats.get('num_3d_points', 0)}")
            logger.info(f"  Mean reprojection error: {stats.get('mean_reprojection_error', 0):.3f} px")

        except Exception as e:
            logger.error(f"COLMAP reconstruction failed: {e}")
            stats["reconstruction_time"] = time.time() - reconstruction_start_time
            stats["reconstruction_failed"] = True

    else:
        logger.info("Skipping reconstruction (--skip_reconstruction)")

    stats["total_time"] = time.time() - total_start_time

    # ========== 7. Save Statistics ==========
    # Save JSON
    json_file = output_dir / "statistics.json"
    with open(json_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved to {json_file}")

    # Save CSV
    csv_file = output_dir / "statistics.csv"
    save_statistics_csv(stats, csv_file)

    # Print summary
    print("\n" + "=" * 70)
    print("RECONSTRUCTION PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Extractor: {stats['extractor']}")
    print(f"  Matcher: {stats['matcher']}")
    print(f"  Pair Selector: {stats['pair_selector']}")
    print(f"  Device: {stats['device']}")
    print("-" * 70)
    print(f"Feature Extraction & Matching:")
    print(f"  Input images: {stats['num_images']}")
    print(f"  Images with features: {stats.get('num_images_with_features', stats['num_images'])}")
    print(f"  Images removed (insufficient features): {stats.get('num_images_removed', 0)}")
    print(f"  Pairs: {stats['num_pairs']}")
    print(f"  Total matches: {stats['num_matches_total']}")
    print(f"  Avg matches/pair: {stats['avg_matches_per_pair']:.1f}")
    print(f"  Feature extraction time: {stats['feature_extraction_time']:.2f}s")
    print(f"  Matching time: {stats['matching_time']:.2f}s")

    if 'num_registered_images' in stats:
        print("-" * 70)
        print(f"Reconstruction Statistics:")
        print(f"  Registered images: {stats['num_registered_images']}/{stats['num_images']} ({stats.get('registration_rate', 0)*100:.1f}%)")
        print(f"  3D points: {stats.get('num_3d_points', 0)}")
        print(f"  Observations: {stats.get('num_observations', 0)}")
        print(f"  Mean track length: {stats.get('mean_track_length', 0):.2f}")
        print(f"  Mean reprojection error: {stats.get('mean_reprojection_error', 0):.3f} px")
        print(f"  Median reprojection error: {stats.get('median_reprojection_error', 0):.3f} px")
        print(f"  Reconstruction time: {stats.get('reconstruction_time', 0):.2f}s")

    print("-" * 70)
    print(f"Total time: {stats['total_time']:.2f}s")
    print("=" * 70 + "\n")

    return stats


def main():
    args = parse_args()
    run_reconstruction_pipeline(args)


if __name__ == "__main__":
    main()
