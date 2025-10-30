#!/usr/bin/env python3
"""
Enhanced SfM Pipeline for 3D Gaussian Splatting
Optimized for high-quality camera poses
"""

import argparse
import logging
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from sfm.core.feature_extractor import FeatureExtractorFactory
from sfm.core.feature_matcher import EnhancedLightGlueMatcher
from sfm.core.algebraic_consensus import AlgebraicConsensus, convert_matches_to_correspondences
from sfm.core.gpu_vocabulary_tree import GPUVocabularyTree
from sfm.utils.io_utils import (
    save_colmap_format,
    load_images,
    save_features,
    save_matches,
)
from sfm.utils.image_utils import resize_image

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced SfM Pipeline for 3DGS")

    # Input/Output
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )

    # Feature extraction
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="aliked",
        choices=["superpoint", "aliked", "disk"],
        help="Feature extractor to use",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1600,
        help="Maximum image size for processing",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=4096,
        help="Maximum number of keypoints per image",
    )

    # Matching and verification
    parser.add_argument(
        "--use_brute_force",
        action="store_true",
        default=False,
        help="Use GPU brute force matching (default and recommended)",
    )
    parser.add_argument(
        "--use_vocab_tree",
        action="store_true",
        help="Use vocabulary tree for smart pair selection (for very large datasets)",
    )
    parser.add_argument(
        "--max_pairs_per_image",
        type=int,
        default=20,
        help="Maximum pairs per image for vocabulary tree",
    )
    parser.add_argument(
        "--max_total_pairs",
        type=int,
        default=None,
        help="Maximum total pairs for brute force matching",
    )
    parser.add_argument(
        "--consensus_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "ransac"],
        help="Algebraic consensus mode for geometric verification",
    )
    parser.add_argument(
        "--consensus_tau_deg",
        type=float,
        default=17.0,
        help="Orientation threshold in degrees for consensus filtering",
    )
    parser.add_argument(
        "--consensus_inlier_threshold",
        type=float,
        default=5.0,
        help="Inlier threshold in pixels for consensus verification",
    )
    parser.add_argument(
        "--consensus_min_inlier_ratio",
        type=float,
        default=0.2,
        help="Minimum inlier ratio required to keep a verified pair",
    )
    parser.add_argument(
        "--consensus_max_combos",
        type=int,
        default=500,
        help="Maximal minimal-set combinations in deterministic mode (<=0 for unlimited)",
    )
    parser.add_argument(
        "--consensus_min_correspondences",
        type=int,
        default=10,
        help="Minimum correspondences required before running consensus verification",
    )
    parser.add_argument(
        "--consensus_n_trials",
        type=int,
        default=100,
        help="Maximum RANSAC trials when consensus mode is 'ransac'",
    )
    parser.add_argument(
        "--consensus_confidence",
        type=float,
        default=0.99,
        help="RANSAC confidence level for consensus verification",
    )

    # Device and performance
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for feature extraction"
    )

    # Quality settings for 3DGS
    parser.add_argument(
        "--high_quality", action="store_true", help="Enable high-quality mode for 3DGS"
    )

    # 3DGS Integration
    parser.add_argument(
        "--copy_to_3dgs_dir",
        type=str,
        default=None,
        help="Directory to copy COLMAP sparse files for 3D Gaussian Splatting",
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup device for computation"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        logger.info(
            f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB"
        )

    return device


def cleanup_gpu_memory(device: torch.device, stage_name: str = ""):
    """Comprehensive GPU memory cleanup"""
    if device.type == "cuda":
        # Force garbage collection
        gc.collect()

        # Clear PyTorch CUDA cache
        torch.cuda.empty_cache()

        # Synchronize CUDA operations
        torch.cuda.synchronize()

        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2

        stage_info = f" after {stage_name}" if stage_name else ""
        logger.info(
            f"GPU memory{stage_info}: {memory_allocated:.1f} MB allocated, {memory_cached:.1f} MB cached"
        )


def setup_logging(output_dir: str):
    """Setup logging configuration"""
    log_file = Path(output_dir) / "sfm_pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def sfm_pipeline(input_dir: str = None, output_dir: str = None, **kwargs):
    """Enhanced SfM pipeline for 3DGS - Main API function"""

    # Handle both command line args and direct function calls
    if input_dir is None or output_dir is None:
        # Command line mode
        args = parse_args()
        input_dir = args.input_dir
        output_dir = args.output_dir
        device = setup_device(args.device)

        # Convert args to kwargs for consistency
        kwargs = {
            "feature_extractor": args.feature_extractor,
            "max_keypoints": args.max_keypoints,
            "max_image_size": args.max_image_size,
            "use_brute_force": args.use_brute_force,
            "use_vocab_tree": args.use_vocab_tree,
            "max_pairs_per_image": args.max_pairs_per_image,
            "max_total_pairs": args.max_total_pairs,
            "copy_to_3dgs_dir": args.copy_to_3dgs_dir,
            "high_quality": args.high_quality,
            "device": args.device,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "consensus_mode": args.consensus_mode,
            "consensus_tau_deg": args.consensus_tau_deg,
            "consensus_inlier_threshold": args.consensus_inlier_threshold,
            "consensus_min_inlier_ratio": args.consensus_min_inlier_ratio,
            "consensus_max_combos": args.consensus_max_combos,
            "consensus_min_correspondences": args.consensus_min_correspondences,
            "consensus_n_trials": args.consensus_n_trials,
            "consensus_confidence": args.consensus_confidence,
        }
    else:
        # Direct function call mode
        device = setup_device(kwargs.get("device", "auto"))

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_path))

    logger.info("=" * 60)
    logger.info("Enhanced SfM Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Feature extractor: {kwargs.get('feature_extractor', 'superpoint')}")
    logger.info(f"GPU brute force matching: {kwargs.get('use_brute_force', True)}")
    logger.info(f"High quality mode: {kwargs.get('high_quality', False)}")

    consensus_defaults = {
        "consensus_mode": "deterministic",
        "consensus_tau_deg": 17.0,
        "consensus_inlier_threshold": 5.0,
        "consensus_min_inlier_ratio": 0.2,
        "consensus_max_combos": 500,
        "consensus_min_correspondences": 10,
        "consensus_n_trials": 100,
        "consensus_confidence": 0.99,
        "consensus_use_closed_form": True,
        "consensus_use_orientation_filter": True,
        "consensus_min_inliers": 3,
    }
    for key, value in consensus_defaults.items():
        kwargs.setdefault(key, value)
    kwargs["consensus_mode"] = str(kwargs.get("consensus_mode", "deterministic")).lower()

    logger.info(
        "Algebraic consensus mode: %s (tau=%.1f°, inlier_threshold=%.2fpx, min_ratio=%.2f)",
        kwargs["consensus_mode"],
        kwargs.get("consensus_tau_deg", 17.0),
        kwargs.get("consensus_inlier_threshold", 5.0),
        kwargs.get("consensus_min_inlier_ratio", 0.2),
    )

    # Performance tracking
    start_time = time.time()
    stage_times = {}

    # Stage 1: Load and preprocess images
    logger.info("Stage 1: Loading and preprocessing images...")
    stage_start = time.time()

    image_paths = load_images(input_dir)
    logger.info(f"Found {len(image_paths)} images")

    # Resize images for processing
    processed_images = {}
    for img_path in tqdm(image_paths, desc="Preprocessing images"):
        img = resize_image(img_path, kwargs.get("max_image_size", 1600))
        processed_images[img_path] = img

    stage_times["preprocessing"] = time.time() - stage_start
    logger.info(f"Preprocessing completed in {stage_times['preprocessing']:.2f}s")

    # Memory cleanup after preprocessing
    cleanup_gpu_memory(device, "preprocessing")

    # Stage 2: Feature extraction
    logger.info("Stage 2: Extracting features...")
    stage_start = time.time()

    # Check if features already exist
    features_file = output_path / "features.h5"
    features_tensor_file = output_path / "features_tensors.pt"

    # Try to load existing features from both H5 and tensor formats
    features = None

    if features is None and features_tensor_file.exists():
        try:
            existing_tensors = torch.load(features_tensor_file, map_location=device)
            if len(existing_tensors) == len(processed_images):
                logger.info(
                    f"Found existing tensor features for {len(existing_tensors)} images, using those"
                )
                # Convert tensor format to expected format
                features = {}
                for img_path, tensor_data in existing_tensors.items():
                    features[img_path] = {
                        "keypoints": tensor_data["keypoints"].cpu().numpy()
                        if torch.is_tensor(tensor_data["keypoints"])
                        else tensor_data["keypoints"],
                        "descriptors": tensor_data["descriptors"].cpu().numpy()
                        if torch.is_tensor(tensor_data["descriptors"])
                        else tensor_data["descriptors"],
                        "scores": tensor_data["scores"].cpu().numpy()
                        if torch.is_tensor(tensor_data["scores"])
                        else tensor_data["scores"],
                        "image_shape": tensor_data["image_shape"],
                    }
                features_tensors = existing_tensors
                stage_times["feature_extraction"] = 0.0
            else:
                logger.info(
                    f"Tensor feature count mismatch: {len(existing_tensors)} vs {len(processed_images)}, re-extracting"
                )
                features = None
        except Exception as e:
            logger.info(f"Could not load tensor features ({e}), extracting new ones")
            features = None

    if features is None:
        feature_extractor = FeatureExtractorFactory.create(
            kwargs.get("feature_extractor", "superpoint"),
            device=device,
            config={
                "max_keypoints": kwargs.get("max_keypoints", 4096),
                "high_quality": kwargs.get("high_quality", True),
            },
        )

        # Prepare images in the format expected by extractors
        images_for_extraction = []
        for img_path, img_array in processed_images.items():
            images_for_extraction.append({"image": img_array, "path": img_path})

        features = feature_extractor.extract_features(
            images_for_extraction, batch_size=kwargs.get("batch_size", 8)
        )

        # Save features (traditional format)
        save_features(features, features_file)

        # Save features as tensors for backup and later use
        features_tensors = {}
        for img_path, feat_data in features.items():
            features_tensors[img_path] = {
                "keypoints": torch.from_numpy(feat_data["keypoints"]).to(device),
                "descriptors": torch.from_numpy(feat_data["descriptors"]).to(device),
                "scores": torch.from_numpy(feat_data["scores"]).to(device),
                "image_shape": feat_data["image_shape"],
            }
        torch.save(features_tensors, features_tensor_file)
        logger.info(f"Saved feature tensors to {features_tensor_file}")

        stage_times["feature_extraction"] = time.time() - stage_start
        logger.info(
            f"Feature extraction completed in {stage_times['feature_extraction']:.2f}s"
        )

        # Clean up feature extractor memory
        if "feature_extractor" in locals():
            try:
                if hasattr(feature_extractor, "model"):
                    del feature_extractor.model
                if hasattr(feature_extractor, "extractor"):
                    del feature_extractor.extractor
                del feature_extractor
            except Exception as e:
                logger.warning(f"Error cleaning up feature extractor: {e}")

        # Clean up large tensor data that's no longer needed
        if "images_for_extraction" in locals():
            del images_for_extraction

        cleanup_gpu_memory(device, "feature extraction")

    # Stage 3: Smart pair selection (vocabulary tree + sequential)
    logger.info("Stage 3: Smart pair selection (vocabulary tree + sequential)...")
    stage_start = time.time()

    # Always generate sequential pairs for temporal consistency
    sequential_overlap = 5  # Number of consecutive images to match
    sequential_pairs = []

    # Sort image paths for sequential ordering
    sorted_image_paths = sorted(image_paths)

    for i, img1 in enumerate(sorted_image_paths):
        # Match with next few images in sequence
        for j in range(1, min(sequential_overlap + 1, len(sorted_image_paths) - i)):
            img2 = sorted_image_paths[i + j]
            sequential_pairs.append((img1, img2))

    logger.info(f"Generated {len(sequential_pairs)} sequential pairs")

    if kwargs.get("use_vocab_tree", False):
        # Use vocabulary tree for additional similarity-based pairs
        vocab_tree = GPUVocabularyTree(
            device=device,
            config={
                "vocab_size": 10000,
                "vocab_depth": 6,
                "vocab_branching_factor": 10,
            },
            output_path=str(output_path),
        )

        # Build vocabulary
        vocab_tree.build_vocabulary(features)

        # Get vocabulary tree pairs
        vocab_tree_pairs = vocab_tree.get_image_pairs_for_matching(
            features, max_pairs_per_image=kwargs.get("max_pairs_per_image", 20)
        )

        # Combine sequential and vocabulary tree pairs (remove duplicates)
        all_pairs = list(set(sequential_pairs + vocab_tree_pairs))
        image_pairs = all_pairs

        logger.info(f"Selected {len(vocab_tree_pairs)} pairs using vocabulary tree")
        logger.info(
            f"Combined total: {len(image_pairs)} unique pairs (sequential + vocabulary tree)"
        )
    else:
        # Use only sequential pairs for smaller datasets
        image_pairs = sequential_pairs
        logger.info(f"Using sequential matching: {len(image_pairs)} pairs")

    stage_times["pair_selection"] = time.time() - stage_start
    logger.info(f"Pair selection completed in {stage_times['pair_selection']:.2f}s")

    # Clean up vocabulary tree memory if used
    if "vocab_tree" in locals():
        try:
            if hasattr(vocab_tree, "clear_memory"):
                vocab_tree.clear_memory()
            del vocab_tree
        except Exception as e:
            logger.warning(f"Error cleaning up vocabulary tree: {e}")

    cleanup_gpu_memory(device, "pair selection")

    # Stage 4: Feature matching
    logger.info("Stage 4: Feature matching...")
    stage_start = time.time()

    # Check if matches already exist
    matches_file = output_path / "matches.h5"
    matches_tensor_file = output_path / "matches_tensors.pt"

    # Calculate expected number of matches for validation
    expected_pairs = len(image_pairs)

    if matches_file.exists() and matches_tensor_file.exists():
        try:
            # Load existing matches and validate
            from sfm.utils.io_utils import load_matches

            existing_matches = load_matches(matches_file)
            existing_match_tensors = torch.load(
                matches_tensor_file, map_location=device
            )

            # Check if we have reasonable number of matches
            if (
                len(existing_matches) >= expected_pairs * 0.1
            ):  # At least 10% success rate
                logger.info(
                    f"Found existing matches for {len(existing_matches)} pairs (expected ~{expected_pairs}), skipping matching"
                )
                matches = existing_matches
                matches_tensors = existing_match_tensors
                stage_times["feature_matching"] = 0.0
            else:
                logger.info(
                    f"Match count too low: {len(existing_matches)} vs expected ~{expected_pairs}, re-matching"
                )
                raise ValueError("Match count too low")
        except Exception as e:
            logger.info(f"Could not load existing matches ({e}), matching new ones")
            matches = None
    else:
        matches = None

    if matches is None:
        feature_type = kwargs.get("feature_extractor", "superpoint")

        # Configure matcher based on vocabulary tree usage
        matcher_config = {
            "use_brute_force": kwargs.get("use_brute_force", True),
            "use_vocabulary_tree": kwargs.get("use_vocab_tree", False),
            "max_pairs_per_image": kwargs.get("max_pairs_per_image", 20),
            "max_total_pairs": kwargs.get("max_total_pairs", None),
            "output_path": str(output_path),
        }

        # If vocabulary tree was used, pass the selected pairs to the matcher
        if kwargs.get("use_vocab_tree", False) and "image_pairs" in locals():
            matcher_config["predefined_pairs"] = image_pairs
            matcher_config["use_brute_force"] = (
                False  # Force to use only predefined pairs
            )

        matcher = EnhancedLightGlueMatcher(
            device=device, feature_type=feature_type, config=matcher_config
        )

        # Ensure features are in the correct format for the matcher
        # If features were loaded from tensor file, ensure numpy format
        formatted_features = {}
        for img_path, feat_data in features.items():
            formatted_feat = {}
            for key, value in feat_data.items():
                if torch.is_tensor(value):
                    formatted_feat[key] = value.cpu().numpy()
                else:
                    formatted_feat[key] = value
            formatted_features[img_path] = formatted_feat

        # Use the enhanced matcher
        matches = matcher.match_features(formatted_features)

        # Create tensor version for backup
        matches_tensors = {}
        for pair, match_result in matches.items():
            matches_tensors[pair] = {
                "matches0": torch.from_numpy(match_result["matches0"]).to(device),
                "matches1": torch.from_numpy(match_result["matches1"]).to(device),
                "mscores0": torch.from_numpy(match_result["mscores0"]).to(device),
                "mscores1": torch.from_numpy(match_result["mscores1"]).to(device),
                "image_shape0": match_result["image_shape0"],
                "image_shape1": match_result["image_shape1"],
            }

        # Save matches (traditional format)
        save_matches(matches, matches_file)

        # Save matches as tensors for backup
        torch.save(matches_tensors, matches_tensor_file)
        logger.info(f"Saved match tensors to {matches_tensor_file}")

        stage_times["feature_matching"] = time.time() - stage_start
        logger.info(
            f"Feature matching completed in {stage_times['feature_matching']:.2f}s"
        )

        # Clean up matcher memory
        if "matcher" in locals():
            try:
                if hasattr(matcher, "clear_memory"):
                    matcher.clear_memory()
                if hasattr(matcher, "matcher") and hasattr(
                    matcher.matcher, "clear_memory"
                ):
                    matcher.matcher.clear_memory()
                del matcher
            except Exception as e:
                logger.warning(f"Error cleaning up matcher: {e}")

        # Clean up large tensor data
        if "matches_tensors" in locals():
            try:
                del matches_tensors
            except Exception as e:
                logger.warning(f"Error cleaning up match tensors: {e}")

        if "formatted_features" in locals():
            del formatted_features

        cleanup_gpu_memory(device, "feature matching")

    # Stage 4.5: Geometric verification with Algebraic Consensus
    logger.info("Stage 4.5: Geometric verification (Algebraic Consensus)...")
    stage_start = time.time()

    if not matches:
        logger.warning("No matches available for geometric verification; skipping stage.")
        stage_times["geometric_verification"] = 0.0
        logger.info("Geometric verification skipped (0 pairs to verify).")
    else:
        consensus_mode = kwargs.get("consensus_mode", "hybrid")
        tau_deg = kwargs.get("consensus_tau_deg", 17.0)
        inlier_threshold = kwargs.get("consensus_inlier_threshold", 5.0)
        min_ratio = kwargs.get("consensus_min_inlier_ratio", 0.1)
        max_combos_raw = kwargs.get("consensus_max_combos", 150)
        max_combos = None if max_combos_raw is None or max_combos_raw <= 0 else max_combos_raw
        min_correspondences = kwargs.get("consensus_min_correspondences", 10)
        use_orientation_filter = kwargs.get("consensus_use_orientation_filter", True)
        use_closed_form = kwargs.get("consensus_use_closed_form", True)
        n_trials = kwargs.get("consensus_n_trials", 100)
        confidence = kwargs.get("consensus_confidence", 0.99)
        min_inliers = kwargs.get("consensus_min_inliers", 3)
        hybrid_min_ratio = kwargs.get("consensus_hybrid_min_ratio", 0.1)

        logger.info(
            "Consensus config → mode=%s, tau=%.1f°, inlier_threshold=%.2fpx, "
            "min_ratio=%.2f, min_corr=%d, max_combos=%s, hybrid_fallback_ratio=%.2f",
            consensus_mode,
            tau_deg,
            inlier_threshold,
            min_ratio,
            min_correspondences,
            "unlimited" if max_combos is None else str(max_combos),
            hybrid_min_ratio,
        )

        verifier = AlgebraicConsensus(
            orientation_tau=np.radians(tau_deg),
            use_orientation_filter=use_orientation_filter,
            mode=consensus_mode,
            n_trials=n_trials,
            inlier_threshold=inlier_threshold,
            confidence=confidence,
            use_closed_form=use_closed_form,
            max_deterministic_combinations=max_combos,
            min_inliers=min_inliers,
            hybrid_min_ratio=hybrid_min_ratio,
        )

        def _build_kpt_dict(feat: Dict[str, Any]) -> Dict[str, Any]:
            data: Dict[str, Any] = {"keypoints": feat["keypoints"]}
            if "scores" in feat and feat["scores"] is not None:
                data["scores"] = feat["scores"]
            if "orientations" in feat and feat["orientations"] is not None:
                data["orientations"] = feat["orientations"]
            return data

        original_matches = matches
        total_pairs = len(original_matches)
        logger.info("Verifying %d matched pairs...", total_pairs)

        verified_matches: Dict[Any, Any] = {}
        verification_stats = {
            "total_pairs": total_pairs,
            "verified_pairs": 0,
            "avg_inlier_ratio": 0.0,
            "avg_runtime_ms": 0.0,
            "avg_orientation_std_deg": 0.0,
            "orientation_pairs": 0,
            "fallback_pairs": 0,
        }

        for pair, match_result in tqdm(original_matches.items(), desc="Verifying matches"):
            img1, img2 = pair
            feat1 = features[img1]
            feat2 = features[img2]

            matches0 = match_result["matches0"]
            valid_src_indices = np.nonzero(matches0 >= 0)[0]
            if valid_src_indices.size < min_correspondences:
                continue

            src_keypoints = feat1["keypoints"]
            dst_keypoints = feat2["keypoints"]
            src_len = src_keypoints.shape[0]
            dst_len = dst_keypoints.shape[0]

            hits = []
            for src_idx in valid_src_indices:
                if src_idx >= src_len:
                    continue
                dst_idx = matches0[src_idx]
                if 0 <= dst_idx < dst_len:
                    hits.append((int(src_idx), int(dst_idx)))

            if len(hits) < min_correspondences:
                continue

            correspondences = convert_matches_to_correspondences(
                hits,
                _build_kpt_dict(feat1),
                _build_kpt_dict(feat2),
            )

            if len(correspondences) < min_correspondences:
                continue

            result = verifier.verify_pair(
                correspondences,
                image_shape=feat1.get("image_shape"),
            )

            if (
                result.inlier_ratio >= min_ratio
                and result.n_inliers >= min_correspondences
            ):
                verified_matches[pair] = match_result
                verification_stats["verified_pairs"] += 1
                verification_stats["avg_inlier_ratio"] += result.inlier_ratio
                verification_stats["avg_runtime_ms"] += result.runtime * 1000.0
                if result.orientation_stats and result.orientation_stats.n_samples > 0:
                    verification_stats["avg_orientation_std_deg"] += np.degrees(
                        result.orientation_stats.std
                    )
                verification_stats["orientation_pairs"] += 1

                if result.method.startswith("hybrid"):
                    verification_stats["fallback_pairs"] += 1
            else:
                logger.debug(
                    "Rejected pair %s ↔ %s (ratio=%.3f, inliers=%d, certificate=%s)",
                    img1,
                    img2,
                    result.inlier_ratio,
                    result.n_inliers,
                    result.certificate,
                )

        if verification_stats["verified_pairs"] > 0:
            verification_stats["avg_inlier_ratio"] /= verification_stats["verified_pairs"]
            verification_stats["avg_runtime_ms"] /= verification_stats["verified_pairs"]
        if verification_stats["orientation_pairs"] > 0:
            verification_stats["avg_orientation_std_deg"] /= verification_stats["orientation_pairs"]

        kept = verification_stats["verified_pairs"]
        total = verification_stats["total_pairs"]
        logger.info("Verified %d/%d pairs (%.1f%%)", kept, total, (kept / total * 100.0) if total else 0.0)
        logger.info(
            "Average inlier ratio (kept): %.3f (threshold %.2f)",
            verification_stats["avg_inlier_ratio"],
            min_ratio,
        )
        logger.info(
            "Average verification time (kept): %.2f ms per pair",
            verification_stats["avg_runtime_ms"],
        )
        if verification_stats["verified_pairs"] > 0 and verification_stats["fallback_pairs"] > 0:
            logger.info(
                "Hybrid fallback triggered for %d kept pairs (%.1f%%)",
                verification_stats["fallback_pairs"],
                verification_stats["fallback_pairs"] / verification_stats["verified_pairs"] * 100.0,
            )
        if verification_stats["orientation_pairs"] > 0:
            logger.info(
                "Average orientation std among kept pairs: %.2f°",
                verification_stats["avg_orientation_std_deg"],
            )

        verifier_stats = verifier.get_statistics()
        if consensus_mode in {"deterministic", "hybrid"}:
            total_combos = verifier_stats.get("n_deterministic_combos", 0)
            call_count = verifier_stats.get("n_calls", 0)
            avg_combos = (total_combos / call_count) if call_count else 0.0
            logger.info(
                "Deterministic consensus combos tried: total=%d, avg=%.1f per call",
                total_combos,
                avg_combos,
            )
        if consensus_mode in {"hybrid", "ransac"}:
            ransac_calls = verifier_stats.get("n_ransac_calls", 0)
            if ransac_calls:
                logger.info("RANSAC verification calls: %d", ransac_calls)
        if consensus_mode == "hybrid":
            fallback_calls = verifier_stats.get("n_hybrid_fallbacks", 0)
            if fallback_calls:
                logger.info("Hybrid fallback triggered %d times", fallback_calls)

        if kept == 0:
            logger.warning(
                "No pairs passed algebraic consensus. Falling back to unfiltered matches."
            )
            matches = original_matches
        else:
            matches = verified_matches

        stage_times["geometric_verification"] = time.time() - stage_start
        logger.info(
            "Geometric verification completed in %.2fs",
            stage_times["geometric_verification"],
        )

    cleanup_gpu_memory(device, "geometric verification")

    # Stage 5: COLMAP-based SfM reconstruction using binary (avoid pycolmap CUDA issues)
    logger.info("Stage 5: COLMAP-based SfM reconstruction using binary...")
    stage_start = time.time()

    from sfm.core.colmap_binary import colmap_binary_reconstruction

    # Extract image directory from first image path
    first_image_path = Path(next(iter(features.keys())))
    image_dir = first_image_path.parent

    sparse_points, cameras, images = colmap_binary_reconstruction(
        features=features, matches=matches, output_path=output_path, image_dir=image_dir
    )

    stage_times["sfm_reconstruction"] = time.time() - stage_start
    logger.info(
        f"COLMAP SfM reconstruction completed in {stage_times['sfm_reconstruction']:.2f}s"
    )

    # Clean up reconstruction memory
    cleanup_gpu_memory(device, "SfM reconstruction")

    # Stage 9: Copy reconstruction files for 3DGS compatibility
    gs_input_dir = kwargs.get("copy_to_3dgs_dir")
    if gs_input_dir:
        logger.info("Stage 9: Preparing files for 3DGS...")
        stage_start = time.time()

        try:
            import shutil

            gs_input_path = Path(gs_input_dir)
            gs_sparse_dir = gs_input_path / "sparse" / "0"
            gs_sparse_dir.mkdir(parents=True, exist_ok=True)

            # Use original sparse reconstruction
            original_sparse_dir = output_path / "sparse" / "0"

            if original_sparse_dir.exists():
                source_sparse_dir = original_sparse_dir
                logger.info("Using original sparse reconstruction for 3DGS")
            else:
                logger.warning(
                    "No sparse reconstruction found - skipping 3DGS file preparation"
                )
                stage_times["3dgs_preparation"] = 0

            # Copy all COLMAP files (cameras.bin, images.bin, points3D.bin) if we have a source
            if "source_sparse_dir" in locals():
                for filename in ["cameras.bin", "images.bin", "points3D.bin"]:
                    src_file = source_sparse_dir / filename
                    if src_file.exists():
                        shutil.copy2(src_file, gs_sparse_dir / filename)
                        logger.info(f"Copied {filename} to 3DGS directory")
                    else:
                        logger.warning(
                            f"File {filename} not found in source sparse directory"
                        )

                # Copy images directory if it exists
                input_image_dir = Path(input_dir)
                gs_images_dir = gs_input_path / "images"
                if input_image_dir.exists():
                    if gs_images_dir.exists():
                        shutil.rmtree(gs_images_dir)
                    shutil.copytree(input_image_dir, gs_images_dir)
                    logger.info(
                        f"Copied {len(list(gs_images_dir.iterdir()))} images to 3DGS directory"
                    )

                logger.info(f" 3DGS files ready at: {gs_sparse_dir}")
                logger.info(f"   - Use with: python train.py -s {gs_input_path}")

        except Exception as e:
            logger.warning(f"Failed to prepare files for 3DGS: {e}")

        stage_times["3dgs_preparation"] = time.time() - stage_start
        logger.info(
            f"3DGS preparation completed in {stage_times['3dgs_preparation']:.2f}s"
        )

    # Stage 10: Save results in COLMAP format (for 3DGS)
    logger.info("Stage 10: Saving results...")
    stage_start = time.time()

    # Save in COLMAP format for 3DGS compatibility
    colmap_dir = output_path / "colmap"
    colmap_dir.mkdir(exist_ok=True)

    save_colmap_format(
        cameras=cameras,
        images=images,
        points3d=sparse_points,
        output_dir=str(colmap_dir),
    )

    stage_times["saving"] = time.time() - stage_start
    logger.info(f"Saving completed in {stage_times['saving']:.2f}s")

    # Final comprehensive memory cleanup
    cleanup_gpu_memory(device, "saving")

    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("ENHANCED SFM PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Number of images: {len(image_paths)}")
    logger.info(f"Number of 3D points: {len(sparse_points)}")
    logger.info(f"Number of cameras: {len(cameras)}")

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Ready for 3D Gaussian Splatting!")

    # Final cleanup of all large variables
    cleanup_variables = [
        "processed_images",
        "features",
        "features_tensors",
        "matches",
        "matches_tensors",
        "image_paths",
    ]

    for var_name in cleanup_variables:
        if var_name in locals():
            try:
                exec(f"del {var_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up {var_name}: {e}")

    # Final GPU memory cleanup
    cleanup_gpu_memory(device, "pipeline completion")

    # Log final memory state
    if device.type == "cuda":
        final_memory = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"Final GPU memory allocated: {final_memory:.1f} MB")

    return {
        "sparse_points": sparse_points,
        "cameras": cameras,
        "images": images,
        "features": None,  # Don't return large feature data to prevent memory retention
        "scale_info": None,  # Avoid returning scale_recovery reference
        "total_time": total_time,
        "stage_times": stage_times,
    }


def main():
    """Main entry point for command line usage"""
    return sfm_pipeline()


if __name__ == "__main__":
    main()
