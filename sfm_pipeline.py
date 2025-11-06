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

from sfm.core.feature_extractor import FeatureExtractorFactory
from sfm.core.feature_matcher import EnhancedLightGlueMatcher
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
        "--max_total_pairs",
        type=int,
        default=None,
        help="Maximum total pairs for brute force matching",
    )
    parser.add_argument(
        "--SARA_device",
        type=str,
        default="cuda",
        help="Device for SARA DINO embedding computation",
    )
    parser.add_argument(
        "--SARA_knn_k",
        type=int,
        default=30,
        help="Number of nearest neighbours per image for SARA candidate graph",
    )
    parser.add_argument(
        "--SARA_top_t_mutual",
        type=int,
        default=128,
        help="Mutual nearest descriptor count per pair for SARA fast pre-matching",
    )
    parser.add_argument(
        "--SARA_min_nn_for_ransac",
        type=int,
        default=32,
        help="Minimum mutual matches to run SARA mini-RANSAC",
    )
    parser.add_argument(
        "--SARA_loop_budget_per_node",
        type=float,
        default=0.5,
        help="Additional loop edges per node budget for SARA augmentation",
    )
    parser.add_argument(
        "--SARA_tau_overlap",
        type=float,
        default=0.10,
        help="Minimum overlap ratio threshold for SARA edges",
    )
    parser.add_argument(
        "--SARA_tau_parallax",
        type=float,
        default=0.05,
        help="Minimum parallax threshold for SARA edges",
    )
    parser.add_argument(
        "--SARA_alpha",
        type=float,
        default=1.0,
        help="Overlap exponent for SARA edge scoring",
    )
    parser.add_argument(
        "--SARA_beta",
        type=float,
        default=1.0,
        help="Parallax exponent for SARA edge scoring",
    )
    parser.add_argument(
        "--SARA_degree_cap",
        type=int,
        default=6,
        help="Optional degree cap enforced during SARA leaf augmentation",
    )
    parser.add_argument(
        "--SARA_disable_intrinsics",
        action="store_true",
        help="Disable intrinsics usage inside SARA parallax proxy",
    )
    parser.add_argument(
        "--SARA_fx",
        type=float,
        default=None,
        help="Fallback fx value if intrinsics are supplied manually to SARA",
    )
    parser.add_argument(
        "--SARA_fy",
        type=float,
        default=None,
        help="Fallback fy value if intrinsics are supplied manually to SARA",
    )
    # SARA Advanced Augmentation
    parser.add_argument(
        "--SARA_enable_multi_scale_loops",
        action="store_true",
        default=True,
        help="Enable multi-scale loop augmentation for SARA",
    )
    parser.add_argument(
        "--SARA_small_loop_ratio",
        type=float,
        default=0.5,
        help="Fraction of loop budget for small loops (path length 2)",
    )
    parser.add_argument(
        "--SARA_medium_loop_ratio",
        type=float,
        default=0.3,
        help="Fraction of loop budget for medium loops (path length 3-4)",
    )
    parser.add_argument(
        "--SARA_large_loop_ratio",
        type=float,
        default=0.2,
        help="Fraction of loop budget for large loops (path length 5+)",
    )
    parser.add_argument(
        "--SARA_enable_long_baseline_anchors",
        action="store_true",
        default=True,
        help="Enable long-baseline anchor edges for scale stability",
    )
    parser.add_argument(
        "--SARA_anchor_count",
        type=int,
        default=10,
        help="Number of long-baseline anchor edges to add",
    )
    parser.add_argument(
        "--SARA_anchor_percentile",
        type=float,
        default=0.95,
        help="Percentile threshold for baseline length (top 5%)",
    )
    parser.add_argument(
        "--SARA_enable_weak_view_reinforcement",
        action="store_true",
        default=True,
        help="Enable weak-view reinforcement for robust initialization",
    )
    parser.add_argument(
        "--SARA_weak_view_percentile",
        type=float,
        default=0.20,
        help="Percentile threshold for weak views (bottom 20%)",
    )
    parser.add_argument(
        "--SARA_weak_view_extra_edges",
        type=int,
        default=2,
        help="Number of extra edges to add per weak view",
    )
    parser.add_argument(
        "--use_SARA_matches",
        action="store_true",
        default=False,
        help="Use matches computed by SARA (mutual NN + RANSAC) instead of LightGlue matching",
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


def _export_features_for_SARA(features: Dict[str, Any], SARA_root: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Write per-image ALIKED npz files expected by the SARA pipeline."""
    try:
        from SARA.io_utils import ensure_dir, normalise_feature_dict
    except ImportError as exc:
        raise ImportError("SARA modules are required but not installed.") from exc

    features_dir = ensure_dir(SARA_root / "features")
    converted: Dict[str, Dict[str, Any]] = {}
    for key, payload in features.items():
        converted_payload: Dict[str, Any] = {}
        for field in ("keypoints", "descriptors", "scores"):
            value = payload[field]
            if torch.is_tensor(value):
                converted_payload[field] = value.detach().cpu().numpy()
            else:
                converted_payload[field] = np.asarray(value)
        converted_payload["image_shape"] = payload.get("image_shape", payload.get("shape"))
        converted[key] = converted_payload

    normalised = normalise_feature_dict(converted)

    for stem, payload in normalised.items():
        shape = np.asarray(payload.get("shape", (0, 0)), dtype=np.int32)
        np.savez(
            features_dir / f"{stem}.npz",
            keypoints=payload["kpt"].astype(np.float32),
            descriptors=payload["desc"].astype(np.float32),
            scores=payload["score"].astype(np.float32),
            image_shape=shape,
        )
    return normalised


def _select_pairs_with_SARA(
    input_dir: str,
    output_path: Path,
    image_paths: List[str],
    features: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> List[tuple[str, str]]:
    """Run the SARA pair selection pipeline and return image path pairs."""
    try:
        from SARA.config import SARAConfig
        from SARA.cli import run_SARA
    except ImportError as exc:
        raise ImportError("SARA modules are required but not installed.") from exc

    SARA_root = output_path / "SARA"
    SARA_root.mkdir(parents=True, exist_ok=True)
    _export_features_for_SARA(features, SARA_root)

    cfg = SARAConfig(
        img_dir=input_dir,
        out_dir=str(SARA_root),
        knn_k=kwargs.get("SARA_knn_k", 30),
        top_t_mutual=kwargs.get("SARA_top_t_mutual", 128),
        min_nn_for_ransac=kwargs.get("SARA_min_nn_for_ransac", 32),
        ransac_iters=kwargs.get("SARA_ransac_iters", 15),
        ransac_conf=kwargs.get("SARA_ransac_conf", 0.999),
        tau_overlap=kwargs.get("SARA_tau_overlap", 0.10),
        tau_parallax=kwargs.get("SARA_tau_parallax", 0.05),
        alpha=kwargs.get("SARA_alpha", 1.0),
        beta=kwargs.get("SARA_beta", 1.0),
        loop_budget_per_node=kwargs.get("SARA_loop_budget_per_node", 0.5),
        deg_cap=kwargs.get("SARA_degree_cap", 6),
        # Advanced augmentation strategies
        enable_multi_scale_loops=kwargs.get("SARA_enable_multi_scale_loops", True),
        small_loop_ratio=kwargs.get("SARA_small_loop_ratio", 0.5),
        medium_loop_ratio=kwargs.get("SARA_medium_loop_ratio", 0.3),
        large_loop_ratio=kwargs.get("SARA_large_loop_ratio", 0.2),
        enable_long_baseline_anchors=kwargs.get("SARA_enable_long_baseline_anchors", True),
        anchor_count=kwargs.get("SARA_anchor_count", 10),
        anchor_percentile=kwargs.get("SARA_anchor_percentile", 0.95),
        enable_weak_view_reinforcement=kwargs.get("SARA_enable_weak_view_reinforcement", True),
        weak_view_percentile=kwargs.get("SARA_weak_view_percentile", 0.20),
        weak_view_extra_edges=kwargs.get("SARA_weak_view_extra_edges", 2),
        # Standard parameters
        use_intrinsics=not kwargs.get("SARA_disable_intrinsics", False),
        fx=kwargs.get("SARA_fx"),
        fy=kwargs.get("SARA_fy"),
        num_workers=kwargs.get("SARA_num_workers", 8),
        cache_dir=kwargs.get("SARA_cache_dir"),
        device=kwargs.get("SARA_device", "cuda"),
    )

    result = run_SARA(cfg)
    stem_to_path = {Path(p).stem: p for p in image_paths}

    pairs = []
    seen = set()
    for stem_i, stem_j, *_ in result["pairs"]:
        if stem_i not in stem_to_path or stem_j not in stem_to_path:
            logger.warning(
                f"SARA pair ({stem_i}, {stem_j}) missing from image set, skipping."
            )
            continue
        key = tuple(sorted((stem_i, stem_j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((stem_to_path[stem_i], stem_to_path[stem_j]))

    if not pairs:
        raise RuntimeError("SARA did not produce any valid image pairs.")
    return pairs


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
            "max_total_pairs": args.max_total_pairs,
            "copy_to_3dgs_dir": args.copy_to_3dgs_dir,
            "high_quality": args.high_quality,
            "device": args.device,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "SARA_device": args.SARA_device,
            "SARA_knn_k": args.SARA_knn_k,
            "SARA_top_t_mutual": args.SARA_top_t_mutual,
            "SARA_min_nn_for_ransac": args.SARA_min_nn_for_ransac,
            "SARA_loop_budget_per_node": args.SARA_loop_budget_per_node,
            "SARA_tau_overlap": args.SARA_tau_overlap,
            "SARA_tau_parallax": args.SARA_tau_parallax,
            "SARA_alpha": args.SARA_alpha,
            "SARA_beta": args.SARA_beta,
            "SARA_degree_cap": args.SARA_degree_cap,
            "SARA_disable_intrinsics": args.SARA_disable_intrinsics,
            "SARA_fx": args.SARA_fx,
            "SARA_fy": args.SARA_fy,
            # Advanced SARA augmentation
            "SARA_enable_multi_scale_loops": args.SARA_enable_multi_scale_loops,
            "SARA_small_loop_ratio": args.SARA_small_loop_ratio,
            "SARA_medium_loop_ratio": args.SARA_medium_loop_ratio,
            "SARA_large_loop_ratio": args.SARA_large_loop_ratio,
            "SARA_enable_long_baseline_anchors": args.SARA_enable_long_baseline_anchors,
            "SARA_anchor_count": args.SARA_anchor_count,
            "SARA_anchor_percentile": args.SARA_anchor_percentile,
            "SARA_enable_weak_view_reinforcement": args.SARA_enable_weak_view_reinforcement,
            "SARA_weak_view_percentile": args.SARA_weak_view_percentile,
            "SARA_weak_view_extra_edges": args.SARA_weak_view_extra_edges,
            "use_SARA_matches": args.use_SARA_matches,
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
    logger.info(f"Use SARA matches: {kwargs.get('use_SARA_matches', False)}")
    logger.info(f"GPU brute force matching: {kwargs.get('use_brute_force', True)}")
    logger.info(f"High quality mode: {kwargs.get('high_quality', False)}")

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

    # Stage 3: Pair selection (SARA only)
    logger.info("Stage 3: Selecting pairs with SARA...")
    stage_start = time.time()

    image_pairs = _select_pairs_with_SARA(
        input_dir,
        output_path,
        image_paths,
        features,
        kwargs,
    )

    stage_times["pair_selection"] = time.time() - stage_start
    logger.info(
        f"SARA produced {len(image_pairs)} candidate pairs in {stage_times['pair_selection']:.2f}s"
    )

    cleanup_gpu_memory(device, "pair selection")

    # Stage 4: Feature matching
    logger.info("Stage 4: Feature matching...")
    stage_start = time.time()

    # Check if we should use SARA matches
    use_SARA_matches = kwargs.get("use_SARA_matches", False)
    SARA_matches_file = output_path / "SARA" / "SARA_matches.h5"

    # Check if matches already exist
    matches_file = output_path / "matches.h5"
    matches_tensor_file = output_path / "matches_tensors.pt"

    # Calculate expected number of matches for validation
    expected_pairs = len(image_pairs)

    # Try to load SARA matches if enabled
    if use_SARA_matches and SARA_matches_file.exists():
        try:
            from sfm.utils.io_utils import load_matches

            logger.info(f"Loading SARA matches from {SARA_matches_file}...")
            SARA_matches_raw = load_matches(SARA_matches_file)

            # Convert stem-based keys to full path-based keys
            stem_to_path = {Path(p).stem: p for p in image_paths}
            matches = {}

            for (stem_i, stem_j), match_data in SARA_matches_raw.items():
                if stem_i in stem_to_path and stem_j in stem_to_path:
                    path_i = stem_to_path[stem_i]
                    path_j = stem_to_path[stem_j]
                    matches[(path_i, path_j)] = match_data
                else:
                    logger.warning(f"SARA match pair ({stem_i}, {stem_j}) not found in image paths")

            if len(matches) >= expected_pairs * 0.1:
                logger.info(
                    f"Loaded {len(matches)} matches from SARA (expected ~{expected_pairs})"
                )
                stage_times["feature_matching"] = 0.0

                # Save in standard format for consistency
                save_matches(matches, matches_file)
                logger.info(f"Saved SARA matches to standard format: {matches_file}")
            else:
                logger.warning(
                    f"SARA match count too low: {len(matches)} vs expected ~{expected_pairs}, falling back to LightGlue"
                )
                matches = None
        except Exception as e:
            logger.warning(f"Could not load SARA matches ({e}), falling back to LightGlue")
            matches = None
    elif use_SARA_matches and not SARA_matches_file.exists():
        logger.warning(
            f"SARA matches requested but file not found: {SARA_matches_file}, falling back to LightGlue"
        )
        matches = None
    elif matches_file.exists() and matches_tensor_file.exists():
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
            "use_vocabulary_tree": False,
            "max_total_pairs": kwargs.get("max_total_pairs", None),
            "output_path": str(output_path),
        }

        matcher_config["predefined_pairs"] = image_pairs

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

    # Stage 9: Copy reconstruction files for 3DGS coSARAtibility
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

    # Save in COLMAP format for 3DGS coSARAtibility
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
