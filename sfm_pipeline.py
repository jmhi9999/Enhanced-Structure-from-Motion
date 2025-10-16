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
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from sfm.core.feature_extractor import FeatureExtractorFactory
from sfm.core.feature_matcher import EnhancedLightGlueMatcher
from sfm.core.geometric_verification import GeometricVerification, RANSACMethod
from sfm.core.gpu_vocabulary_tree import GPUVocabularyTree
from sfm.utils.io_utils import (
    save_colmap_format,
    load_images,
    save_features,
    save_matches,
)
from sfm.utils.image_utils import resize_image
from sfm.pipelines import run_dino_pipeline

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
        default="dinov3",
        choices=["superpoint", "aliked", "disk", "dinov3", "dinov2"],
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

    # Context-Aware Bundle Adjustment
    parser.add_argument(
        "--use_context_ba",
        action="store_true",
        help="Use Context-Aware Bundle Adjustment instead of COLMAP BA",
    )
    parser.add_argument(
        "--confidence_mode",
        type=str,
        default="rule_based",
        choices=["rule_based", "hybrid"],
        help="Confidence computation mode (rule_based or hybrid with MLP)",
    )
    parser.add_argument(
        "--context_ba_checkpoint",
        type=str,
        default=None,
        help="Path to hybrid MLP checkpoint (only for --confidence_mode hybrid)",
    )

    # Loop Closure Detection
    parser.add_argument(
        "--detect_loop_closures",
        action="store_true",
        help="Detect and add loop closures to improve reconstruction quality",
    )
    parser.add_argument(
        "--loop_similarity_threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for loop detection (default: 0.70)",
    )
    parser.add_argument(
        "--loop_min_temporal_gap",
        type=int,
        default=30,
        help="Minimum temporal gap for loop closure detection (default: 30)",
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


def _run_legacy_pair_selection_and_matching(
    features: Dict[str, Any],
    device: torch.device,
    output_path: Path,
    image_paths: List[str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[Tuple[str, str], Any], Dict[Tuple[str, str], Dict[str, torch.Tensor]], Dict[str, float]]:
    """Original sequential pair selection and LightGlue matching pipeline."""
    stage_times: Dict[str, float] = {}

    logger.info("Stage 3: Smart pair selection (vocabulary tree + sequential)...")
    stage_start = time.time()

    sequential_overlap = 5
    sequential_pairs: List[Tuple[str, str]] = []
    sorted_image_paths = sorted(image_paths)

    for i, img1 in enumerate(sorted_image_paths):
        for j in range(1, min(sequential_overlap + 1, len(sorted_image_paths) - i)):
            img2 = sorted_image_paths[i + j]
            sequential_pairs.append((img1, img2))

    logger.info(f"Generated {len(sequential_pairs)} sequential pairs")

    if kwargs.get("use_vocab_tree", False):
        vocab_tree = GPUVocabularyTree(
            device=device,
            config={
                "vocab_size": 10000,
                "vocab_depth": 6,
                "vocab_branching_factor": 10,
            },
            output_path=str(output_path),
        )

        vocab_tree.build_vocabulary(features)
        vocab_tree_pairs = vocab_tree.get_image_pairs_for_matching(
            features, max_pairs_per_image=kwargs.get("max_pairs_per_image", 20)
        )

        image_pairs = list(set(sequential_pairs + vocab_tree_pairs))
        logger.info(f"Selected {len(vocab_tree_pairs)} pairs using vocabulary tree")
        logger.info(
            f"Combined total: {len(image_pairs)} unique pairs (sequential + vocabulary tree)"
        )
    else:
        image_pairs = sequential_pairs
        logger.info(f"Using sequential matching: {len(image_pairs)} pairs")

    stage_times["pair_selection"] = time.time() - stage_start
    logger.info(f"Pair selection completed in {stage_times['pair_selection']:.2f}s")

    if "vocab_tree" in locals():
        try:
            if hasattr(vocab_tree, "clear_memory"):
                vocab_tree.clear_memory()
            del vocab_tree
        except Exception as e:
            logger.warning(f"Error cleaning up vocabulary tree: {e}")

    cleanup_gpu_memory(device, "pair selection")

    logger.info("Stage 4: Feature matching...")
    stage_start = time.time()

    matches_file = output_path / "matches.h5"
    matches_tensor_file = output_path / "matches_tensors.pt"
    expected_pairs = len(image_pairs)

    matches = None
    matches_tensors = None

    if matches_file.exists() and matches_tensor_file.exists():
        try:
            from sfm.utils.io_utils import load_matches

            existing_matches = load_matches(matches_file)
            existing_match_tensors = torch.load(
                matches_tensor_file, map_location=device
            )

            if len(existing_matches) >= expected_pairs * 0.1:
                logger.info(
                    f"Found existing matches for {len(existing_matches)} pairs (expected ~{expected_pairs}), skipping matching"
                )
                matches = existing_matches
                matches_tensors = existing_match_tensors
                stage_times["feature_matching"] = 0.0
        except Exception as e:
            logger.info(f"Could not load matches ({e}), recomputing")

    if matches is None:
        matcher_config = {
            "use_brute_force": kwargs.get("use_brute_force", True),
            "use_vocabulary_tree": kwargs.get("use_vocab_tree", False),
            "max_pairs_per_image": kwargs.get("max_pairs_per_image", 20),
            "max_total_pairs": kwargs.get("max_total_pairs", None),
            "parallel_workers": kwargs.get("num_workers", 4),
            "batch_size": kwargs.get("batch_size", 32),
        }

        matcher = EnhancedLightGlueMatcher(
            device=device,
            use_brute_force=kwargs.get("use_brute_force", True),
            use_vocabulary_tree=kwargs.get("use_vocab_tree", False),
            feature_type=kwargs.get("feature_extractor", "superpoint"),
            config=matcher_config,
        )

        if kwargs.get("use_vocab_tree", False):
            matcher.config["predefined_pairs"] = image_pairs

        matches = matcher.match_features(features)

        matches_tensors = {}
        for pair, match in matches.items():
            matches_tensors[pair] = {
                "matches0": torch.from_numpy(match["matches0"]).to(device),
                "matches1": torch.from_numpy(match["matches1"]).to(device),
                "mscores0": torch.from_numpy(match.get("mscores0", np.ones_like(match["matches0"], dtype=np.float32))).to(device),
                "mscores1": torch.from_numpy(match.get("mscores1", np.ones_like(match["matches1"], dtype=np.float32))).to(device),
            }

        save_matches(matches, matches_file)
        torch.save(matches_tensors, matches_tensor_file)

        stage_times["feature_matching"] = time.time() - stage_start
        logger.info(
            f"Feature matching completed in {stage_times['feature_matching']:.2f}s"
        )

        if "matcher" in locals():
            try:
                matcher.clear_memory()
                del matcher
            except Exception as e:
                logger.warning(f"Error cleaning up matcher: {e}")

        cleanup_gpu_memory(device, "feature matching")

    return matches, matches_tensors, stage_times
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
            "use_context_ba": args.use_context_ba,
            "confidence_mode": args.confidence_mode,
            "context_ba_checkpoint": args.context_ba_checkpoint,
            "detect_loop_closures": args.detect_loop_closures,
            "loop_similarity_threshold": args.loop_similarity_threshold,
            "loop_min_temporal_gap": args.loop_min_temporal_gap,
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

    feature_type = kwargs.get("feature_extractor", "superpoint")
    matches_file = output_path / "matches.h5"
    matches_tensor_file = output_path / "matches_tensors.pt"

    if feature_type in {"dinov3", "dinov2", "dino"}:
        logger.info("Stage 3: DINO retrieval + LoFTR matching...")
        features, matches, matches_tensors, dino_times = run_dino_pipeline(
            processed_images=processed_images,
            features=features,
            device=device,
            output_path=output_path,
            kwargs=kwargs,
        )
        stage_times.update(dino_times)

        save_features(features, features_file)

        features_tensors = {}
        for img_path, feat_data in features.items():
            features_tensors[img_path] = {
                "keypoints": torch.from_numpy(feat_data["keypoints"]).to(device),
                "descriptors": torch.from_numpy(feat_data["descriptors"]).to(device),
                "scores": torch.from_numpy(feat_data["scores"]).to(device),
                "image_shape": feat_data["image_shape"],
                "dino_cls": torch.from_numpy(feat_data.get("dino_cls", np.zeros((0,), dtype=np.float32))).to(device),
            }
        torch.save(features_tensors, features_tensor_file)

        save_matches(matches, matches_file)
        torch.save(matches_tensors, matches_tensor_file)

        cleanup_gpu_memory(device, "DINO matching")

    else:
        matches, matches_tensors, legacy_times = _run_legacy_pair_selection_and_matching(
            features=features,
            device=device,
            output_path=output_path,
            image_paths=image_paths,
            kwargs=kwargs,
        )
        stage_times.update(legacy_times)

    # Stage 4.5: Loop Closure Detection (optional)
    if kwargs.get("detect_loop_closures", False):
        logger.info("Stage 4.5: Loop Closure Detection...")
        stage_start = time.time()

        try:
            from sfm.core.loop_closure_detector import detect_and_add_loop_closures

            # Configure loop detection
            loop_config = {
                'similarity_threshold': kwargs.get('loop_similarity_threshold', 0.70),
                'min_temporal_gap': kwargs.get('loop_min_temporal_gap', 30),
                'min_matches_for_verification': 15,
                'max_loops_to_verify': 100
            }

            # Detect and add loop closures
            num_matches_before = len(matches)

            # Need to recreate matcher for loop verification
            feature_type = kwargs.get("feature_extractor", "superpoint")
            matcher_config = {
                "use_brute_force": False,  # Only match specific pairs
                "use_vocabulary_tree": False,
            }

            # Create a lightweight matcher for verification
            loop_matcher = EnhancedLightGlueMatcher(
                device=device, feature_type=feature_type, config=matcher_config
            )

            matches = detect_and_add_loop_closures(
                features=features,
                matches=matches,
                matcher=loop_matcher,
                config=loop_config
            )

            num_loops_added = len(matches) - num_matches_before
            logger.info(f"Added {num_loops_added} loop closure edges")
            logger.info(f"Total matches: {len(matches)} pairs")

            # Save updated matches
            save_matches(matches, matches_file)
            logger.info(f"Updated matches saved to {matches_file}")

            # Clean up loop matcher
            if hasattr(loop_matcher, "clear_memory"):
                loop_matcher.clear_memory()
            del loop_matcher

            stage_times["loop_closure_detection"] = time.time() - stage_start
            logger.info(
                f"Loop closure detection completed in {stage_times['loop_closure_detection']:.2f}s"
            )

        except Exception as e:
            logger.warning(f"Loop closure detection failed: {e}")
            logger.warning("Continuing with existing matches")
            stage_times["loop_closure_detection"] = 0.0

        cleanup_gpu_memory(device, "loop closure detection")
    else:
        logger.info("Loop closure detection disabled (use --detect_loop_closures to enable)")

    # Stage 5: SfM reconstruction (COLMAP or Context-Aware BA)
    logger.info("Stage 5: SfM reconstruction...")
    stage_start = time.time()

    # Extract image directory from first image path
    first_image_path = Path(next(iter(features.keys())))
    image_dir = first_image_path.parent

    # Choose reconstruction method
    if kwargs.get("use_context_ba", False):
        logger.info("Using Context-Aware Bundle Adjustment...")

        from sfm.core.context_ba import ContextAwareBundleAdjustment, ContextBAConfig
        from sfm.core.context_ba.config import HybridMLPConfig

        # Configure context BA
        ba_config = ContextBAConfig(
            confidence_mode=kwargs.get("confidence_mode", "rule_based"),
            log_level="INFO",
        )

        # Add hybrid MLP checkpoint if provided
        if kwargs.get("context_ba_checkpoint"):
            ba_config.hybrid_mlp = HybridMLPConfig(
                checkpoint_path=Path(kwargs["context_ba_checkpoint"])
            )

        # Initialize and run context-aware BA
        context_ba = ContextAwareBundleAdjustment(ba_config)

        cameras, images, sparse_points = context_ba.optimize(
            features=features,
            matches=matches,
            image_dir=image_dir,
            database_path=output_path,  # Pass output_path, not database file
        )

        logger.info("Context-Aware BA completed")
    else:
        logger.info("Using COLMAP binary reconstruction...")

        from sfm.core.colmap_binary import colmap_binary_reconstruction

        sparse_points, cameras, images = colmap_binary_reconstruction(
            features=features, matches=matches, output_path=output_path, image_dir=image_dir
        )

    stage_times["sfm_reconstruction"] = time.time() - stage_start
    logger.info(
        f"SfM reconstruction completed in {stage_times['sfm_reconstruction']:.2f}s"
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
                for filename in [
                    "cameras.bin",
                    "images.bin",
                    "points3D.bin",
                    "cameras.txt",
                    "images.txt",
                    "points3D.txt",
                    "project.ini",
                ]:
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
        source_sparse_dir=output_path / "sparse" / "0",
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
