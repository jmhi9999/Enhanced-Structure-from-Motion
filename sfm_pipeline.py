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

    # DINOv3 Feature extraction
    parser.add_argument(
        "--dino_model_family",
        type=str,
        default="dinov3",
        choices=["dinov3", "dinov2"],
        help="DINO model family (dinov3 recommended)",
    )
    parser.add_argument(
        "--dino_model_name",
        type=str,
        default="dinov3_vitl14",
        help="DINO model variant (e.g., dinov3_vitl14, dinov3_vitb14)",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=1600,
        help="Maximum image size for processing",
    )

    # DINO CLS Retrieval
    parser.add_argument(
        "--dino_topk_primary",
        type=int,
        default=60,
        help="Initial CLS retrieval candidates",
    )
    parser.add_argument(
        "--dino_topk_final",
        type=int,
        default=20,
        help="Final pairs after re-ranking",
    )

    # LoFTR Matching
    parser.add_argument(
        "--loftr_pretrained",
        type=str,
        default="outdoor",
        choices=["outdoor", "indoor"],
        help="LoFTR pretrained weights",
    )
    parser.add_argument(
        "--loftr_max_long_edge",
        type=int,
        default=832,
        help="LoFTR maximum long edge for matching",
    )
    parser.add_argument(
        "--dino_use_attention_guidance",
        action="store_true",
        default=True,
        help="Enable attention-guided LoFTR",
    )

    # Geometric Verification
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=15,
        help="Minimum inliers for valid pair",
    )
    parser.add_argument(
        "--magsac_threshold",
        type=float,
        default=2.0,
        help="MAGSAC++ threshold",
    )

    # Device and performance
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for feature extraction"
    )

    # Context-Aware Bundle Adjustment (always enabled)
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
            "dino_model_family": args.dino_model_family,
            "dino_model_name": args.dino_model_name,
            "max_image_size": args.max_image_size,
            "dino_topk_primary": args.dino_topk_primary,
            "dino_topk_final": args.dino_topk_final,
            "loftr_pretrained": args.loftr_pretrained,
            "loftr_max_long_edge": args.loftr_max_long_edge,
            "dino_use_attention_guidance": args.dino_use_attention_guidance,
            "min_inliers": args.min_inliers,
            "magsac_threshold": args.magsac_threshold,
            "device": args.device,
            "batch_size": args.batch_size,
            "confidence_mode": args.confidence_mode,
            "context_ba_checkpoint": args.context_ba_checkpoint,
        }
    else:
        # Direct function call mode
        device = setup_device(kwargs.get("device", "auto"))

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_path))

    logger.info("=" * 60)
    logger.info("DINOv3-SfM Pipeline (Global Reconstruction)")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DINO model: {kwargs.get('dino_model_family', 'dinov3')}/{kwargs.get('dino_model_name', 'dinov3_vitl14')}")
    logger.info(f"CLS retrieval: Top-{kwargs.get('dino_topk_primary', 60)} → {kwargs.get('dino_topk_final', 20)}")
    logger.info(f"Matcher: LoFTR ({kwargs.get('loftr_pretrained', 'outdoor')})")

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
        # DINOv3 feature extraction
        feature_extractor = FeatureExtractorFactory.create(
            kwargs.get("dino_model_family", "dinov3"),
            device=device,
            config={
                "model_name": kwargs.get("dino_model_name", "dinov3_vitl14"),
                "patch_topk": kwargs.get("dino_patch_topk", 800),
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

    # Stage 3: DINO CLS Retrieval + LoFTR Matching
    logger.info("Stage 3: DINO retrieval + LoFTR matching...")
    matches_file = output_path / "matches.h5"
    matches_tensor_file = output_path / "matches_tensors.pt"

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

    # Stage 4: Global SfM Reconstruction (GLOMAP)
    logger.info("Stage 4: GLOMAP global reconstruction...")
    stage_start = time.time()

    # Extract image directory from first image path
    first_image_path = Path(next(iter(features.keys())))
    image_dir = first_image_path.parent

    # Use GLOMAP for global reconstruction
    # GLOMAP performs:
    # 1. View graph construction from matches
    # 2. Rotation averaging (can use DINO edge weights: N_inlier × cos(CLS_i, CLS_j))
    # 3. Translation averaging + triangulation
    # See DINOv2_SfM_Proposal.md Section 2d for details

    from sfm.core.colmap_binary import glomap_reconstruction

    sparse_points, cameras, images = glomap_reconstruction(
        features=features, matches=matches, output_path=output_path, image_dir=image_dir
    )

    stage_times["sfm_reconstruction"] = time.time() - stage_start
    logger.info(
        f"SfM reconstruction completed in {stage_times['sfm_reconstruction']:.2f}s"
    )

    cleanup_gpu_memory(device, "SfM reconstruction")

    # Stage 5: Context-Aware Bundle Adjustment
    logger.info("Stage 5: Context-Aware Bundle Adjustment...")
    stage_start = time.time()

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
        database_path=output_path,
    )

    stage_times["context_ba"] = time.time() - stage_start
    logger.info(f"Context-Aware BA completed in {stage_times['context_ba']:.2f}s")

    cleanup_gpu_memory(device, "Context-Aware BA")

    # Stage 6: Save results in COLMAP format
    logger.info("Stage 6: Saving results...")
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
    logger.info("DINOv3-SfM PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Number of images: {len(image_paths)}")
    logger.info(f"Number of verified pairs: {len(matches)}")
    logger.info(f"Number of 3D points: {len(sparse_points)}")
    logger.info(f"Number of cameras: {len(cameras)}")

    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Pipeline stages:")
    for stage, duration in stage_times.items():
        logger.info(f"  - {stage}: {duration:.2f}s")
    logger.info("\nReady for 3D Gaussian Splatting!")

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
