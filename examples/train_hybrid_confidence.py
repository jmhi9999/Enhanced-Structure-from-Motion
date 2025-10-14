#!/usr/bin/env python3
"""
Training script for Hybrid Confidence MLP

This script demonstrates how to train the hybrid confidence MLP using
pseudo-ground-truth labels from COLMAP reconstructions.

Usage:
    python examples/train_hybrid_confidence.py \
        --data_dir /path/to/datasets \
        --output_dir models/ \
        --num_epochs 50
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sfm.core.context_ba import (
    ContextBAConfig,
    SceneGraphBuilder,
    RuleBasedConfidence,
    HybridConfidence,
)
from sfm.core.context_ba.config import HybridMLPConfig

logger = logging.getLogger(__name__)


def load_colmap_reconstruction(sparse_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Load COLMAP reconstruction from binary files

    Args:
        sparse_dir: Path to sparse/0 directory

    Returns:
        Tuple of (cameras, images, points3d)
    """
    # This is a placeholder - implement actual COLMAP binary reading
    # For now, use the colmap_binary module
    from sfm.core.colmap_binary import read_binary_cameras, read_binary_images, read_binary_points3d

    cameras_file = sparse_dir / "cameras.bin"
    images_file = sparse_dir / "images.bin"
    points3d_file = sparse_dir / "points3D.bin"

    if not all([cameras_file.exists(), images_file.exists(), points3d_file.exists()]):
        raise FileNotFoundError(f"COLMAP reconstruction not found in {sparse_dir}")

    cameras = read_binary_cameras(cameras_file)
    images = read_binary_images(images_file)
    points3d = read_binary_points3d(points3d_file)

    return cameras, images, points3d


def compute_pseudo_labels_from_colmap(
    scene_graph: Any,
    colmap_images: Dict[int, Any],
    colmap_points3d: Dict[int, Any],
) -> np.ndarray:
    """
    Compute pseudo-ground-truth confidence labels from COLMAP results

    Cameras with low reprojection error → high confidence (1.0)
    Cameras with high reprojection error → low confidence (0.0)

    Args:
        scene_graph: SceneGraph instance
        colmap_images: COLMAP image data
        colmap_points3d: COLMAP 3D points

    Returns:
        Array of confidence labels, shape (num_cameras,)
    """
    labels = np.zeros(scene_graph.num_cameras())

    # Compute reprojection error for each camera
    for cam_id, camera in scene_graph.cameras.items():
        # Find corresponding COLMAP image
        colmap_img = None
        for img_id, img_data in colmap_images.items():
            if img_data['name'] == Path(camera.image_path).name:
                colmap_img = img_data
                break

        if colmap_img is None:
            labels[cam_id] = 0.5  # Unknown, assign medium confidence
            continue

        # Compute mean reprojection error for this camera
        errors = []
        for pt3d_id in colmap_img.get('point3D_ids', []):
            if pt3d_id == -1:
                continue
            if pt3d_id not in colmap_points3d:
                continue

            point = colmap_points3d[pt3d_id]
            errors.append(point['error'])

        if not errors:
            labels[cam_id] = 0.5
            continue

        mean_error = np.mean(errors)

        # Convert error to confidence label
        # Good cameras (error < 0.5px) → 1.0
        # Medium cameras (0.5 - 1.0px) → 0.5
        # Bad cameras (> 1.0px) → 0.1
        if mean_error < 0.5:
            labels[cam_id] = 1.0
        elif mean_error < 1.0:
            labels[cam_id] = 0.5
        else:
            labels[cam_id] = max(0.1, 1.0 / (1.0 + mean_error))

    return labels


def prepare_training_data(
    dataset_dirs: List[Path],
    config: ContextBAConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from multiple datasets

    Args:
        dataset_dirs: List of dataset directories (each with features/matches/colmap)
        config: ContextBAConfig

    Returns:
        Tuple of (features, labels) arrays
    """
    logger.info(f"Preparing training data from {len(dataset_dirs)} datasets...")

    all_features = []
    all_labels = []

    rule_confidence = RuleBasedConfidence(config)
    graph_builder = SceneGraphBuilder(config.scene_graph)

    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
        try:
            # Load features and matches
            # This is a simplified version - adapt to your data format
            features_file = dataset_dir / "features.h5"
            matches_file = dataset_dir / "matches.h5"

            if not features_file.exists() or not matches_file.exists():
                logger.warning(f"Skipping {dataset_dir}: missing features/matches")
                continue

            from sfm.utils.io_utils import load_features, load_matches
            features = load_features(features_file)
            matches = load_matches(matches_file)

            # Build scene graph
            scene_graph = graph_builder.build(features, matches)

            # Load COLMAP reconstruction
            sparse_dir = dataset_dir / "colmap" / "sparse" / "0"
            if not sparse_dir.exists():
                sparse_dir = dataset_dir / "sparse" / "0"

            cameras, images, points3d = load_colmap_reconstruction(sparse_dir)

            # Extract rule-based features
            for cam_id, camera in scene_graph.cameras.items():
                feature_vec = rule_confidence.extract_feature_vector(camera, scene_graph)
                all_features.append(feature_vec)

            # Compute pseudo labels
            labels = compute_pseudo_labels_from_colmap(scene_graph, images, points3d)
            all_labels.extend(labels)

            logger.info(f"Extracted {scene_graph.num_cameras()} samples from {dataset_dir.name}")

        except Exception as e:
            logger.warning(f"Error processing {dataset_dir}: {e}")
            continue

    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.float32)

    logger.info(f"Total samples: {len(features_array)}")
    logger.info(f"Features shape: {features_array.shape}")
    logger.info(f"Labels shape: {labels_array.shape}")

    return features_array, labels_array


def train_hybrid_confidence(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: ContextBAConfig,
    output_path: Path,
) -> Dict[str, Any]:
    """
    Train hybrid confidence MLP

    Args:
        train_features: Training features (N, 6)
        train_labels: Training labels (N,)
        val_features: Validation features (M, 6)
        val_labels: Validation labels (M,)
        config: ContextBAConfig with hybrid settings
        output_path: Output directory for checkpoints

    Returns:
        Training history
    """
    logger.info("Training Hybrid Confidence MLP...")

    # Initialize hybrid confidence
    hybrid = HybridConfidence(config)

    # Train
    history = hybrid.train(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
    )

    # Save checkpoint
    checkpoint_path = output_path / "confidence_mlp.pth"
    hybrid.save_checkpoint(checkpoint_path)

    logger.info(f"Training completed. Checkpoint saved to {checkpoint_path}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Confidence MLP")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing dataset subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cpu, cuda, or auto)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Find all dataset directories
    data_dir = Path(args.data_dir)
    dataset_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not dataset_dirs:
        logger.error(f"No dataset directories found in {data_dir}")
        return

    logger.info(f"Found {len(dataset_dirs)} datasets")

    # Configure hybrid MLP
    config = ContextBAConfig(
        confidence_mode="hybrid",
        hybrid_mlp=HybridMLPConfig(
            hidden_dim=16,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            device=args.device,
        ),
    )

    # Prepare training data
    features, labels = prepare_training_data(dataset_dirs, config)

    if len(features) == 0:
        logger.error("No training data extracted")
        return

    # Split train/val
    num_samples = len(features)
    num_val = int(num_samples * args.val_split)
    indices = np.random.permutation(num_samples)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]

    logger.info(f"Train samples: {len(train_features)}")
    logger.info(f"Val samples: {len(val_features)}")

    # Train
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history = train_hybrid_confidence(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        config=config,
        output_path=output_path,
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
