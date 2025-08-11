#!/usr/bin/env python3
"""
Semantic-aware robust point selection for 3D Gaussian Splatting
Replaces dense pointcloud approach with semantic segmentation filtering
"""

import numpy as np
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import struct
import torch

from .semantic_segmentation import SemanticSegmenter

logger = logging.getLogger(__name__)

# ADE20K class priorities for 3DGS (SegFormer model classes)
SEMANTIC_PRIORITIES = {
    # High priority - Structural elements (good for 3DGS)
    'building': {'id': 1, 'weight': 1.0, 'max_points_per_image': 500},
    'wall': {'id': 12, 'weight': 0.9, 'max_points_per_image': 400},
    'fence': {'id': 13, 'weight': 0.8, 'max_points_per_image': 300},
    'floor': {'id': 3, 'weight': 0.8, 'max_points_per_image': 400},
    'road': {'id': 6, 'weight': 0.8, 'max_points_per_image': 400},
    'sidewalk': {'id': 11, 'weight': 0.7, 'max_points_per_image': 300},
    
    # Medium priority - Textured objects
    'tree': {'id': 4, 'weight': 0.6, 'max_points_per_image': 200},
    'vegetation': {'id': 9, 'weight': 0.6, 'max_points_per_image': 200},
    'car': {'id': 20, 'weight': 0.7, 'max_points_per_image': 150},
    'furniture': {'id': 31, 'weight': 0.6, 'max_points_per_image': 100},
    
    # Low priority - Less important for structure
    'window': {'id': 8, 'weight': 0.4, 'max_points_per_image': 100},
    'door': {'id': 14, 'weight': 0.5, 'max_points_per_image': 100},
    'person': {'id': 19, 'weight': 0.3, 'max_points_per_image': 50},
    
    # Filter out - Not useful for 3DGS
    'sky': {'id': 2, 'weight': 0.0, 'max_points_per_image': 0},
    'ceiling': {'id': 5, 'weight': 0.1, 'max_points_per_image': 50},
    'water': {'id': 21, 'weight': 0.2, 'max_points_per_image': 50},
}

# Fallback for unknown classes
DEFAULT_CLASS_CONFIG = {'weight': 0.5, 'max_points_per_image': 100}


class SemanticRobustPoints:
    """Generate robust points3D.bin using semantic segmentation filtering"""
    
    def __init__(self, device: str = "cuda", segmentation_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
        """
        Initialize semantic-aware point filtering
        
        Args:
            device: Device to run segmentation on
            segmentation_model: HuggingFace model name
        """
        self.device = device
        
        # Initialize semantic segmentation
        logger.info("Initializing semantic segmentation model...")
        try:
            self.segmenter = SemanticSegmenter(model_name=segmentation_model, device=device)
            self.label_map = self.segmenter.get_label_info()
            logger.info(f"✅ Loaded segmentation model with {len(self.label_map)} classes")
        except Exception as e:
            logger.warning(f"❌ Semantic segmentation unavailable: {e}")
            logger.warning("Falling back to quality-only filtering (no semantic analysis)")
            self.segmenter = None
            self.label_map = {}
    
    def _get_class_config(self, class_id: int) -> Dict:
        """Get configuration for semantic class"""
        if class_id in self.label_map:
            class_name = self.label_map[class_id].lower()
            
            # Find best match in priorities
            for priority_name, config in SEMANTIC_PRIORITIES.items():
                if priority_name in class_name or class_name in priority_name:
                    return config
        
        return DEFAULT_CLASS_CONFIG
    
    def filter_points_by_semantics(
        self,
        points3d: Dict[int, Dict],
        images: Dict[str, Dict],
        image_dir: Path,
        target_points: int = 20000,
        quality_threshold: float = 2.0
    ) -> Dict[int, Dict]:
        """
        Filter COLMAP points3D using semantic segmentation
        
        Args:
            points3d: Original COLMAP 3D points
            images: COLMAP image data with camera poses
            image_dir: Directory containing source images
            target_points: Target number of points to keep
            quality_threshold: Maximum reprojection error to keep
            
        Returns:
            Filtered points3D dictionary
        """
        if not self.segmenter:
            logger.warning("No semantic segmentation available, using quality filtering only")
            return self._filter_by_quality_only(points3d, target_points, quality_threshold)
        
        logger.info(f"Semantic filtering: {len(points3d)} → target {target_points} points")
        
        # Step 1: Generate semantic masks for all images
        image_paths = []
        valid_images = {}
        
        for img_path, img_data in images.items():
            full_path = image_dir / img_data.get('name', Path(img_path).name)
            if full_path.exists():
                image_paths.append(str(full_path))
                valid_images[str(full_path)] = img_data
        
        logger.info(f"Generating semantic masks for {len(image_paths)} images...")
        semantic_masks = self.segmenter.segment_images_batch(image_paths, batch_size=4)
        
        # Step 2: Analyze each 3D point's semantic context
        point_scores = {}
        class_point_counts = {}
        
        logger.info("Analyzing semantic context of 3D points...")
        for point_id, point_data in tqdm(points3d.items(), desc="Semantic analysis"):
            try:
                # Get point's 3D position and track
                xyz = np.array(point_data['xyz'])
                error = point_data.get('error', 0.0)
                track = point_data.get('track', [])
                
                if error > quality_threshold or len(track) < 2:
                    continue
                
                # Analyze semantic context from observations
                semantic_scores = []
                observation_count = 0
                
                for img_id, point2d_idx in track[:10]:  # Limit to 10 observations for efficiency
                    # Find corresponding image
                    matching_img_path = None
                    for img_path, img_data in valid_images.items():
                        if img_data.get('name', '').split('.')[0] in img_path or str(img_id) in img_path:
                            matching_img_path = img_path
                            break
                    
                    if not matching_img_path or matching_img_path not in semantic_masks:
                        continue
                    
                    semantic_mask = semantic_masks[matching_img_path]
                    if semantic_mask is None:
                        continue
                    
                    # Get 2D projection coordinates from COLMAP data
                    img_data = valid_images[matching_img_path]
                    xys = img_data.get('xys', np.array([]))
                    
                    if len(xys) > point2d_idx:
                        x, y = xys[point2d_idx].astype(int)
                        h, w = semantic_mask.shape
                        
                        if 0 <= x < w and 0 <= y < h:
                            class_id = semantic_mask[y, x]
                            class_config = self._get_class_config(class_id)
                            semantic_scores.append(class_config['weight'])
                            observation_count += 1
                
                if observation_count == 0:
                    continue
                
                # Calculate final score
                avg_semantic_score = np.mean(semantic_scores)
                quality_score = 1.0 / (1.0 + error)  # Higher score for lower error
                final_score = avg_semantic_score * quality_score
                
                point_scores[point_id] = {
                    'score': final_score,
                    'error': error,
                    'semantic_weight': avg_semantic_score,
                    'observations': observation_count
                }
                
                # Track class distribution
                primary_class = int(np.argmax([semantic_scores.count(s) for s in set(semantic_scores)]) if semantic_scores else 0)
                class_point_counts[primary_class] = class_point_counts.get(primary_class, 0) + 1
                
            except Exception as e:
                logger.debug(f"Error analyzing point {point_id}: {e}")
                continue
        
        logger.info(f"Analyzed {len(point_scores)} points with semantic context")
        logger.info(f"Class distribution: {dict(sorted(class_point_counts.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        # Step 3: Select best points with class balancing
        selected_points = {}
        
        # Sort points by score
        sorted_points = sorted(point_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Select top points with class balancing
        class_quotas = {}
        points_per_class = target_points // len(SEMANTIC_PRIORITIES)
        
        for point_id, score_data in sorted_points:
            if len(selected_points) >= target_points:
                break
            
            # Simple selection strategy - take top scoring points
            selected_points[point_id] = points3d[point_id]
        
        logger.info(f"Selected {len(selected_points)} robust points for 3DGS")
        return selected_points
    
    def _filter_by_quality_only(self, points3d: Dict[int, Dict], target_points: int, quality_threshold: float) -> Dict[int, Dict]:
        """Fallback filtering by reprojection error only"""
        logger.info("Filtering by quality only (no semantic segmentation)")
        
        # Filter by quality and sort by error
        quality_points = {}
        for point_id, point_data in points3d.items():
            error = point_data.get('error', 0.0)
            track = point_data.get('track', [])
            
            if error <= quality_threshold and len(track) >= 2:
                quality_points[point_id] = {'point_data': point_data, 'error': error, 'track_length': len(track)}
        
        # Sort by quality (low error + high track length)
        sorted_points = sorted(quality_points.items(), key=lambda x: x[1]['error'] - 0.1 * x[1]['track_length'])
        
        # Select top points
        selected_points = {}
        for point_id, data in sorted_points[:target_points]:
            selected_points[point_id] = data['point_data']
        
        logger.info(f"Quality filtering: {len(points3d)} → {len(selected_points)} points")
        return selected_points
    
    def create_robust_points3d_bin(
        self,
        colmap_points3d: Dict[int, Dict],
        colmap_images: Dict[str, Dict],
        image_dir: Path,
        output_path: Path,
        target_points: int = 20000,
        quality_threshold: float = 2.0
    ) -> int:
        """
        Create robust points3D.bin file optimized for 3DGS
        
        Args:
            colmap_points3d: Original COLMAP 3D points
            colmap_images: COLMAP image data
            image_dir: Source image directory
            output_path: Output points3D.bin path
            target_points: Target number of points
            quality_threshold: Maximum reprojection error
            
        Returns:
            Number of points written
        """
        logger.info("Creating semantic-robust points3D.bin for 3DGS...")
        
        # Filter points using semantic analysis
        robust_points = self.filter_points_by_semantics(
            colmap_points3d, colmap_images, image_dir, target_points, quality_threshold
        )
        
        if not robust_points:
            logger.error("No points passed semantic filtering")
            return 0
        
        # Write COLMAP binary format
        logger.info(f"Writing {len(robust_points)} points to {output_path}")
        
        try:
            with open(output_path, 'wb') as f:
                # Write number of points
                f.write(struct.pack('<Q', len(robust_points)))
                
                for point_id, point_data in robust_points.items():
                    xyz = point_data['xyz']
                    rgb = point_data['rgb']
                    error = point_data.get('error', 0.0)
                    track = point_data.get('track', [])
                    
                    # Write point data (COLMAP format)
                    f.write(struct.pack('<Q', point_id))  # point3D_id
                    f.write(struct.pack('<ddd', float(xyz[0]), float(xyz[1]), float(xyz[2])))  # XYZ
                    f.write(struct.pack('<BBB', int(rgb[0]), int(rgb[1]), int(rgb[2])))  # RGB
                    f.write(struct.pack('<d', float(error)))  # error
                    
                    # Write track
                    f.write(struct.pack('<Q', len(track)))  # track length
                    for img_id, point2d_idx in track:
                        f.write(struct.pack('<ii', int(img_id), int(point2d_idx)))
            
            logger.info(f"✅ Created robust points3D.bin with {len(robust_points)} points optimized for 3DGS")
            return len(robust_points)
            
        except Exception as e:
            logger.error(f"Failed to write points3D.bin: {e}")
            return 0


def create_semantic_robust_points3d(
    sparse_dir: Path,
    image_dir: Path,
    output_dir: Path,
    target_points: int = 20000,
    quality_threshold: float = 2.0,
    device: str = "cuda"
) -> bool:
    """
    Main function to create semantic-robust points3D.bin
    
    Args:
        sparse_dir: COLMAP sparse reconstruction directory
        image_dir: Source images directory  
        output_dir: Output directory for robust points3D.bin
        target_points: Target number of points to keep
        quality_threshold: Maximum reprojection error threshold
        device: Device for semantic segmentation
        
    Returns:
        Success status
    """
    try:
        # Read COLMAP data
        from .colmap_binary import read_colmap_binary_results
        
        points3d, cameras, images = read_colmap_binary_results(sparse_dir)
        
        if not points3d:
            logger.error("No COLMAP 3D points found")
            return False
        
        # Create semantic robust points
        semantic_filter = SemanticRobustPoints(device=device)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_points = semantic_filter.create_robust_points3d_bin(
            points3d, images, image_dir, output_dir / "points3D.bin",
            target_points, quality_threshold
        )
        
        if num_points > 0:
            logger.info(f"✅ Success! Created semantic-robust reconstruction with {num_points} points")
            return True
        else:
            logger.error("Failed to create robust points3D.bin")
            return False
            
    except Exception as e:
        logger.error(f"Semantic robust points creation failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    sparse_dir = Path("output/enhanced_sfm_test/sparse/0")
    image_dir = Path("ImageInputs/images")
    output_dir = Path("output/enhanced_sfm_test/robust_sparse")
    
    success = create_semantic_robust_points3d(
        sparse_dir=sparse_dir,
        image_dir=image_dir, 
        output_dir=output_dir,
        target_points=25000,
        quality_threshold=1.5,
        device="cuda"
    )
    
    if success:
        print("🎯 Semantic-robust points3D.bin created successfully!")
        print("📁 Ready for 3D Gaussian Splatting training")
    else:
        print("❌ Failed to create semantic-robust points3D.bin")