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

# Scene type detection keywords
INDOOR_KEYWORDS = ['room', 'living', 'kitchen', 'bedroom', 'bathroom', 'office', 'indoor', 'interior', 'house', 'apartment']
OUTDOOR_KEYWORDS = ['street', 'park', 'outdoor', 'exterior', 'landscape', 'building', 'facade', 'city']

# ADE20K class priorities for 3DGS (SegFormer model classes) - OUTDOOR optimized
OUTDOOR_SEMANTIC_PRIORITIES = {
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
    
    # Filter out - Not useful for outdoor 3DGS
    'sky': {'id': 2, 'weight': 0.0, 'max_points_per_image': 0},
    'ceiling': {'id': 5, 'weight': 0.1, 'max_points_per_image': 50},
    'water': {'id': 21, 'weight': 0.2, 'max_points_per_image': 50},
}

# ADE20K class priorities for 3DGS (SegFormer model classes) - INDOOR optimized
INDOOR_SEMANTIC_PRIORITIES = {
    # High priority - Indoor structural elements
    'wall': {'id': 12, 'weight': 1.0, 'max_points_per_image': 600},
    'ceiling': {'id': 5, 'weight': 0.9, 'max_points_per_image': 500},  # Important for indoor!
    'floor': {'id': 3, 'weight': 0.9, 'max_points_per_image': 500},
    'door': {'id': 14, 'weight': 0.8, 'max_points_per_image': 300},
    'window': {'id': 8, 'weight': 0.7, 'max_points_per_image': 300},
    
    # Medium priority - Indoor furniture and objects
    'furniture': {'id': 31, 'weight': 0.8, 'max_points_per_image': 400},
    'cabinet': {'id': 67, 'weight': 0.7, 'max_points_per_image': 300},
    'table': {'id': 15, 'weight': 0.7, 'max_points_per_image': 250},
    'chair': {'id': 18, 'weight': 0.6, 'max_points_per_image': 200},
    'bed': {'id': 7, 'weight': 0.7, 'max_points_per_image': 300},
    'sofa': {'id': 17, 'weight': 0.7, 'max_points_per_image': 300},
    'shelf': {'id': 22, 'weight': 0.6, 'max_points_per_image': 200},
    
    # Lower priority - Less structural but still important
    'television': {'id': 141, 'weight': 0.5, 'max_points_per_image': 150},
    'lamp': {'id': 36, 'weight': 0.4, 'max_points_per_image': 100},
    'book': {'id': 84, 'weight': 0.3, 'max_points_per_image': 100},
    'picture': {'id': 118, 'weight': 0.4, 'max_points_per_image': 150},
    
    # Dynamic elements - lower priority
    'person': {'id': 19, 'weight': 0.2, 'max_points_per_image': 50},
    
    # Rarely relevant indoors
    'building': {'id': 1, 'weight': 0.3, 'max_points_per_image': 100},
    'sky': {'id': 2, 'weight': 0.0, 'max_points_per_image': 0},  # Still filter out
    'tree': {'id': 4, 'weight': 0.2, 'max_points_per_image': 50},
    'car': {'id': 20, 'weight': 0.1, 'max_points_per_image': 25},
}

# Default to outdoor priorities for backward compatibility
SEMANTIC_PRIORITIES = OUTDOOR_SEMANTIC_PRIORITIES

# Fallback for unknown classes
DEFAULT_CLASS_CONFIG = {'weight': 0.5, 'max_points_per_image': 100}

# Quality-based filtering thresholds (no more target_points limit)
QUALITY_THRESHOLDS = {
    'max_reprojection_error': 1.0,      # 1.0 pixel max error
    'min_track_length': 3,              # Minimum 3 observations
    'min_semantic_weight': 0.3,         # Minimum semantic importance
    'max_total_points': 100000          # Safety limit to prevent memory issues
}

# Class-specific thresholds for fine-grained filtering - OUTDOOR optimized
OUTDOOR_CLASS_THRESHOLDS = {
    # High priority structural elements - strict quality requirements
    'building': {'min_weight': 0.8, 'max_error': 1.5, 'min_track': 3},
    'wall': {'min_weight': 0.7, 'max_error': 1.2, 'min_track': 3},
    'fence': {'min_weight': 0.6, 'max_error': 1.0, 'min_track': 3},
    'floor': {'min_weight': 0.6, 'max_error': 1.2, 'min_track': 3},
    'road': {'min_weight': 0.6, 'max_error': 1.0, 'min_track': 3},
    'sidewalk': {'min_weight': 0.5, 'max_error': 1.0, 'min_track': 2},
    
    # Medium priority textured objects
    'tree': {'min_weight': 0.4, 'max_error': 0.8, 'min_track': 2},
    'vegetation': {'min_weight': 0.4, 'max_error': 0.8, 'min_track': 2},
    'car': {'min_weight': 0.5, 'max_error': 0.8, 'min_track': 2},
    'furniture': {'min_weight': 0.4, 'max_error': 0.8, 'min_track': 2},
    
    # Lower priority elements - relaxed requirements
    'window': {'min_weight': 0.3, 'max_error': 0.6, 'min_track': 2},
    'door': {'min_weight': 0.3, 'max_error': 0.8, 'min_track': 2},
    'person': {'min_weight': 0.2, 'max_error': 0.5, 'min_track': 2},
    
    # Filter out completely - set impossible requirements
    'sky': {'min_weight': 1.0, 'max_error': 0.0, 'min_track': 10},        # Impossible to meet
    'ceiling': {'min_weight': 0.8, 'max_error': 0.3, 'min_track': 4},     # Very strict for outdoor
    'water': {'min_weight': 0.6, 'max_error': 0.4, 'min_track': 3}        # Strict
}

# Class-specific thresholds for fine-grained filtering - INDOOR optimized
INDOOR_CLASS_THRESHOLDS = {
    # High priority indoor structural elements - reasonable requirements
    'wall': {'min_weight': 0.7, 'max_error': 1.5, 'min_track': 3},
    'ceiling': {'min_weight': 0.6, 'max_error': 1.2, 'min_track': 3},     # Much more lenient for indoor!
    'floor': {'min_weight': 0.6, 'max_error': 1.5, 'min_track': 3},
    'door': {'min_weight': 0.5, 'max_error': 1.0, 'min_track': 2},
    'window': {'min_weight': 0.4, 'max_error': 1.0, 'min_track': 2},
    
    # Indoor furniture - reasonable quality requirements
    'furniture': {'min_weight': 0.5, 'max_error': 1.0, 'min_track': 2},
    'cabinet': {'min_weight': 0.4, 'max_error': 1.0, 'min_track': 2},
    'table': {'min_weight': 0.4, 'max_error': 1.0, 'min_track': 2},
    'chair': {'min_weight': 0.3, 'max_error': 0.8, 'min_track': 2},
    'bed': {'min_weight': 0.4, 'max_error': 1.0, 'min_track': 2},
    'sofa': {'min_weight': 0.4, 'max_error': 1.0, 'min_track': 2},
    'shelf': {'min_weight': 0.3, 'max_error': 0.8, 'min_track': 2},
    
    # Decorative elements - relaxed requirements
    'television': {'min_weight': 0.3, 'max_error': 0.8, 'min_track': 2},
    'lamp': {'min_weight': 0.2, 'max_error': 0.6, 'min_track': 2},
    'book': {'min_weight': 0.2, 'max_error': 0.5, 'min_track': 2},
    'picture': {'min_weight': 0.2, 'max_error': 0.6, 'min_track': 2},
    
    # Dynamic elements - lower priority
    'person': {'min_weight': 0.1, 'max_error': 0.5, 'min_track': 2},
    
    # Rarely relevant indoors - very relaxed or filtered
    'building': {'min_weight': 0.2, 'max_error': 1.0, 'min_track': 2},
    'tree': {'min_weight': 0.1, 'max_error': 0.8, 'min_track': 2},
    'car': {'min_weight': 0.1, 'max_error': 0.5, 'min_track': 2},
    
    # Still filter out sky completely
    'sky': {'min_weight': 1.0, 'max_error': 0.0, 'min_track': 10},        # Impossible to meet
    'water': {'min_weight': 0.4, 'max_error': 0.6, 'min_track': 2}        # More lenient for indoor water features
}

# Default to outdoor thresholds for backward compatibility
CLASS_THRESHOLDS = OUTDOOR_CLASS_THRESHOLDS


class SemanticRobustPoints:
    """Generate robust points3D.bin using semantic segmentation filtering"""
    
    def __init__(self, device: str = "cuda", segmentation_model: str = "nvidia/segformer-b0-finetuned-ade-512-512", precomputed_masks: Dict = None, indoor_mode: bool = False):
        """
        Initialize semantic-aware point filtering
        
        Args:
            device: Device to run segmentation on
            segmentation_model: HuggingFace model name
            precomputed_masks: Pre-computed semantic masks (to avoid duplicate computation)
            indoor_mode: Whether to optimize for indoor scenes (ceiling preservation, furniture focus)
        """
        self.device = device
        self.precomputed_masks = precomputed_masks
        self.indoor_mode = indoor_mode
        
        # Set appropriate priorities and thresholds based on mode
        if indoor_mode:
            self.semantic_priorities = INDOOR_SEMANTIC_PRIORITIES
            self.class_thresholds = INDOOR_CLASS_THRESHOLDS
            logger.info("🏠 Indoor mode: Optimized for ceiling, walls, and furniture")
        else:
            self.semantic_priorities = OUTDOOR_SEMANTIC_PRIORITIES
            self.class_thresholds = OUTDOOR_CLASS_THRESHOLDS
            logger.info("🌳 Outdoor mode: Optimized for buildings, roads, and vegetation")
        
        # Initialize semantic segmentation only if no precomputed masks
        if precomputed_masks:
            logger.info("♻️ Using precomputed semantic masks (avoiding duplicate segmentation)")
            self.segmenter = None
            self.label_map = {}  # Will use default mapping
        else:
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
            
            # Find best match in priorities (using instance-specific priorities)
            for priority_name, config in self.semantic_priorities.items():
                if priority_name in class_name or class_name in priority_name:
                    return config
        
        return DEFAULT_CLASS_CONFIG
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if class_id in self.label_map:
            class_name = self.label_map[class_id].lower()
            
            # Find best match in priorities (using instance-specific priorities)
            for priority_name in self.semantic_priorities.keys():
                if priority_name in class_name or class_name in priority_name:
                    return priority_name
        
        return 'unknown'
    
    def _should_keep_point(self, error: float, semantic_weight: float, track_length: int, class_name: str) -> bool:
        """
        Determine if a point should be kept based on quality thresholds
        
        Args:
            error: Reprojection error in pixels
            semantic_weight: Average semantic importance weight
            track_length: Number of observations across images
            class_name: Semantic class name
            
        Returns:
            True if point meets quality requirements
        """
        # Basic quality checks
        if error > QUALITY_THRESHOLDS['max_reprojection_error']:
            return False
        
        if track_length < QUALITY_THRESHOLDS['min_track_length']:
            return False
        
        if semantic_weight < QUALITY_THRESHOLDS['min_semantic_weight']:
            return False
        
        # Class-specific checks (using instance-specific thresholds)
        if class_name in self.class_thresholds:
            thresholds = self.class_thresholds[class_name]
            
            # Check class-specific requirements
            if semantic_weight < thresholds['min_weight']:
                return False
            
            if error > thresholds['max_error']:
                return False
                
            if track_length < thresholds['min_track']:
                return False
        
        return True
    
    def filter_points_by_semantics(
        self,
        points3d: Dict[int, Dict],
        images: Dict[str, Dict],
        image_dir: Path,
        quality_threshold: float = None
    ) -> Dict[int, Dict]:
        """
        Filter COLMAP points3D using quality-based semantic filtering
        
        Args:
            points3d: Original COLMAP 3D points
            images: COLMAP image data with camera poses
            image_dir: Directory containing source images
            quality_threshold: Optional override for max reprojection error (uses QUALITY_THRESHOLDS if None)
            
        Returns:
            Filtered points3D dictionary (keeps all points meeting quality requirements)
        """
        if not self.segmenter and not self.precomputed_masks:
            logger.warning("No semantic segmentation available, using quality filtering only")
            return self._filter_by_quality_only(points3d, quality_threshold or QUALITY_THRESHOLDS['max_reprojection_error'])
        
        # Use configured threshold if not provided
        if quality_threshold is None:
            quality_threshold = QUALITY_THRESHOLDS['max_reprojection_error']
        
        logger.info(f"Quality-based semantic filtering: {len(points3d)} points → filtering by quality thresholds")
        
        # Step 1: Generate semantic masks for all images
        image_paths = []
        valid_images = {}
        image_name_to_path = {}  # Map image names to full paths
        
        for img_path, img_data in images.items():
            image_name = img_data.get('name', Path(img_path).name)
            full_path = image_dir / image_name
            if full_path.exists():
                image_paths.append(str(full_path))
                valid_images[str(full_path)] = img_data
                image_name_to_path[image_name] = str(full_path)
        
        # Use precomputed masks if available, otherwise generate new ones
        if self.precomputed_masks:
            logger.info("♻️ Using precomputed semantic masks")
            semantic_masks = self.precomputed_masks
        else:
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
                    # Find corresponding image using image_id mapping
                    matching_img_path = None
                    
                    # Try to find image by ID first
                    for img_path, img_data in valid_images.items():
                        if img_data.get('id') == img_id:
                            matching_img_path = img_path
                            break
                    
                    # Fallback: try to find by name matching
                    if not matching_img_path:
                        for img_path, img_data in valid_images.items():
                            image_name = img_data.get('name', '')
                            if image_name and (str(img_id) in image_name or image_name.split('.')[0] == str(img_id)):
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
                
                # Calculate semantic analysis
                avg_semantic_score = np.mean(semantic_scores)
                
                # Get primary semantic class
                if semantic_scores:
                    # Find the most common semantic score and its corresponding class
                    score_counts = {}
                    class_ids_for_scores = {}
                    for i, score in enumerate(semantic_scores):
                        if score not in score_counts:
                            score_counts[score] = 0
                            class_ids_for_scores[score] = []
                        score_counts[score] += 1
                        # We need to track which class_id corresponds to this score
                    
                    # For simplicity, estimate primary class from semantic weight
                    primary_class_name = 'unknown'
                    if avg_semantic_score >= 0.8:
                        primary_class_name = 'building'
                    elif avg_semantic_score >= 0.6:
                        primary_class_name = 'wall'
                    elif avg_semantic_score >= 0.4:
                        primary_class_name = 'tree'
                    elif avg_semantic_score >= 0.2:
                        primary_class_name = 'person'
                    elif avg_semantic_score <= 0.1:
                        primary_class_name = 'sky'
                else:
                    primary_class_name = 'unknown'
                    avg_semantic_score = 0.5  # Default for unknown
                
                # Quality-based filtering decision
                should_keep = self._should_keep_point(
                    error=error,
                    semantic_weight=avg_semantic_score,
                    track_length=observation_count,
                    class_name=primary_class_name
                )
                
                if should_keep:
                    point_scores[point_id] = {
                        'error': error,
                        'semantic_weight': avg_semantic_score,
                        'observations': observation_count,
                        'class_name': primary_class_name
                    }
                    
                    # Track class distribution
                    class_point_counts[primary_class_name] = class_point_counts.get(primary_class_name, 0) + 1
                
            except Exception as e:
                logger.debug(f"Error analyzing point {point_id}: {e}")
                continue
        
        logger.info(f"Quality filtering completed: {len(point_scores)} points passed quality thresholds")
        logger.info(f"Class distribution: {dict(sorted(class_point_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
        
        # Step 3: Return all points that passed quality filtering
        selected_points = {}
        
        # Safety check - limit total points if too many pass filtering
        if len(point_scores) > QUALITY_THRESHOLDS['max_total_points']:
            logger.warning(f"Too many points ({len(point_scores)}) passed filtering. Applying safety limit of {QUALITY_THRESHOLDS['max_total_points']}")
            
            # Sort by combined quality score for safety limit
            sorted_points = sorted(
                point_scores.items(), 
                key=lambda x: x[1]['semantic_weight'] * (1.0 / (1.0 + x[1]['error'])), 
                reverse=True
            )
            
            for point_id, score_data in sorted_points[:QUALITY_THRESHOLDS['max_total_points']]:
                selected_points[point_id] = points3d[point_id]
        else:
            # Use all points that passed quality filtering
            for point_id in point_scores.keys():
                selected_points[point_id] = points3d[point_id]
        
        logger.info(f"Final selection: {len(selected_points)} robust points for 3DGS")
        return selected_points
    
    def _filter_by_quality_only(self, points3d: Dict[int, Dict], quality_threshold: float) -> Dict[int, Dict]:
        """Fallback filtering by reprojection error only"""
        logger.info("Filtering by quality only (no semantic segmentation)")
        
        # Filter by quality thresholds
        selected_points = {}
        
        for point_id, point_data in points3d.items():
            error = point_data.get('error', 0.0)
            track = point_data.get('track', [])
            track_length = len(track)
            
            # Apply basic quality filtering (same as semantic filtering but without class-specific rules)
            if (error <= quality_threshold and 
                track_length >= QUALITY_THRESHOLDS['min_track_length']):
                
                selected_points[point_id] = point_data
        
        # Apply safety limit if too many points
        if len(selected_points) > QUALITY_THRESHOLDS['max_total_points']:
            logger.warning(f"Too many points ({len(selected_points)}) passed quality-only filtering. Applying safety limit.")
            
            # Sort by quality and take the best ones
            quality_sorted = sorted(
                selected_points.items(),
                key=lambda x: x[1].get('error', 0.0) - 0.1 * len(x[1].get('track', []))
            )
            
            selected_points = dict(quality_sorted[:QUALITY_THRESHOLDS['max_total_points']])
        
        logger.info(f"Quality-only filtering: {len(points3d)} → {len(selected_points)} points")
        return selected_points
    
    def create_robust_points3d_bin(
        self,
        colmap_points3d: Dict[int, Dict],
        colmap_images: Dict[str, Dict],
        image_dir: Path,
        output_path: Path,
        quality_threshold: float = None
    ) -> int:
        """
        Create robust points3D.bin file optimized for 3DGS using quality-based filtering
        
        Args:
            colmap_points3d: Original COLMAP 3D points
            colmap_images: COLMAP image data
            image_dir: Source image directory
            output_path: Output points3D.bin path
            quality_threshold: Optional override for max reprojection error (uses QUALITY_THRESHOLDS if None)
            
        Returns:
            Number of points written
        """
        logger.info("Creating semantic-robust points3D.bin for 3DGS...")
        
        # Filter points using quality-based semantic analysis
        robust_points = self.filter_points_by_semantics(
            colmap_points3d, colmap_images, image_dir, quality_threshold
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


def detect_scene_type(image_dir: Path) -> bool:
    """
    Automatically detect if scene is indoor or outdoor based on image directory name
    
    Args:
        image_dir: Path to image directory
        
    Returns:
        True if indoor scene detected, False if outdoor
    """
    dir_name = str(image_dir).lower()
    path_parts = [part.lower() for part in image_dir.parts]
    
    # Check for indoor keywords in directory path
    for keyword in INDOOR_KEYWORDS:
        if any(keyword in part for part in path_parts):
            logger.info(f"🏠 Indoor scene detected (keyword: '{keyword}')")
            return True
    
    # Check for outdoor keywords in directory path
    for keyword in OUTDOOR_KEYWORDS:
        if any(keyword in part for part in path_parts):
            logger.info(f"🌳 Outdoor scene detected (keyword: '{keyword}')")
            return False
    
    # Default to outdoor if uncertain
    logger.info("🤔 Scene type uncertain, defaulting to outdoor mode")
    return False


def create_semantic_robust_points3d(
    sparse_dir: Path,
    image_dir: Path,
    output_dir: Path,
    quality_threshold: float = None,
    device: str = "cuda",
    indoor_mode: bool = None
) -> bool:
    """
    Main function to create quality-based semantic-robust points3D.bin
    
    Args:
        sparse_dir: COLMAP sparse reconstruction directory
        image_dir: Source images directory  
        output_dir: Output directory for robust points3D.bin
        quality_threshold: Optional override for max reprojection error (uses QUALITY_THRESHOLDS if None)
        device: Device for semantic segmentation
        indoor_mode: Override for indoor/outdoor mode (None for auto-detection)
        
    Returns:
        Success status
    """
    try:
        # Auto-detect scene type if not specified
        if indoor_mode is None:
            indoor_mode = detect_scene_type(image_dir)
        
        # Read COLMAP data
        from .colmap_binary import read_colmap_binary_results
        
        points3d, cameras, images = read_colmap_binary_results(sparse_dir)
        
        if not points3d:
            logger.error("No COLMAP 3D points found")
            return False
        
        # Create semantic robust points with appropriate mode
        semantic_filter = SemanticRobustPoints(device=device, indoor_mode=indoor_mode)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_points = semantic_filter.create_robust_points3d_bin(
            points3d, images, image_dir, output_dir / "points3D.bin",
            quality_threshold
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
        quality_threshold=1.0,  # Use stricter quality threshold
        device="cuda"
    )
    
    if success:
        print("🎯 Semantic-robust points3D.bin created successfully!")
        print("📁 Ready for 3D Gaussian Splatting training")
    else:
        print("❌ Failed to create semantic-robust points3D.bin")