"""
Semantic-aware filtering for Enhanced SfM Pipeline
Provides semantic consistency filtering as regularization for feature matching
Reduces false positive matches between semantically incompatible regions
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)


class SemanticFilter:
    """
    Advanced semantic filtering for feature matching regularization
    Uses semantic segmentation to filter out semantically inconsistent matches
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Filtering parameters - MORE PERMISSIVE for better point cloud density
        self.light_filtering = self.config.get('light_filtering', False)  # Ultra-light filtering mode
        
        if self.light_filtering:
            # Ultra-light filtering - minimal regularization, preserves most matches
            self.consistency_threshold = self.config.get('consistency_threshold', 0.2)  # Only 20% need to be consistent
            self.min_consistent_matches = self.config.get('min_consistent_matches', 5)  # Very low minimum
            self.strict_mode = False
        else:
            # Standard permissive filtering
            self.consistency_threshold = self.config.get('consistency_threshold', 0.4)  # 40% matches must be consistent (was 70%)
            self.min_consistent_matches = self.config.get('min_consistent_matches', 8)  # Minimum consistent matches to keep pair (was 15)
            self.strict_mode = self.config.get('strict_mode', False)  # Disable strict semantic filtering by default
        
        # Semantic class compatibility matrix
        self._setup_class_compatibility()
        
        # Performance stats
        self.filtering_stats = {
            'total_pairs_processed': 0,
            'pairs_filtered_out': 0,
            'matches_filtered_out': 0,
            'semantic_consistency_rate': 0.0
        }
        
        logger.info(f"Semantic Filter initialized with consistency threshold: {self.consistency_threshold}")
    
    def _setup_class_compatibility(self):
        """
        Setup semantic class compatibility matrix for ADE20K dataset
        Defines which semantic classes can have valid feature matches between them
        """
        # ADE20K class groups that are typically compatible for feature matching
        self.compatible_groups = {
            # Structural elements - can match with each other
            'structures': {1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 19, 25, 27, 32, 34, 46, 50, 67, 75, 81, 82, 86, 95, 98, 102, 119},  # building, wall, house, door, window, table, ceiling, floor, etc.
            
            # Natural outdoor elements  
            'nature': {17, 21, 96, 123, 124, 134},  # tree, grass, plant, flower, etc.
            
            # Ground/terrain elements
            'terrain': {3, 6, 12, 15, 29, 52, 53, 93, 94},  # floor, road, earth, field, path, etc.
            
            # Water elements
            'water': {22, 28, 60, 128},  # water, sea, river, etc.
            
            # Vehicles and transportation
            'vehicles': {20, 77, 83, 84, 103, 125},  # car, boat, bus, truck, etc.
            
            # Sky and atmospheric
            'sky': {2, 16},  # sky, cloud
            
            # Human and clothing
            'human': {12, 18, 24, 92, 107, 115, 120, 142},  # person, head, body, clothing, etc.
        }
        
        # Create compatibility matrix
        self.compatibility_matrix = {}
        all_classes = set()
        for group_classes in self.compatible_groups.values():
            all_classes.update(group_classes)
        
        # Initialize all classes as incompatible
        for class1 in range(150):  # ADE20K has 150 classes
            self.compatibility_matrix[class1] = set()
        
        # Set compatible classes within each group
        for group_name, group_classes in self.compatible_groups.items():
            for class1 in group_classes:
                for class2 in group_classes:
                    if class1 != class2:
                        if class1 not in self.compatibility_matrix:
                            self.compatibility_matrix[class1] = set()
                        self.compatibility_matrix[class1].add(class2)
        
        # Special compatibility rules
        self._add_special_compatibility_rules()
        
        logger.info(f"Setup compatibility matrix for {len(self.compatible_groups)} semantic groups")
    
    def _add_special_compatibility_rules(self):
        """Add special cross-group compatibility rules"""
        
        # Structures can match with terrain (e.g., building-ground junction)
        structures = self.compatible_groups['structures']
        terrain = self.compatible_groups['terrain']
        
        for struct_class in structures:
            if struct_class not in self.compatibility_matrix:
                self.compatibility_matrix[struct_class] = set()
            self.compatibility_matrix[struct_class].update(terrain)
        
        for terrain_class in terrain:
            if terrain_class not in self.compatibility_matrix:
                self.compatibility_matrix[terrain_class] = set()
            self.compatibility_matrix[terrain_class].update(structures)
        
        # Nature can match with terrain
        nature = self.compatible_groups['nature']
        for nature_class in nature:
            if nature_class not in self.compatibility_matrix:
                self.compatibility_matrix[nature_class] = set()
            self.compatibility_matrix[nature_class].update(terrain)
        
        for terrain_class in terrain:
            self.compatibility_matrix[terrain_class].update(nature)
        
        # MORE PERMISSIVE: Add cross-group compatibility for common scene elements
        # Structures can match with nature (buildings with trees, etc.)
        for struct_class in structures:
            self.compatibility_matrix[struct_class].update(nature)
        for nature_class in nature:
            self.compatibility_matrix[nature_class].update(structures)
        
        # Vehicles can match with terrain and structures
        vehicles = self.compatible_groups['vehicles']
        for vehicle_class in vehicles:
            if vehicle_class not in self.compatibility_matrix:
                self.compatibility_matrix[vehicle_class] = set()
            self.compatibility_matrix[vehicle_class].update(terrain)
            self.compatibility_matrix[vehicle_class].update(structures)
        
        for terrain_class in terrain:
            self.compatibility_matrix[terrain_class].update(vehicles)
        for struct_class in structures:
            self.compatibility_matrix[struct_class].update(vehicles)
        
        logger.info("Added MORE PERMISSIVE cross-group compatibility rules")
    
    def _get_semantic_labels_at_keypoints(self, keypoints: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
        """
        Get semantic labels at keypoint locations
        
        Args:
            keypoints: Nx2 array of keypoint coordinates [x, y]
            semantic_mask: HxW semantic segmentation mask
            
        Returns:
            N-length array of semantic labels
        """
        if len(keypoints) == 0:
            return np.array([])
        
        # Ensure coordinates are within image bounds
        h, w = semantic_mask.shape
        x_coords = np.clip(keypoints[:, 0].astype(int), 0, w - 1)
        y_coords = np.clip(keypoints[:, 1].astype(int), 0, h - 1)
        
        # Extract semantic labels
        labels = semantic_mask[y_coords, x_coords]
        
        return labels
    
    def _check_semantic_consistency(self, labels1: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Check semantic consistency between matched keypoints
        
        Args:
            labels1: Semantic labels for keypoints in image 1
            labels2: Semantic labels for keypoints in image 2
            
        Returns:
            (consistency_mask, consistency_rate): Boolean mask of consistent matches and overall rate
        """
        if len(labels1) != len(labels2) or len(labels1) == 0:
            return np.array([]), 0.0
        
        consistency_mask = np.zeros(len(labels1), dtype=bool)
        
        for i, (label1, label2) in enumerate(zip(labels1, labels2)):
            # Same class is always consistent
            if label1 == label2:
                consistency_mask[i] = True
            # Check compatibility matrix
            elif label1 in self.compatibility_matrix and label2 in self.compatibility_matrix[label1]:
                consistency_mask[i] = True
            # In non-strict mode, allow some flexibility for unknown classes
            elif not self.strict_mode and (label1 >= 150 or label2 >= 150):
                consistency_mask[i] = True
        
        consistency_rate = consistency_mask.sum() / len(consistency_mask) if len(consistency_mask) > 0 else 0.0
        
        return consistency_mask, consistency_rate
    
    def filter_matches_semantic_consistency(self, 
                                          matches: Dict[Tuple[str, str], Any],
                                          semantic_masks: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Any]:
        """
        Filter matches based on semantic consistency
        
        Args:
            matches: Dictionary of feature matches
            semantic_masks: Dictionary of semantic segmentation masks
            
        Returns:
            Filtered matches dictionary
        """
        if not matches or not semantic_masks:
            return matches
        
        logger.info(f"Applying semantic consistency filtering to {len(matches)} match pairs...")
        
        filtered_matches = {}
        total_matches_before = sum(len(match_data.get('matches0', [])) for match_data in matches.values())
        total_matches_after = 0
        
        for (img1_path, img2_path), match_data in matches.items():
            self.filtering_stats['total_pairs_processed'] += 1
            
            # Check if we have semantic masks for both images
            if img1_path not in semantic_masks or img2_path not in semantic_masks:
                # Keep matches if semantic masks not available
                filtered_matches[(img1_path, img2_path)] = match_data
                total_matches_after += len(match_data.get('matches0', []))
                continue
            
            try:
                # Get matched keypoints
                keypoints1 = match_data.get('keypoints0', np.array([]))
                keypoints2 = match_data.get('keypoints1', np.array([]))
                matches0 = match_data.get('matches0', np.array([]))
                matches1 = match_data.get('matches1', np.array([]))
                
                if len(matches0) == 0 or len(matches1) == 0:
                    continue
                
                # Get matched keypoint coordinates
                matched_kpts1 = keypoints1[matches0]
                matched_kpts2 = keypoints2[matches1]
                
                # Get semantic labels at keypoint locations
                labels1 = self._get_semantic_labels_at_keypoints(matched_kpts1, semantic_masks[img1_path])
                labels2 = self._get_semantic_labels_at_keypoints(matched_kpts2, semantic_masks[img2_path])
                
                # Check semantic consistency
                consistency_mask, consistency_rate = self._check_semantic_consistency(labels1, labels2)
                
                # Apply filtering based on consistency threshold
                if consistency_rate >= self.consistency_threshold and consistency_mask.sum() >= self.min_consistent_matches:
                    # Keep only consistent matches
                    consistent_indices = np.where(consistency_mask)[0]
                    
                    filtered_match_data = match_data.copy()
                    filtered_match_data['matches0'] = matches0[consistent_indices]
                    filtered_match_data['matches1'] = matches1[consistent_indices]
                    
                    # Update match scores if available
                    if 'mscores0' in match_data:
                        filtered_match_data['mscores0'] = match_data['mscores0'][consistent_indices]
                    if 'mscores1' in match_data:
                        filtered_match_data['mscores1'] = match_data['mscores1'][consistent_indices]
                    
                    filtered_matches[(img1_path, img2_path)] = filtered_match_data
                    total_matches_after += len(consistent_indices)
                    
                    logger.debug(f"Pair {Path(img1_path).name}-{Path(img2_path).name}: "
                               f"{len(matches0)} -> {len(consistent_indices)} matches "
                               f"(consistency: {consistency_rate:.2f})")
                else:
                    # Filter out entire pair if consistency too low
                    self.filtering_stats['pairs_filtered_out'] += 1
                    logger.debug(f"Filtered out pair {Path(img1_path).name}-{Path(img2_path).name}: "
                               f"consistency {consistency_rate:.2f} < {self.consistency_threshold}")
                    
            except Exception as e:
                logger.warning(f"Error filtering matches for pair {Path(img1_path).name}-{Path(img2_path).name}: {e}")
                # Keep original matches on error
                filtered_matches[(img1_path, img2_path)] = match_data
                total_matches_after += len(match_data.get('matches0', []))
        
        # Update statistics
        matches_filtered_out = total_matches_before - total_matches_after
        self.filtering_stats['matches_filtered_out'] += matches_filtered_out
        self.filtering_stats['semantic_consistency_rate'] = (
            total_matches_after / total_matches_before if total_matches_before > 0 else 1.0
        )
        
        retention_rate = len(filtered_matches) / len(matches) * 100 if matches else 0
        match_retention_rate = total_matches_after / total_matches_before * 100 if total_matches_before > 0 else 0
        
        logger.info(f"Semantic filtering completed:")
        logger.info(f"  Pair retention: {len(filtered_matches)}/{len(matches)} ({retention_rate:.1f}%)")
        logger.info(f"  Match retention: {total_matches_after}/{total_matches_before} ({match_retention_rate:.1f}%)")
        logger.info(f"  Matches filtered out: {matches_filtered_out}")
        
        return filtered_matches
    
    def filter_matches_hierarchical(self, 
                                   matches: Dict[Tuple[str, str], Any],
                                   semantic_masks: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Any]:
        """
        Apply hierarchical semantic filtering
        First removes obviously incompatible matches, then applies consistency filtering
        
        Args:
            matches: Dictionary of feature matches
            semantic_masks: Dictionary of semantic segmentation masks
            
        Returns:
            Hierarchically filtered matches
        """
        logger.info("Applying hierarchical semantic filtering...")
        
        # Step 1: Remove obviously incompatible matches (coarse filtering)
        coarse_filtered = self._apply_coarse_semantic_filter(matches, semantic_masks)
        
        # Step 2: Apply fine-grained consistency filtering
        fine_filtered = self.filter_matches_semantic_consistency(coarse_filtered, semantic_masks)
        
        return fine_filtered
    
    def _apply_coarse_semantic_filter(self,
                                    matches: Dict[Tuple[str, str], Any],
                                    semantic_masks: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Any]:
        """
        Apply coarse semantic filtering to remove obviously bad matches
        """
        logger.info("Applying coarse semantic filtering...")
        
        # Define obviously incompatible class pairs
        incompatible_pairs = [
            # Sky vs ground-level objects
            (2, 3), (2, 6), (2, 12), (2, 20),  # sky vs floor, road, person, car
            (16, 3), (16, 6), (16, 12), (16, 20),  # cloud vs floor, road, person, car
            
            # Water vs dry objects  
            (22, 1), (22, 4), (22, 7), (22, 20),  # water vs building, house, table, car
            
            # Human vs structural
            (12, 2), (12, 16), (12, 22),  # person vs sky, cloud, water
        ]
        
        coarse_filtered = {}
        
        for (img1_path, img2_path), match_data in matches.items():
            if img1_path not in semantic_masks or img2_path not in semantic_masks:
                coarse_filtered[(img1_path, img2_path)] = match_data
                continue
            
            try:
                keypoints1 = match_data.get('keypoints0', np.array([]))
                keypoints2 = match_data.get('keypoints1', np.array([]))
                matches0 = match_data.get('matches0', np.array([]))
                matches1 = match_data.get('matches1', np.array([]))
                
                if len(matches0) == 0:
                    continue
                
                matched_kpts1 = keypoints1[matches0]
                matched_kpts2 = keypoints2[matches1]
                
                labels1 = self._get_semantic_labels_at_keypoints(matched_kpts1, semantic_masks[img1_path])
                labels2 = self._get_semantic_labels_at_keypoints(matched_kpts2, semantic_masks[img2_path])
                
                # Filter out obviously incompatible matches
                keep_mask = np.ones(len(matches0), dtype=bool)
                
                for i, (label1, label2) in enumerate(zip(labels1, labels2)):
                    for incompatible_pair in incompatible_pairs:
                        if ((label1 == incompatible_pair[0] and label2 == incompatible_pair[1]) or
                            (label1 == incompatible_pair[1] and label2 == incompatible_pair[0])):
                            keep_mask[i] = False
                            break
                
                if keep_mask.sum() >= self.min_consistent_matches:
                    # Keep filtered matches
                    filtered_match_data = match_data.copy()
                    filtered_match_data['matches0'] = matches0[keep_mask]
                    filtered_match_data['matches1'] = matches1[keep_mask]
                    
                    if 'mscores0' in match_data:
                        filtered_match_data['mscores0'] = match_data['mscores0'][keep_mask]
                    if 'mscores1' in match_data:
                        filtered_match_data['mscores1'] = match_data['mscores1'][keep_mask]
                    
                    coarse_filtered[(img1_path, img2_path)] = filtered_match_data
                    
            except Exception as e:
                logger.warning(f"Error in coarse filtering: {e}")
                coarse_filtered[(img1_path, img2_path)] = match_data
        
        logger.info(f"Coarse filtering: {len(matches)} -> {len(coarse_filtered)} pairs")
        return coarse_filtered
    
    def get_semantic_statistics(self, 
                              matches: Dict[Tuple[str, str], Any],
                              semantic_masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Get statistics about semantic distribution in matches
        
        Args:
            matches: Dictionary of feature matches
            semantic_masks: Dictionary of semantic segmentation masks
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_pairs': len(matches),
            'pairs_with_semantic_data': 0,
            'semantic_class_distribution': defaultdict(int),
            'cross_class_matches': defaultdict(int),
            'consistency_rates': []
        }
        
        for (img1_path, img2_path), match_data in matches.items():
            if img1_path not in semantic_masks or img2_path not in semantic_masks:
                continue
                
            stats['pairs_with_semantic_data'] += 1
            
            try:
                keypoints1 = match_data.get('keypoints0', np.array([]))
                keypoints2 = match_data.get('keypoints1', np.array([]))
                matches0 = match_data.get('matches0', np.array([]))
                matches1 = match_data.get('matches1', np.array([]))
                
                if len(matches0) == 0:
                    continue
                
                matched_kpts1 = keypoints1[matches0]
                matched_kpts2 = keypoints2[matches1]
                
                labels1 = self._get_semantic_labels_at_keypoints(matched_kpts1, semantic_masks[img1_path])
                labels2 = self._get_semantic_labels_at_keypoints(matched_kpts2, semantic_masks[img2_path])
                
                # Count class distribution
                for label in labels1:
                    stats['semantic_class_distribution'][int(label)] += 1
                for label in labels2:
                    stats['semantic_class_distribution'][int(label)] += 1
                
                # Count cross-class matches
                for label1, label2 in zip(labels1, labels2):
                    pair_key = f"{min(label1, label2)}_{max(label1, label2)}"
                    stats['cross_class_matches'][pair_key] += 1
                
                # Calculate consistency rate for this pair
                _, consistency_rate = self._check_semantic_consistency(labels1, labels2)
                stats['consistency_rates'].append(consistency_rate)
                
            except Exception as e:
                logger.warning(f"Error calculating semantic statistics: {e}")
        
        # Calculate summary statistics
        if stats['consistency_rates']:
            stats['avg_consistency_rate'] = np.mean(stats['consistency_rates'])
            stats['median_consistency_rate'] = np.median(stats['consistency_rates'])
            stats['min_consistency_rate'] = np.min(stats['consistency_rates'])
            stats['max_consistency_rate'] = np.max(stats['consistency_rates'])
        
        return dict(stats)
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """Get current filtering statistics"""
        return self.filtering_stats.copy()
    
    def reset_stats(self):
        """Reset filtering statistics"""
        self.filtering_stats = {
            'total_pairs_processed': 0,
            'pairs_filtered_out': 0,
            'matches_filtered_out': 0,
            'semantic_consistency_rate': 0.0
        }


def create_semantic_filter(config: Dict[str, Any] = None) -> SemanticFilter:
    """
    Factory function to create semantic filter
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SemanticFilter instance
    """
    return SemanticFilter(config)


def filter_matches_with_semantics(matches: Dict[Tuple[str, str], Any],
                                 semantic_masks: Dict[str, np.ndarray],
                                 config: Dict[str, Any] = None) -> Dict[Tuple[str, str], Any]:
    """
    Convenience function to apply semantic filtering to matches
    
    Args:
        matches: Dictionary of feature matches
        semantic_masks: Dictionary of semantic segmentation masks
        config: Filtering configuration
        
    Returns:
        Semantically filtered matches
    """
    semantic_filter = create_semantic_filter(config)
    
    if config and config.get('use_hierarchical_filtering', True):
        return semantic_filter.filter_matches_hierarchical(matches, semantic_masks)
    else:
        return semantic_filter.filter_matches_semantic_consistency(matches, semantic_masks)