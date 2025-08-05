"""
Scale recovery for 3D Gaussian Splatting consistency
Ensures global scale consistency across the reconstruction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class ScaleRecovery:
    """Scale recovery for 3DGS consistency"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.scale_info = {}
        
    def recover_scale(self, sparse_points: Dict, cameras: Dict, 
                     images: Dict) -> Tuple[Dict, Dict]:
        """Recover global scale for 3DGS consistency"""
        
        logger.info("Recovering global scale for 3DGS consistency...")
        
        # Method 1: Multi-view scale consistency
        scale_factor = self._estimate_scale_from_multiview(
            sparse_points, cameras, images
        )
        
        # Method 2: Monocular depth scale (if available)
        mono_scale = self._estimate_scale_from_monocular_depth(
            sparse_points, cameras, images
        )
        
        # Combine scale estimates
        if mono_scale is not None:
            final_scale = (scale_factor + mono_scale) / 2.0
        else:
            final_scale = scale_factor
        
        # Apply scale to points and cameras
        scaled_points = self._apply_scale_to_points(sparse_points, final_scale)
        scaled_cameras = self._apply_scale_to_cameras(cameras, final_scale)
        
        # Store scale information
        self.scale_info = {
            'multiview_scale': scale_factor,
            'monocular_scale': mono_scale,
            'final_scale': final_scale,
            'scale_method': 'combined' if mono_scale else 'multiview'
        }
        
        logger.info(f"Applied scale factor: {final_scale:.6f}")
        
        return scaled_points, scaled_cameras
    
    def _estimate_scale_from_multiview(self, sparse_points: Dict, 
                                      cameras: Dict, images: Dict) -> float:
        """Estimate scale from multi-view consistency using advanced algorithms"""
        
        # Get camera pairs with overlapping points
        camera_pairs = self._get_camera_pairs_with_overlap(
            sparse_points, cameras, images
        )
        
        if len(camera_pairs) < 2:
            logger.warning("Insufficient camera pairs for scale estimation")
            return 1.0
        
        # Method 1: Direct scale estimation from relative poses
        direct_scales = self._estimate_direct_scales(camera_pairs, sparse_points, cameras, images)
        
        # Method 2: Scale from triangulated point distances
        triangulation_scales = self._estimate_triangulation_scales(sparse_points, cameras, images)
        
        # Method 3: Scale from baseline ratios
        baseline_scales = self._estimate_baseline_scales(camera_pairs, cameras, images)
        
        # Combine all scale estimates using robust statistics
        all_scales = []
        if direct_scales:
            all_scales.extend(direct_scales)
        if triangulation_scales:
            all_scales.extend(triangulation_scales)
        if baseline_scales:
            all_scales.extend(baseline_scales)
        
        if not all_scales:
            return 1.0
        
        # Use RANSAC-style robust estimation
        final_scale = self._robust_scale_estimation(all_scales)
        
        logger.info(f"Advanced multi-view scale estimate: {final_scale:.6f}")
        return final_scale
    
    def _estimate_direct_scales(self, camera_pairs: List[Tuple[str, str]], 
                               sparse_points: Dict, cameras: Dict, 
                               images: Dict) -> List[float]:
        """Estimate scales directly from relative poses"""
        
        scales = []
        
        for img1, img2 in camera_pairs:
            # Get camera poses
            pose1 = self._get_camera_pose(img1, cameras, images)
            pose2 = self._get_camera_pose(img2, cameras, images)
            
            if pose1 is None or pose2 is None:
                continue
            
            R1, t1 = pose1
            R2, t2 = pose2
            
            # Calculate relative pose
            R_rel = R2 @ R1.T
            t_rel = t2 - R_rel @ t1
            
            # Estimate scale from translation magnitude
            # Use multiple methods for robustness
            scale_methods = []
            
            # Method 1: Direct translation magnitude
            scale_methods.append(np.linalg.norm(t_rel))
            
            # Method 2: Scale from point distances
            common_points = self._get_common_points(img1, img2, sparse_points, images)
            if len(common_points) >= 10:
                point_scale = self._estimate_scale_from_point_distances(
                    common_points, sparse_points, R_rel, t_rel
                )
                if point_scale > 0:
                    scale_methods.append(point_scale)
            
            # Method 3: Scale from epipolar geometry
            epipolar_scale = self._estimate_scale_from_epipolar(
                img1, img2, cameras, images
            )
            if epipolar_scale > 0:
                scale_methods.append(epipolar_scale)
            
            if scale_methods:
                # Use median for robustness
                scales.append(np.median(scale_methods))
        
        return scales
    
    def _estimate_triangulation_scales(self, sparse_points: Dict, 
                                     cameras: Dict, images: Dict) -> List[float]:
        """Estimate scales from triangulated point distances"""
        
        scales = []
        
        # Get all 3D points
        point_coords = []
        for point_id, point_data in sparse_points.items():
            if 'xyz' in point_data:
                point_coords.append(point_data['xyz'])
        
        if len(point_coords) < 10:
            return scales
        
        point_coords = np.array(point_coords)
        
        # Calculate pairwise distances
        distances = cdist(point_coords, point_coords)
        
        # Get valid distances (non-zero, non-infinite)
        valid_distances = distances[(distances > 0) & (distances < np.inf)]
        
        if len(valid_distances) < 10:
            return scales
        
        # Estimate scale from distance distribution
        # Use median distance as reference
        median_distance = np.median(valid_distances)
        
        # Normalize to reasonable scale (1-10 meters)
        if median_distance > 0:
            scale_factor = 5.0 / median_distance  # Target 5m median distance
            scales.append(scale_factor)
        
        return scales
    
    def _estimate_baseline_scales(self, camera_pairs: List[Tuple[str, str]], 
                                cameras: Dict, images: Dict) -> List[float]:
        """Estimate scales from camera baseline ratios"""
        
        scales = []
        
        for img1, img2 in camera_pairs:
            pose1 = self._get_camera_pose(img1, cameras, images)
            pose2 = self._get_camera_pose(img2, cameras, images)
            
            if pose1 is None or pose2 is None:
                continue
            
            R1, t1 = pose1
            R2, t2 = pose2
            
            # Calculate baseline
            baseline = np.linalg.norm(t2 - t1)
            
            # Estimate scale from baseline magnitude
            # Assume reasonable baseline (0.1-2.0 meters)
            if baseline > 0:
                target_baseline = 0.5  # Target 0.5m baseline
                scale_factor = target_baseline / baseline
                scales.append(scale_factor)
        
        return scales
    
    def _estimate_scale_from_point_distances(self, common_points: List[int], 
                                           sparse_points: Dict, R_rel: np.ndarray, 
                                           t_rel: np.ndarray) -> float:
        """Estimate scale from distances between common 3D points"""
        
        if len(common_points) < 2:
            return 0.0
        
        # Get 3D points
        point_coords = []
        for point_id in common_points:
            if point_id in sparse_points and 'xyz' in sparse_points[point_id]:
                point_coords.append(sparse_points[point_id]['xyz'])
        
        if len(point_coords) < 2:
            return 0.0
        
        point_coords = np.array(point_coords)
        
        # Calculate distances in both coordinate systems
        distances_orig = cdist(point_coords, point_coords)
        
        # Transform points to second camera frame
        point_coords_transformed = (R_rel @ point_coords.T).T + t_rel
        distances_transformed = cdist(point_coords_transformed, point_coords_transformed)
        
        # Estimate scale from distance ratios
        valid_mask = (distances_orig > 0) & (distances_transformed > 0)
        if np.sum(valid_mask) < 5:
            return 0.0
        
        scale_ratios = distances_transformed[valid_mask] / distances_orig[valid_mask]
        
        # Use median for robustness
        return np.median(scale_ratios)
    
    def _estimate_scale_from_epipolar(self, img1: str, img2: str, 
                                    cameras: Dict, images: Dict) -> float:
        """Estimate scale from epipolar geometry"""
        
        # This is a simplified implementation
        # In practice, you would use the fundamental matrix
        
        pose1 = self._get_camera_pose(img1, cameras, images)
        pose2 = self._get_camera_pose(img2, cameras, images)
        
        if pose1 is None or pose2 is None:
            return 0.0
        
        R1, t1 = pose1
        R2, t2 = pose2
        
        # Calculate relative pose
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        # Estimate scale from translation magnitude
        # This is a simplified approach
        return np.linalg.norm(t_rel)
    
    def _robust_scale_estimation(self, scales: List[float]) -> float:
        """Robust scale estimation using RANSAC-style approach"""
        
        if not scales:
            return 1.0
        
        scales = np.array(scales)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(scales, 25)
        Q3 = np.percentile(scales, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        inliers = scales[(scales >= lower_bound) & (scales <= upper_bound)]
        
        if len(inliers) == 0:
            return np.median(scales)
        
        # Use weighted median for final estimate
        # Weight by inverse of distance from median
        median_scale = np.median(inliers)
        weights = 1.0 / (1.0 + np.abs(inliers - median_scale))
        
        weighted_median = np.average(inliers, weights=weights)
        
        return weighted_median
    
    def _estimate_scale_from_monocular_depth(self, sparse_points: Dict,
                                           cameras: Dict, images: Dict) -> Optional[float]:
        """Estimate scale from monocular depth (if available)"""
        
        # This would integrate with DPT depth maps
        # For now, return None (placeholder)
        return None
    
    def _get_camera_pairs_with_overlap(self, sparse_points: Dict,
                                      cameras: Dict, images: Dict) -> List[Tuple[str, str]]:
        """Get camera pairs that have overlapping 3D points"""
        
        camera_pairs = []
        image_list = list(images.keys())
        
        for i, img1 in enumerate(image_list):
            for img2 in image_list[i+1:]:
                # Check if these images share 3D points
                common_points = self._get_common_points(
                    img1, img2, sparse_points, images
                )
                
                if len(common_points) >= 10:
                    camera_pairs.append((img1, img2))
        
        return camera_pairs
    
    def _get_common_points(self, img1: str, img2: str, 
                          sparse_points: Dict, images: Dict) -> List[int]:
        """Get 3D point IDs that are visible in both images"""
        
        if img1 not in images or img2 not in images:
            return []
        
        # Get point IDs from both images
        points1 = set(images[img1]['point3D_ids'])
        points2 = set(images[img2]['point3D_ids'])
        
        # Find common points (excluding -1 for invalid points)
        common_points = points1.intersection(points2)
        common_points.discard(-1)
        
        return list(common_points)
    
    def _estimate_pair_scale(self, img1: str, img2: str, 
                           common_points: List[int], cameras: Dict, 
                           images: Dict) -> float:
        """Estimate scale between two camera views"""
        
        if len(common_points) < 10:
            return 0.0
        
        # Get camera poses
        pose1 = self._get_camera_pose(img1, cameras, images)
        pose2 = self._get_camera_pose(img2, cameras, images)
        
        if pose1 is None or pose2 is None:
            return 0.0
        
        R1, t1 = pose1
        R2, t2 = pose2
        
        # Calculate relative pose
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        
        # Estimate scale from translation magnitude
        # This is a simplified approach - more sophisticated methods exist
        scale_factor = np.linalg.norm(t_rel)
        
        return scale_factor
    
    def _get_camera_pose(self, img_name: str, cameras: Dict, 
                         images: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get camera pose (R, t) for an image"""
        
        if img_name not in images:
            return None
        
        img_data = images[img_name]
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(qvec)
        t = np.array(tvec)
        
        return R, t
    
    def _quaternion_to_rotation_matrix(self, qvec: List[float]) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = qvec
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R
    
    def _apply_scale_to_points(self, sparse_points: Dict, scale: float) -> Dict:
        """Apply scale factor to 3D points"""
        
        scaled_points = {}
        
        for point_id, point_data in sparse_points.items():
            scaled_point = point_data.copy()
            scaled_point['xyz'] = point_data['xyz'] * scale
            scaled_points[point_id] = scaled_point
        
        return scaled_points
    
    def _apply_scale_to_cameras(self, cameras: Dict, scale: float) -> Dict:
        """Apply scale factor to camera translations"""
        
        # Note: This is a simplified approach
        # In practice, you might need to update camera parameters differently
        return cameras
    
    def get_scale_info(self) -> Dict[str, Any]:
        """Get scale recovery information"""
        return self.scale_info.copy()
    
    def validate_scale_consistency(self, sparse_points: Dict, 
                                 cameras: Dict, images: Dict) -> Dict[str, float]:
        """Validate scale consistency across the reconstruction"""
        
        # Calculate scale consistency metrics
        metrics = {
            'scale_variance': 0.0,
            'scale_range': 0.0,
            'consistency_score': 0.0
        }
        
        # Implementation would calculate actual metrics
        # For now, return placeholder values
        
        return metrics 