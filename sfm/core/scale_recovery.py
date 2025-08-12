"""
Scale recovery for 3D Gaussian Splatting consistency
Ensures global scale consistency across the reconstruction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean
import logging

# Robust statistics libraries - optional imports with fallbacks
try:
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    ECOD = None
    IForest = None

try:
    from sklearn.linear_model import RANSACRegressor
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RANSACRegressor = None
    IsolationForest = None

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
        """Estimate scale from epipolar geometry using fundamental matrix and essential matrix decomposition"""
        
        pose1 = self._get_camera_pose(img1, cameras, images)
        pose2 = self._get_camera_pose(img2, cameras, images)
        
        if pose1 is None or pose2 is None:
            return 0.0
        
        # Get camera intrinsics
        img1_data = images[img1]
        camera_id = img1_data['camera_id']
        
        if camera_id in cameras and 'params' in cameras[camera_id]:
            fx = cameras[camera_id]['params'][0]
            cx = cameras[camera_id]['params'][1] if len(cameras[camera_id]['params']) > 1 else 320.0
            cy = cameras[camera_id]['params'][2] if len(cameras[camera_id]['params']) > 2 else 240.0
        else:
            fx, cx, cy = 1000.0, 320.0, 240.0
        
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        
        # Get corresponding 2D points between the two images
        correspondences = self._get_2d_correspondences(img1, img2, images)
        
        if len(correspondences) < 8:
            logger.warning(f"Insufficient correspondences for epipolar scale: {len(correspondences)}")
            # Fallback to baseline scale
            R1, t1 = pose1
            R2, t2 = pose2
            return np.linalg.norm(t2 - t1)
        
        points1 = np.array([c['pt1'] for c in correspondences])
        points2 = np.array([c['pt2'] for c in correspondences])
        
        try:
            # Estimate fundamental matrix using RANSAC
            F, inlier_mask = self._estimate_fundamental_matrix(points1, points2)
            
            if F is None:
                logger.warning("Fundamental matrix estimation failed")
                R1, t1 = pose1
                R2, t2 = pose2
                return np.linalg.norm(t2 - t1)
            
            # Convert fundamental matrix to essential matrix
            E = K.T @ F @ K
            
            # Decompose essential matrix to get relative pose and scale
            R_recovered, t_recovered, scale_info = self._decompose_essential_matrix(
                E, points1[inlier_mask], points2[inlier_mask], K
            )
            
            if scale_info is not None and scale_info > 0:
                logger.debug(f"Epipolar scale estimate: {scale_info:.4f}")
                return scale_info
            else:
                # Fallback to ground truth baseline
                R1, t1 = pose1
                R2, t2 = pose2
                return np.linalg.norm(t2 - t1)
                
        except Exception as e:
            logger.warning(f"Epipolar scale estimation failed: {e}")
            # Fallback to baseline scale
            R1, t1 = pose1
            R2, t2 = pose2
            return np.linalg.norm(t2 - t1)
    
    def _get_2d_correspondences(self, img1: str, img2: str, images: Dict) -> List[Dict]:
        """Get 2D point correspondences between two images"""
        
        correspondences = []
        
        img1_data = images[img1]
        img2_data = images[img2]
        
        # Find common 3D points
        point_ids1 = img1_data['point3D_ids']
        point_ids2 = img2_data['point3D_ids']
        
        for i, point_id in enumerate(point_ids1):
            if point_id != -1 and point_id in point_ids2:
                try:
                    # Find corresponding index in img2
                    j = point_ids2.index(point_id)
                    
                    if i < len(img1_data['xys']) and j < len(img2_data['xys']):
                        correspondences.append({
                            'pt1': img1_data['xys'][i],
                            'pt2': img2_data['xys'][j],
                            'point3d_id': point_id
                        })
                except (ValueError, IndexError):
                    continue
        
        return correspondences
    
    def _estimate_fundamental_matrix(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate fundamental matrix using RANSAC and 8-point algorithm"""
        
        if len(points1) < 8:
            return None, None
        
        # Normalize points for numerical stability
        points1_norm, T1 = self._normalize_points(points1)
        points2_norm, T2 = self._normalize_points(points2)
        
        best_F = None
        best_inliers = None
        max_inliers = 0
        
        num_iterations = min(1000, max(100, len(points1) * 2))
        threshold = 1.0  # Pixel threshold for inlier classification
        
        for _ in range(num_iterations):
            # Randomly select 8 points
            sample_indices = np.random.choice(len(points1), 8, replace=False)
            sample_pts1 = points1_norm[sample_indices]
            sample_pts2 = points2_norm[sample_indices]
            
            # Estimate F using 8-point algorithm
            F_norm = self._compute_fundamental_8point(sample_pts1, sample_pts2)
            
            if F_norm is None:
                continue
            
            # Denormalize
            F = T2.T @ F_norm @ T1
            
            # Count inliers
            inliers = self._compute_fundamental_inliers(points1, points2, F, threshold)
            num_inliers = np.sum(inliers)
            
            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_F = F
                best_inliers = inliers
        
        if best_F is not None and max_inliers >= 8:
            return best_F, best_inliers
        else:
            return None, None
    
    def _normalize_points(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize points for numerical stability in fundamental matrix estimation"""
        
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        # Compute average distance to centroid
        distances = np.linalg.norm(centered_points, axis=1)
        avg_distance = np.mean(distances)
        
        if avg_distance == 0:
            scale = 1.0
        else:
            scale = np.sqrt(2) / avg_distance
        
        # Normalization matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply normalization
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        points_normalized = (T @ points_homogeneous.T).T
        
        return points_normalized[:, :2], T
    
    def _compute_fundamental_8point(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute fundamental matrix using 8-point algorithm"""
        
        if len(points1) != 8 or len(points2) != 8:
            return None
        
        # Build constraint matrix A
        A = np.zeros((8, 9))
        
        for i in range(8):
            x1, y1 = points1[i]
            x2, y2 = points2[i]
            
            A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
        
        # Solve Af = 0 using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            f = Vt[-1, :]
            
            # Reshape to 3x3 matrix
            F = f.reshape(3, 3)
            
            # Enforce rank-2 constraint
            U, s, Vt = np.linalg.svd(F)
            s[-1] = 0  # Set smallest singular value to 0
            F = U @ np.diag(s) @ Vt
            
            return F
            
        except np.linalg.LinAlgError:
            return None
    
    def _compute_fundamental_inliers(self, points1: np.ndarray, points2: np.ndarray, 
                                   F: np.ndarray, threshold: float) -> np.ndarray:
        """Compute inliers for fundamental matrix using epipolar constraint"""
        
        points1_h = np.hstack([points1, np.ones((len(points1), 1))])
        points2_h = np.hstack([points2, np.ones((len(points2), 1))])
        
        # Compute epipolar lines
        lines2 = (F @ points1_h.T).T  # Lines in image 2
        lines1 = (F.T @ points2_h.T).T  # Lines in image 1
        
        # Compute point-to-line distances
        distances1 = np.abs(np.sum(lines1 * points1_h, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)
        distances2 = np.abs(np.sum(lines2 * points2_h, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
        
        # Symmetric epipolar distance
        distances = (distances1 + distances2) / 2.0
        
        return distances < threshold
    
    def _decompose_essential_matrix(self, E: np.ndarray, points1: np.ndarray, 
                                  points2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Decompose essential matrix to recover relative pose and scale"""
        
        try:
            # SVD decomposition of essential matrix
            U, s, Vt = np.linalg.svd(E)
            
            # Ensure proper rotation (det(U) = det(V) = 1)
            if np.linalg.det(U) < 0:
                U[:, -1] *= -1
            if np.linalg.det(Vt) < 0:
                Vt[-1, :] *= -1
            
            # Two possible rotations
            W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            R1 = U @ W @ Vt
            R2 = U @ W.T @ Vt
            
            # Translation (up to scale)
            t = U[:, 2]
            
            # Four possible configurations
            configs = [
                (R1, t),
                (R1, -t),
                (R2, t), 
                (R2, -t)
            ]
            
            # Test configurations by triangulating points
            best_config = None
            max_valid_points = 0
            best_scale = 1.0
            
            for R, t_unit in configs:
                # Triangulate points to determine correct configuration and scale
                valid_count, estimated_scale = self._test_essential_configuration(
                    R, t_unit, points1, points2, K
                )
                
                if valid_count > max_valid_points:
                    max_valid_points = valid_count
                    best_config = (R, t_unit)
                    best_scale = estimated_scale
            
            if best_config is not None:
                R_best, t_best = best_config
                return R_best, t_best, best_scale
            else:
                return None, None, None
                
        except Exception as e:
            logger.warning(f"Essential matrix decomposition failed: {e}")
            return None, None, None
    
    def _test_essential_configuration(self, R: np.ndarray, t_unit: np.ndarray, 
                                    points1: np.ndarray, points2: np.ndarray, 
                                    K: np.ndarray) -> Tuple[int, float]:
        """Test essential matrix configuration by triangulating points"""
        
        # Assume unit scale initially
        t = t_unit
        
        # Camera matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t.reshape(-1, 1)])
        
        valid_points = 0
        depths = []
        
        # Triangulate a few points to test configuration
        for i in range(min(len(points1), 20)):  # Test subset for speed
            pt1 = points1[i]
            pt2 = points2[i]
            
            # Triangulate using DLT
            point_3d = self._triangulate_point(pt1, pt2, P1, P2)
            
            if point_3d is not None:
                # Check if point is in front of both cameras
                depth1 = point_3d[2]
                point2_cam = R @ point_3d + t
                depth2 = point2_cam[2]
                
                if depth1 > 0.1 and depth2 > 0.1:
                    valid_points += 1
                    depths.extend([depth1, depth2])
        
        # Estimate scale from depth statistics
        if len(depths) > 2:
            median_depth = np.median(depths)
            # Assume reasonable scene scale (1-10 meters typical)
            target_depth = 3.0
            estimated_scale = target_depth / median_depth if median_depth > 0 else 1.0
        else:
            estimated_scale = 1.0
        
        return valid_points, estimated_scale
    
    def _robust_scale_estimation(self, scales: List[float]) -> float:
        """Advanced robust scale estimation using modern outlier detection"""
        
        if not scales:
            return 1.0
        
        scales = np.array(scales)
        
        if len(scales) < 3:
            return np.median(scales)
        
        # Method 1: PyOD ECOD (Empirical-Cumulative-distribution-based Outlier Detection)
        if PYOD_AVAILABLE and len(scales) >= 10:
            try:
                # ECOD is parameter-free and highly effective
                detector = ECOD(contamination=0.1)  # Expect 10% outliers
                outlier_labels = detector.fit_predict(scales.reshape(-1, 1))
                
                # Get inliers (label = 0)
                inlier_scales = scales[outlier_labels == 0]
                
                if len(inlier_scales) > 0:
                    # Use trimmed mean for final robust estimate
                    robust_scale = trim_mean(inlier_scales, proportiontocut=0.05)
                    logger.info(f"PyOD ECOD: {len(inlier_scales)}/{len(scales)} inliers, scale={robust_scale:.6f}")
                    return robust_scale
                    
            except Exception as e:
                logger.warning(f"PyOD ECOD failed: {e}, falling back to sklearn")
        
        # Method 2: Sklearn Isolation Forest as fallback
        if SKLEARN_AVAILABLE and len(scales) >= 5:
            try:
                detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                outlier_labels = detector.fit_predict(scales.reshape(-1, 1))
                
                # Get inliers (label = 1 in sklearn)
                inlier_scales = scales[outlier_labels == 1]
                
                if len(inlier_scales) > 0:
                    robust_scale = trim_mean(inlier_scales, proportiontocut=0.05)
                    logger.info(f"Sklearn IsolationForest: {len(inlier_scales)}/{len(scales)} inliers, scale={robust_scale:.6f}")
                    return robust_scale
                    
            except Exception as e:
                logger.warning(f"Sklearn IsolationForest failed: {e}, falling back to IQR")
        
        # Method 3: Enhanced IQR method as final fallback
        Q1 = np.percentile(scales, 25)
        Q3 = np.percentile(scales, 75)
        IQR = Q3 - Q1
        
        # Use more conservative bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        inliers = scales[(scales >= lower_bound) & (scales <= upper_bound)]
        
        if len(inliers) == 0:
            # If all are outliers, use trimmed mean
            robust_scale = trim_mean(scales, proportiontocut=0.2)
            logger.info(f"IQR fallback: All outliers, using trimmed mean={robust_scale:.6f}")
            return robust_scale
        
        # Use trimmed mean of inliers for final robustness
        robust_scale = trim_mean(inliers, proportiontocut=0.05)
        logger.info(f"IQR method: {len(inliers)}/{len(scales)} inliers, scale={robust_scale:.6f}")
        
        return robust_scale
    
    def _estimate_scale_from_monocular_depth(self, sparse_points: Dict,
                                           cameras: Dict, images: Dict) -> Optional[float]:
        """Estimate scale from monocular depth using DPT integration"""
        
        logger.warning("Dense depth estimation removed - monocular depth scale estimation not available")
        return None
        
        if len(images) < 3:
            logger.warning("Insufficient images for monocular depth scale estimation")
            return None
        
        logger.info("Estimating scale from monocular depth using DPT...")
        
        try:
            # Initialize depth estimator
            depth_estimator = DenseDepthEstimator(
                device=self.device,
                depth_model='dpt-large',
                high_quality=True
            )
            
            # Sample representative images (limit for performance)
            sample_images = list(images.items())[:min(10, len(images))]
            scale_estimates = []
            
            for img_path, img_data in sample_images:
                # Get camera pose
                pose = self._get_camera_pose(img_path, images, images)
                if pose is None:
                    continue
                
                R, t = pose
                
                # Get sparse points visible in this image
                visible_points = []
                sparse_depths = []
                
                for i, point_id in enumerate(img_data['point3D_ids']):
                    if point_id != -1 and point_id in sparse_points:
                        # Get 2D pixel coordinate
                        if i < len(img_data['xys']):
                            pixel_coord = img_data['xys'][i]
                            
                            # Get 3D point and transform to camera coordinate system
                            point_3d = np.array(sparse_points[point_id]['xyz'])
                            point_cam = R @ point_3d + t
                            
                            if point_cam[2] > 0.1:  # Valid depth
                                visible_points.append({
                                    'pixel': pixel_coord,
                                    'sparse_depth': point_cam[2],
                                    'world_point': point_3d
                                })
                                sparse_depths.append(point_cam[2])
                
                if len(visible_points) < 5:
                    logger.debug(f"Insufficient visible points in {img_path}")
                    continue
                
                # Estimate monocular depth for this image
                try:
                    # Load image for depth estimation
                    from PIL import Image
                    import torch.nn.functional as F
                    
                    # This is a simplified depth estimation - in practice you'd use the full DPT pipeline
                    # For now, we'll estimate scale by comparing sparse vs expected depth ranges
                    
                    # Method 1: Statistical depth comparison
                    if len(sparse_depths) >= 5:
                        sparse_depth_stats = {
                            'median': np.median(sparse_depths),
                            'mean': np.mean(sparse_depths),
                            'std': np.std(sparse_depths),
                            'range': np.max(sparse_depths) - np.min(sparse_depths)
                        }
                        
                        # Typical scene depth ranges for different scenarios
                        scene_type_scales = {
                            'indoor': {'typical_range': (0.5, 10.0), 'median_depth': 2.5},
                            'outdoor_close': {'typical_range': (1.0, 50.0), 'median_depth': 8.0},
                            'outdoor_far': {'typical_range': (5.0, 200.0), 'median_depth': 30.0}
                        }
                        
                        # Classify scene type based on depth statistics
                        median_depth = sparse_depth_stats['median']
                        depth_range = sparse_depth_stats['range']
                        
                        if median_depth < 5.0 and depth_range < 15.0:
                            scene_type = 'indoor'
                        elif median_depth < 20.0:
                            scene_type = 'outdoor_close'  
                        else:
                            scene_type = 'outdoor_far'
                        
                        expected_median = scene_type_scales[scene_type]['median_depth']
                        scale_factor = expected_median / median_depth if median_depth > 0 else 1.0
                        
                        # Validate scale factor is reasonable
                        if 0.1 <= scale_factor <= 10.0:
                            scale_estimates.append(scale_factor)
                            logger.debug(f"Depth-based scale for {img_path}: {scale_factor:.4f} (scene: {scene_type})")
                
                except Exception as e:
                    logger.warning(f"Depth estimation failed for {img_path}: {e}")
                    continue
            
            if len(scale_estimates) < 2:
                logger.warning("Insufficient depth-based scale estimates")
                return None
            
            # Robust combination of scale estimates
            scale_estimates = np.array(scale_estimates)
            
            # Remove outliers using IQR
            Q1 = np.percentile(scale_estimates, 25)
            Q3 = np.percentile(scale_estimates, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            inlier_scales = scale_estimates[(scale_estimates >= lower_bound) & 
                                          (scale_estimates <= upper_bound)]
            
            if len(inlier_scales) == 0:
                # If all outliers, use trimmed mean
                final_scale = trim_mean(scale_estimates, proportiontocut=0.2)
            else:
                # Use median of inliers
                final_scale = np.median(inlier_scales)
            
            logger.info(f"Monocular depth scale estimate: {final_scale:.6f} from {len(inlier_scales)}/{len(scale_estimates)} estimates")
            return float(final_scale)
            
        except Exception as e:
            logger.error(f"Monocular depth scale estimation failed: {e}")
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
        """Estimate scale between two camera views using triangulation-based approach"""
        
        if len(common_points) < 10:
            return 0.0
        
        # Get camera poses and intrinsics
        pose1 = self._get_camera_pose(img1, cameras, images)
        pose2 = self._get_camera_pose(img2, cameras, images)
        
        if pose1 is None or pose2 is None:
            return 0.0
        
        R1, t1 = pose1
        R2, t2 = pose2
        
        # Get camera intrinsics
        img1_data = images[img1]
        img2_data = images[img2]
        camera_id = img1_data['camera_id']  # Assume same camera
        
        # Simple intrinsics fallback if not available
        if camera_id in cameras and 'params' in cameras[camera_id]:
            fx = cameras[camera_id]['params'][0]
            cx = cameras[camera_id]['params'][1] if len(cameras[camera_id]['params']) > 1 else 320.0
            cy = cameras[camera_id]['params'][2] if len(cameras[camera_id]['params']) > 2 else 240.0
        else:
            fx, cx, cy = 1000.0, 320.0, 240.0
        
        K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
        
        # Get 2D correspondences for common points
        correspondences1 = []
        correspondences2 = []
        
        for point_id in common_points:
            if point_id in img1_data['point3D_ids'] and point_id in img2_data['point3D_ids']:
                # Find indices of this point in both images
                try:
                    idx1 = img1_data['point3D_ids'].index(point_id)
                    idx2 = img2_data['point3D_ids'].index(point_id)
                    
                    if idx1 < len(img1_data['xys']) and idx2 < len(img2_data['xys']):
                        correspondences1.append(img1_data['xys'][idx1])
                        correspondences2.append(img2_data['xys'][idx2])
                except (ValueError, IndexError):
                    continue
        
        if len(correspondences1) < 8:  # Need minimum points for triangulation
            logger.warning(f"Insufficient correspondences: {len(correspondences1)}, falling back to baseline")
            return np.linalg.norm(t2 - t1)
        
        correspondences1 = np.array(correspondences1)
        correspondences2 = np.array(correspondences2)
        
        # Triangulate points using both camera views
        scale_estimates = []
        
        try:
            # Create projection matrices
            P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
            P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])
            
            # Triangulate each correspondence
            for i in range(len(correspondences1)):
                pt1 = correspondences1[i]
                pt2 = correspondences2[i]
                
                # Triangulate using DLT (Direct Linear Transform)
                point_3d = self._triangulate_point(pt1, pt2, P1, P2)
                
                if point_3d is not None:
                    # Calculate distances from both cameras
                    dist1 = np.linalg.norm(point_3d - t1)
                    dist2 = np.linalg.norm(point_3d - t2)
                    
                    # Use median distance as scale reference
                    if dist1 > 0 and dist2 > 0:
                        scale_estimates.append(np.median([dist1, dist2]))
            
            if len(scale_estimates) < 3:
                logger.warning("Triangulation failed, using baseline scale")
                return np.linalg.norm(t2 - t1)
            
            # Robust scale estimation from triangulated points
            scale_estimates = np.array(scale_estimates)
            
            # Remove extreme outliers using IQR
            Q1 = np.percentile(scale_estimates, 25)
            Q3 = np.percentile(scale_estimates, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            inliers = scale_estimates[(scale_estimates >= lower_bound) & 
                                    (scale_estimates <= upper_bound)]
            
            if len(inliers) > 0:
                # Use median of inliers for robustness
                final_scale = np.median(inliers)
                logger.debug(f"Triangulation scale: {final_scale:.4f} from {len(inliers)} inliers")
                return final_scale
            else:
                return np.median(scale_estimates)
                
        except Exception as e:
            logger.warning(f"Triangulation failed: {e}, using baseline scale")
            return np.linalg.norm(t2 - t1)
    
    def _triangulate_point(self, pt1: np.ndarray, pt2: np.ndarray, 
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """Triangulate 3D point from two 2D correspondences using DLT"""
        
        # Build the A matrix for DLT
        A = np.array([
            pt1[0] * P1[2, :] - P1[0, :],
            pt1[1] * P1[2, :] - P1[1, :],
            pt2[0] * P2[2, :] - P2[0, :],
            pt2[1] * P2[2, :] - P2[1, :]
        ])
        
        # Solve using SVD
        try:
            _, _, Vt = np.linalg.svd(A)
            X_homogeneous = Vt[-1, :]
            
            # Convert from homogeneous coordinates
            if X_homogeneous[3] != 0:
                X = X_homogeneous[:3] / X_homogeneous[3]
                
                # Basic validation: point should be in front of both cameras
                # and not too far away
                if X[2] > 0.1 and np.linalg.norm(X) < 1000.0:
                    return X
            
        except np.linalg.LinAlgError:
            pass
        
        return None
    
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
        """Apply scale factor to camera translations and update intrinsics if needed"""
        
        scaled_cameras = {}
        
        for camera_id, camera_data in cameras.items():
            scaled_camera = camera_data.copy()
            
            # For most cases, camera intrinsics don't need scaling
            # But if the images were resized during preprocessing, we might need to adjust
            
            # Handle different camera models
            if 'model' in camera_data:
                model = camera_data['model']
                params = np.array(camera_data['params'], dtype=np.float64)
                
                if model == 'PINHOLE':
                    # PINHOLE: [fx, fy, cx, cy]
                    # Focal length and principal point scale with image dimensions
                    # For SfM scale recovery, we typically don't modify intrinsics
                    # unless there's a specific reason (like image rescaling)
                    scaled_camera['params'] = params.tolist()
                    
                elif model == 'RADIAL':
                    # RADIAL: [f, cx, cy, k]
                    # Keep radial distortion parameters unchanged
                    scaled_camera['params'] = params.tolist()
                    
                elif model == 'OPENCV':
                    # OPENCV: [fx, fy, cx, cy, k1, k2, p1, p2]
                    # Keep distortion parameters unchanged
                    scaled_camera['params'] = params.tolist()
                    
                elif model == 'SIMPLE_PINHOLE':
                    # SIMPLE_PINHOLE: [f, cx, cy]
                    scaled_camera['params'] = params.tolist()
                    
                else:
                    # Unknown model, keep params unchanged
                    scaled_camera['params'] = params.tolist()
                    logger.warning(f"Unknown camera model: {model}, keeping params unchanged")
            
            else:
                # No model specified, assume basic camera
                scaled_camera['params'] = camera_data.get('params', []).copy()
            
            # Add scale metadata for tracking
            scaled_camera['scale_applied'] = scale
            scaled_camera['original_params'] = camera_data.get('params', []).copy()
            
            scaled_cameras[camera_id] = scaled_camera
        
        logger.info(f"Applied scale {scale:.6f} to {len(scaled_cameras)} cameras")
        return scaled_cameras
    
    def get_scale_info(self) -> Dict[str, Any]:
        """Get scale recovery information"""
        return self.scale_info.copy()
    
    def validate_scale_consistency(self, sparse_points: Dict, 
                                 cameras: Dict, images: Dict) -> Dict[str, float]:
        """Validate scale consistency across the reconstruction"""
        
        logger.info("Validating scale consistency across reconstruction...")
        
        # Get all camera pairs with sufficient overlap
        camera_pairs = self._get_camera_pairs_with_overlap(sparse_points, cameras, images)
        
        if len(camera_pairs) < 2:
            logger.warning("Insufficient camera pairs for scale consistency validation")
            return {
                'scale_variance': float('inf'),
                'scale_range': float('inf'), 
                'consistency_score': 0.0,
                'num_pairs': 0,
                'baseline_consistency': 0.0,
                'triangulation_consistency': 0.0
            }
        
        # Method 1: Baseline consistency check
        baseline_ratios = []
        for img1, img2 in camera_pairs[:min(50, len(camera_pairs))]:  # Limit for performance
            pose1 = self._get_camera_pose(img1, cameras, images)
            pose2 = self._get_camera_pose(img2, cameras, images)
            
            if pose1 is not None and pose2 is not None:
                R1, t1 = pose1
                R2, t2 = pose2
                baseline = np.linalg.norm(t2 - t1)
                
                # Expected baseline for this type of scene
                # Use median distance to 3D points as reference
                point_distances = []
                for point_data in sparse_points.values():
                    if 'xyz' in point_data:
                        dist1 = np.linalg.norm(np.array(point_data['xyz']) - t1)
                        point_distances.append(dist1)
                
                if len(point_distances) > 10:
                    median_scene_scale = np.median(point_distances)
                    if median_scene_scale > 0:
                        baseline_ratios.append(baseline / median_scene_scale)
        
        # Method 2: Triangulation consistency
        triangulation_scales = []
        sample_pairs = camera_pairs[:min(20, len(camera_pairs))]  # Sample for performance
        
        for img1, img2 in sample_pairs:
            common_points = self._get_common_points(img1, img2, sparse_points, images)
            if len(common_points) >= 10:
                # Get direct scale estimate from this pair
                pair_scale = self._estimate_pair_scale(img1, img2, common_points, cameras, images)
                if pair_scale > 0:
                    triangulation_scales.append(pair_scale)
        
        # Method 3: Point depth consistency  
        depth_ratios = []
        for img_path, img_data in list(images.items())[:min(20, len(images))]:
            pose = self._get_camera_pose(img_path, cameras, images)
            if pose is not None:
                R, t = pose
                depths = []
                
                for point_id in img_data['point3D_ids']:
                    if point_id != -1 and point_id in sparse_points:
                        point_xyz = np.array(sparse_points[point_id]['xyz'])
                        # Transform point to camera coordinate system
                        point_cam = R @ point_xyz + t
                        if point_cam[2] > 0.1:  # Valid depth
                            depths.append(point_cam[2])
                
                if len(depths) >= 5:
                    depth_variance = np.var(depths) / (np.mean(depths) ** 2) if np.mean(depths) > 0 else float('inf')
                    depth_ratios.append(depth_variance)
        
        # Calculate consistency metrics
        metrics = {}
        
        # Baseline consistency
        if len(baseline_ratios) > 1:
            baseline_mean = np.mean(baseline_ratios)
            baseline_std = np.std(baseline_ratios)
            metrics['baseline_consistency'] = 1.0 - min(baseline_std / baseline_mean, 1.0) if baseline_mean > 0 else 0.0
            metrics['baseline_variance'] = baseline_std ** 2
        else:
            metrics['baseline_consistency'] = 0.0
            metrics['baseline_variance'] = float('inf')
        
        # Triangulation scale consistency
        if len(triangulation_scales) > 1:
            tri_mean = np.mean(triangulation_scales)
            tri_std = np.std(triangulation_scales)
            metrics['triangulation_consistency'] = 1.0 - min(tri_std / tri_mean, 1.0) if tri_mean > 0 else 0.0
            metrics['scale_variance'] = tri_std ** 2
            metrics['scale_range'] = np.max(triangulation_scales) - np.min(triangulation_scales)
        else:
            metrics['triangulation_consistency'] = 0.0
            metrics['scale_variance'] = float('inf')
            metrics['scale_range'] = float('inf')
        
        # Depth consistency
        if len(depth_ratios) > 0:
            metrics['depth_consistency'] = 1.0 - min(np.mean(depth_ratios), 1.0)
        else:
            metrics['depth_consistency'] = 0.0
        
        # Overall consistency score (weighted combination)
        consistency_components = []
        if metrics.get('baseline_consistency', 0) > 0:
            consistency_components.append(metrics['baseline_consistency'] * 0.3)
        if metrics.get('triangulation_consistency', 0) > 0:
            consistency_components.append(metrics['triangulation_consistency'] * 0.5)
        if metrics.get('depth_consistency', 0) > 0:
            consistency_components.append(metrics['depth_consistency'] * 0.2)
        
        metrics['consistency_score'] = np.mean(consistency_components) if consistency_components else 0.0
        metrics['num_pairs'] = len(camera_pairs)
        metrics['num_triangulation_samples'] = len(triangulation_scales)
        metrics['num_baseline_samples'] = len(baseline_ratios)
        metrics['num_depth_samples'] = len(depth_ratios)
        
        # Log results
        logger.info(f"Scale consistency validation completed:")
        logger.info(f"  Overall consistency score: {metrics['consistency_score']:.3f}")
        logger.info(f"  Scale variance: {metrics['scale_variance']:.6f}")
        logger.info(f"  Scale range: {metrics['scale_range']:.6f}")
        logger.info(f"  Baseline consistency: {metrics['baseline_consistency']:.3f}")
        logger.info(f"  Triangulation consistency: {metrics['triangulation_consistency']:.3f}")
        logger.info(f"  Depth consistency: {metrics['depth_consistency']:.3f}")
        logger.info(f"  Analyzed {metrics['num_pairs']} camera pairs")
        
        return metrics 