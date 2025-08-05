"""
Incremental SfM reconstruction using OpenCV USAC_MAGSAC via geometric_verification module
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import scipy.spatial as spatial
from pathlib import Path
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import logging
from .geometric_verification import GeometricVerification, RANSACMethod

logger = logging.getLogger(__name__)


class IncrementalSfM:
    """Incremental Structure-from-Motion reconstruction using OpenCV USAC_MAGSAC"""
    
    def __init__(self, device: torch.device, max_image_size: int = 1600):
        self.device = device
        self.max_image_size = max_image_size
        
        # Initialize geometric verification with OpenCV MAGSAC (default)
        self.geometric_verifier = GeometricVerification(
            method=RANSACMethod.OPENCV_MAGSAC,
            confidence=0.999,
            max_iterations=10000,
            threshold=1.0
        )
        
        # Reconstruction state
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        self.point_tracks = {}
        self.next_point_id = 0
        
        logger.info(f"IncrementalSfM initialized with {self.geometric_verifier.method.value}")
        
    def reconstruct(self, features: Dict[str, Any], matches: Dict[Tuple[str, str], Any], 
                   image_paths: List[str]) -> Tuple[Dict, Dict, Dict]:
        """Run incremental SfM reconstruction"""
        
        print("Starting incremental SfM reconstruction...")
        
        # Initialize with first two images
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images for reconstruction")
        
        # Find best initial pair
        initial_pair = self._find_best_initial_pair(matches)
        if initial_pair is None:
            raise ValueError("No suitable initial pair found")
        
        img1_path, img2_path = initial_pair
        print(f"Initial pair: {img1_path} and {img2_path}")
        
        # Initialize reconstruction with first pair
        self._initialize_reconstruction(features, matches, img1_path, img2_path)
        
        # Incrementally add remaining images
        remaining_images = [p for p in image_paths if p not in [img1_path, img2_path]]
        
        for img_path in tqdm(remaining_images, desc="Adding images"):
            try:
                self._add_image_to_reconstruction(features, matches, img_path)
            except Exception as e:
                print(f"Warning: Failed to add {img_path}: {e}")
                continue
        
        # Bundle adjustment
        print("Running bundle adjustment...")
        self._bundle_adjustment()
        
        # Clean up reconstruction
        self._cleanup_reconstruction()
        
        return self.points3d, self.cameras, self.images
    
    def _find_best_initial_pair(self, matches: Dict[Tuple[str, str], Any]) -> Optional[Tuple[str, str]]:
        """Find the best initial pair for reconstruction"""
        best_pair = None
        best_score = 0
        
        for (img1, img2), match_data in matches.items():
            if len(match_data['matches0']) > best_score:
                best_score = len(match_data['matches0'])
                best_pair = (img1, img2)
        
        return best_pair
    
    def _initialize_reconstruction(self, features: Dict[str, Any], 
                                matches: Dict[Tuple[str, str], Any],
                                img1_path: str, img2_path: str):
        """Initialize reconstruction with first two images"""
        
        # Get matched keypoints
        match_data = matches[(img1_path, img2_path)]
        kpts1 = match_data['keypoints0']
        kpts2 = match_data['keypoints1']
        matches1 = match_data['matches0']
        matches2 = match_data['matches1']
        
        # Get matched points
        matched_kpts1 = kpts1[matches1]
        matched_kpts2 = kpts2[matches2]
        
        # Estimate essential matrix using geometric verification (OpenCV USAC_MAGSAC)
        E, mask = self.geometric_verifier.find_essential_matrix(
            matched_kpts1, matched_kpts2
        )
        
        # Convert boolean mask to uint8 for OpenCV compatibility
        if mask is not None:
            mask = mask.astype(np.uint8)
        
        if E is None:
            raise ValueError("Failed to estimate essential matrix")
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, matched_kpts1, matched_kpts2, mask=mask)
        
        # Initialize cameras
        camera_id = 0
        self.cameras[camera_id] = {
            'model': 'SIMPLE_PINHOLE',
            'width': features[img1_path]['image_shape'][1],
            'height': features[img1_path]['image_shape'][0],
            'params': [1000.0, features[img1_path]['image_shape'][1] / 2, 
                      features[img1_path]['image_shape'][0] / 2]
        }
        
        # Initialize images
        self.images[img1_path] = {
            'camera_id': camera_id,
            'qvec': [1.0, 0.0, 0.0, 0.0],  # Identity rotation
            'tvec': [0.0, 0.0, 0.0],  # Origin
            'name': Path(img1_path).name,
            'xys': features[img1_path]['keypoints'],
            'point3D_ids': [-1] * len(features[img1_path]['keypoints'])
        }
        
        # Convert rotation matrix to quaternion
        qvec = self._rotation_matrix_to_quaternion(R)
        
        self.images[img2_path] = {
            'camera_id': camera_id,
            'qvec': qvec,
            'tvec': t.flatten(),
            'name': Path(img2_path).name,
            'xys': features[img2_path]['keypoints'],
            'point3D_ids': [-1] * len(features[img2_path]['keypoints'])
        }
        
        # Triangulate initial points
        self._triangulate_initial_points(matched_kpts1, matched_kpts2, R, t, mask)
    
    def _add_image_to_reconstruction(self, features: Dict[str, Any],
                                   matches: Dict[Tuple[str, str], Any],
                                   img_path: str):
        """Add a new image to the reconstruction"""
        
        # Find best registered image to match with
        best_registered_img = self._find_best_registered_image(matches, img_path)
        if best_registered_img is None:
            raise ValueError(f"No suitable registered image found for {img_path}")
        
        # Get matches with best registered image
        pair_key = (img_path, best_registered_img) if (img_path, best_registered_img) in matches else (best_registered_img, img_path)
        match_data = matches[pair_key]
        
        # Get matched keypoints
        if pair_key[0] == img_path:
            kpts_new = match_data['keypoints0']
            kpts_registered = match_data['keypoints1']
            matches_new = match_data['matches0']
            matches_registered = match_data['matches1']
        else:
            kpts_new = match_data['keypoints1']
            kpts_registered = match_data['keypoints0']
            matches_new = match_data['matches1']
            matches_registered = match_data['matches0']
        
        # Get 2D-3D correspondences
        points2d, points3d = self._get_2d3d_correspondences(
            kpts_new, kpts_registered, matches_new, matches_registered
        )
        
        if len(points2d) < 6:
            raise ValueError("Not enough 2D-3D correspondences")
        
        # Estimate camera pose using PnP with MAGSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3d, points2d,
            np.array(self.cameras[0]['params'][:3]).reshape(3, 3),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise ValueError("Failed to estimate camera pose")
        
        # Convert rotation vector to quaternion
        R, _ = cv2.Rodrigues(rvec)
        qvec = self._rotation_matrix_to_quaternion(R)
        
        # Add image to reconstruction
        self.images[img_path] = {
            'camera_id': 0,  # Use same camera for all images
            'qvec': qvec,
            'tvec': tvec.flatten(),
            'name': Path(img_path).name,
            'xys': features[img_path]['keypoints'],
            'point3D_ids': [-1] * len(features[img_path]['keypoints'])
        }
        
        # Triangulate new points
        self._triangulate_new_points(features, matches, img_path)
    
    def _find_best_registered_image(self, matches: Dict[Tuple[str, str], Any], 
                                  img_path: str) -> Optional[str]:
        """Find the best registered image to match with new image"""
        best_img = None
        best_score = 0
        
        for (img1, img2), match_data in matches.items():
            if img1 == img_path and img2 in self.images:
                if len(match_data['matches0']) > best_score:
                    best_score = len(match_data['matches0'])
                    best_img = img2
            elif img2 == img_path and img1 in self.images:
                if len(match_data['matches1']) > best_score:
                    best_score = len(match_data['matches1'])
                    best_img = img1
        
        return best_img
    
    def _get_2d3d_correspondences(self, kpts_new: np.ndarray, kpts_registered: np.ndarray,
                                 matches_new: np.ndarray, matches_registered: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D-3D correspondences for PnP"""
        points2d = []
        points3d = []
        
        for i, (match_new, match_registered) in enumerate(zip(matches_new, matches_registered)):
            # Check if the registered point has a 3D point
            registered_img = None
            for img_path in self.images:
                if img_path in self.images:
                    point3d_id = self.images[img_path]['point3D_ids'][match_registered]
                    if point3d_id != -1 and point3d_id in self.points3d:
                        points2d.append(kpts_new[match_new])
                        points3d.append(self.points3d[point3d_id]['xyz'])
                        break
        
        return np.array(points2d), np.array(points3d)
    
    def _triangulate_initial_points(self, kpts1: np.ndarray, kpts2: np.ndarray,
                                  R: np.ndarray, t: np.ndarray, mask: np.ndarray):
        """Triangulate initial 3D points"""
        # Camera matrices
        K = np.array(self.cameras[0]['params'][:3]).reshape(3, 3)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        # Add valid points to reconstruction
        for i, (point_3d, is_valid) in enumerate(zip(points_3d.T, mask.flatten())):
            if is_valid:
                self.points3d[self.next_point_id] = {
                    'xyz': point_3d,
                    'rgb': [255, 255, 255],  # Default white color
                    'error': 0.0,
                    'track': [(0, i), (1, i)]  # (image_id, keypoint_id)
                }
                self.next_point_id += 1
    
    def _triangulate_new_points(self, features: Dict[str, Any],
                              matches: Dict[Tuple[str, str], Any],
                              img_path: str):
        """Triangulate new 3D points from newly added image"""
        
        # Find all registered images that have matches with the new image
        registered_matches = []
        
        for (img1, img2), match_data in matches.items():
            if img1 == img_path and img2 in self.images:
                registered_matches.append((img2, match_data, False))  # False = img_path is img1
            elif img2 == img_path and img1 in self.images:
                registered_matches.append((img1, match_data, True))   # True = img_path is img2
        
        if not registered_matches:
            logger.warning(f"No registered matches found for {img_path}")
            return
        
        new_img_keypoints = features[img_path]['keypoints']
        new_img_data = self.images[img_path]
        
        # Get camera matrices
        K = np.array([
            [self.cameras[0]['params'][0], 0, self.cameras[0]['params'][1]],
            [0, self.cameras[0]['params'][0], self.cameras[0]['params'][2]],
            [0, 0, 1]
        ])
        
        # New image projection matrix
        R_new = self._quaternion_to_rotation_matrix(np.array(new_img_data['qvec']))
        t_new = np.array(new_img_data['tvec'])
        P_new = K @ np.hstack([R_new, t_new.reshape(3, 1)])
        
        triangulated_count = 0
        
        # Triangulate with each registered image
        for reg_img_path, match_data, is_flipped in registered_matches:
            reg_img_data = self.images[reg_img_path]
            reg_keypoints = features[reg_img_path]['keypoints']
            
            # Get matched keypoints
            if is_flipped:
                kpts_new = match_data['keypoints1']
                kpts_reg = match_data['keypoints0']
                matches_new = match_data['matches1']
                matches_reg = match_data['matches0']
            else:
                kpts_new = match_data['keypoints0']
                kpts_reg = match_data['keypoints1']
                matches_new = match_data['matches0']
                matches_reg = match_data['matches1']
            
            # Registered image projection matrix
            R_reg = self._quaternion_to_rotation_matrix(np.array(reg_img_data['qvec']))
            t_reg = np.array(reg_img_data['tvec'])
            P_reg = K @ np.hstack([R_reg, t_reg.reshape(3, 1)])
            
            # Triangulate points that don't already have 3D correspondences
            for i, (match_new, match_reg) in enumerate(zip(matches_new, matches_reg)):
                
                # Check if this point already has a 3D correspondence
                if (match_reg < len(reg_img_data['point3D_ids']) and 
                    reg_img_data['point3D_ids'][match_reg] != -1):
                    # This point already has a 3D correspondence, update the new image's association
                    point3d_id = reg_img_data['point3D_ids'][match_reg]
                    if match_new < len(new_img_data['point3D_ids']):
                        new_img_data['point3D_ids'][match_new] = point3d_id
                    continue
                
                # Triangulate new point
                try:
                    pt_new = kpts_new[match_new]
                    pt_reg = kpts_reg[match_reg]
                    
                    # Triangulate using DLT method
                    point_3d = self._triangulate_point_dlt(pt_new, pt_reg, P_new, P_reg)
                    
                    if point_3d is not None and self._is_valid_triangulation(point_3d, pt_new, pt_reg, P_new, P_reg):
                        # Add new 3D point
                        self.points3d[self.next_point_id] = {
                            'xyz': point_3d,
                            'rgb': [128, 128, 128],  # Default gray color
                            'error': 0.0,
                            'track': []
                        }
                        
                        # Update point associations
                        if match_new < len(new_img_data['point3D_ids']):
                            new_img_data['point3D_ids'][match_new] = self.next_point_id
                        if match_reg < len(reg_img_data['point3D_ids']):
                            reg_img_data['point3D_ids'][match_reg] = self.next_point_id
                        
                        triangulated_count += 1
                        self.next_point_id += 1
                        
                except Exception as e:
                    logger.debug(f"Triangulation failed for point pair: {e}")
                    continue
        
        logger.info(f"Triangulated {triangulated_count} new 3D points for {img_path}")
    
    def _triangulate_point_dlt(self, pt1: np.ndarray, pt2: np.ndarray, 
                             P1: np.ndarray, P2: np.ndarray) -> Optional[np.ndarray]:
        """Triangulate a single point using DLT (Direct Linear Transform)"""
        
        # Build the system matrix A
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])
        
        try:
            # Solve using SVD
            U, s, Vt = np.linalg.svd(A)
            
            # The solution is the last column of V (last row of Vt)
            X = Vt[-1]
            
            # Convert from homogeneous coordinates
            if abs(X[3]) < 1e-8:
                return None
            
            point_3d = X[:3] / X[3]
            return point_3d
            
        except np.linalg.LinAlgError:
            return None
    
    def _is_valid_triangulation(self, point_3d: np.ndarray, pt1: np.ndarray, pt2: np.ndarray,
                              P1: np.ndarray, P2: np.ndarray, 
                              reprojection_threshold: float = 4.0) -> bool:
        """Check if triangulated point is valid"""
        
        # Check if point is in front of both cameras
        # Transform to camera coordinates
        point_homo = np.append(point_3d, 1)
        
        # Camera 1 check
        point_cam1 = P1 @ point_homo
        if point_cam1[2] <= 0:  # Behind camera 1
            return False
        
        # Camera 2 check  
        point_cam2 = P2 @ point_homo
        if point_cam2[2] <= 0:  # Behind camera 2
            return False
        
        # Check reprojection error
        proj1 = point_cam1[:2] / point_cam1[2]
        proj2 = point_cam2[:2] / point_cam2[2]
        
        error1 = np.linalg.norm(proj1 - pt1)
        error2 = np.linalg.norm(proj2 - pt2)
        
        if error1 > reprojection_threshold or error2 > reprojection_threshold:
            return False
        
        # Check if point is reasonable distance from cameras
        # (not too far or too close)
        dist1 = np.linalg.norm(point_3d)
        if dist1 < 0.1 or dist1 > 1000:  # Reasonable bounds
            return False
        
        return True
    
    def _bundle_adjustment(self):
        """Run bundle adjustment to refine camera poses and 3D points"""
        if not self.points3d or not self.images:
            logger.warning("No points or cameras to optimize")
            return
        
        print("Running bundle adjustment optimization...")
        
        # Collect optimization parameters
        camera_params, point_params, observations, camera_indices, point_indices = self._prepare_ba_data()
        
        if len(observations) < 10:
            logger.warning("Not enough observations for bundle adjustment")
            return
        
        # Create initial parameter vector
        x0 = np.concatenate([camera_params.flatten(), point_params.flatten()])
        
        # Run optimization
        try:
            result = least_squares(
                self._bundle_adjustment_residuals,
                x0,
                args=(observations, camera_indices, point_indices, len(self.images), len(self.points3d)),
                jac_sparsity=self._bundle_adjustment_sparsity(len(self.images), len(self.points3d), len(observations)),
                verbose=2,
                max_nfev=100,
                ftol=1e-6,
                xtol=1e-6
            )
            
            if result.success:
                self._update_from_ba_result(result.x, len(self.images), len(self.points3d))
                print(f"Bundle adjustment converged: cost reduced from {result.cost:.6f}")
            else:
                logger.warning(f"Bundle adjustment failed to converge: {result.message}")
                
        except Exception as e:
            logger.error(f"Bundle adjustment failed: {e}")
    
    def _prepare_ba_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for bundle adjustment"""
        
        # Convert camera parameters to optimization format
        camera_params = []
        image_to_cam_idx = {}
        
        for cam_idx, (img_path, img_data) in enumerate(self.images.items()):
            # Camera parameters: [qw, qx, qy, qz, tx, ty, tz, f, cx, cy]
            qvec = img_data['qvec']
            tvec = img_data['tvec']
            camera_info = self.cameras[img_data['camera_id']]
            
            cam_params = list(qvec) + list(tvec) + camera_info['params'][:3]
            camera_params.append(cam_params)
            image_to_cam_idx[img_path] = cam_idx
        
        camera_params = np.array(camera_params)
        
        # Convert 3D points
        point_params = []
        point_to_idx = {}
        
        for pt_idx, (point_id, point_data) in enumerate(self.points3d.items()):
            point_params.append(point_data['xyz'])
            point_to_idx[point_id] = pt_idx
        
        point_params = np.array(point_params)
        
        # Collect observations
        observations = []
        camera_indices = []
        point_indices = []
        
        for img_path, img_data in self.images.items():
            cam_idx = image_to_cam_idx[img_path]
            keypoints = img_data['xys']
            point3d_ids = img_data['point3D_ids']
            
            for kp_idx, point3d_id in enumerate(point3d_ids):
                if point3d_id != -1 and point3d_id in point_to_idx:
                    pt_idx = point_to_idx[point3d_id]
                    observation = keypoints[kp_idx]
                    
                    observations.append(observation)
                    camera_indices.append(cam_idx)
                    point_indices.append(pt_idx)
        
        return (camera_params, point_params, 
                np.array(observations), np.array(camera_indices), np.array(point_indices))
    
    def _bundle_adjustment_residuals(self, params: np.ndarray, observations: np.ndarray,
                                   camera_indices: np.ndarray, point_indices: np.ndarray,
                                   n_cameras: int, n_points: int) -> np.ndarray:
        """Compute residuals for bundle adjustment"""
        
        # Split parameters
        camera_params = params[:n_cameras * 10].reshape(n_cameras, 10)
        point_params = params[n_cameras * 10:].reshape(n_points, 3)
        
        residuals = []
        
        for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
            # Get camera parameters
            qvec = camera_params[cam_idx, :4]
            tvec = camera_params[cam_idx, 4:7]
            focal = camera_params[cam_idx, 7]
            cx = camera_params[cam_idx, 8]
            cy = camera_params[cam_idx, 9]
            
            # Get 3D point
            point_3d = point_params[pt_idx]
            
            # Project 3D point to 2D
            projected = self._project_point(point_3d, qvec, tvec, focal, cx, cy)
            
            # Compute residual
            observed = observations[i]
            residual = projected - observed
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def _project_point(self, point_3d: np.ndarray, qvec: np.ndarray, tvec: np.ndarray,
                      focal: float, cx: float, cy: float) -> np.ndarray:
        """Project 3D point to 2D using camera parameters"""
        
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(qvec)
        
        # Transform point to camera coordinate system
        point_cam = R @ point_3d + tvec
        
        # Avoid division by zero
        if abs(point_cam[2]) < 1e-8:
            return np.array([cx, cy])
        
        # Project to image plane
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        
        # Apply intrinsic parameters
        u = focal * x + cx
        v = focal * y + cy
        
        return np.array([u, v])
    
    def _quaternion_to_rotation_matrix(self, qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = qvec
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R
    
    def _bundle_adjustment_sparsity(self, n_cameras: int, n_points: int, n_observations: int):
        """Define sparsity pattern for bundle adjustment Jacobian"""
        
        # Each observation contributes to:
        # - 10 camera parameters (quaternion + translation + intrinsics)
        # - 3 point parameters
        
        m = n_observations * 2  # 2 residuals per observation
        n = n_cameras * 10 + n_points * 3  # Total parameters
        
        # Create sparse matrix structure
        A = lil_matrix((m, n), dtype=int)
        
        for i in range(n_observations):
            # Residual indices for this observation
            res_idx = i * 2
            
            # This would be filled based on camera_indices and point_indices
            # For simplicity, we'll create a dense pattern (less efficient but safe)
            A[res_idx:res_idx+2, :] = 1
        
        return A
    
    def _update_from_ba_result(self, optimized_params: np.ndarray, n_cameras: int, n_points: int):
        """Update camera poses and 3D points from bundle adjustment result"""
        
        # Split optimized parameters
        camera_params = optimized_params[:n_cameras * 10].reshape(n_cameras, 10)
        point_params = optimized_params[n_cameras * 10:].reshape(n_points, 3)
        
        # Update camera poses
        for cam_idx, (img_path, img_data) in enumerate(self.images.items()):
            img_data['qvec'] = camera_params[cam_idx, :4].tolist()
            img_data['tvec'] = camera_params[cam_idx, 4:7].tolist()
            
            # Update camera intrinsics
            camera_id = img_data['camera_id']
            self.cameras[camera_id]['params'][:3] = camera_params[cam_idx, 7:].tolist()
        
        # Update 3D points
        for pt_idx, (point_id, point_data) in enumerate(self.points3d.items()):
            point_data['xyz'] = point_params[pt_idx]
    
    def _cleanup_reconstruction(self):
        """Clean up reconstruction by removing outliers"""
        logger.info("Cleaning up reconstruction...")
        
        initial_points = len(self.points3d)
        initial_images = len(self.images)
        
        # Remove points with high reprojection error
        self._remove_high_error_points()
        
        # Remove points with too few observations
        self._remove_poorly_observed_points()
        
        # Remove images with too few point observations
        self._remove_poorly_registered_images()
        
        # Final validation
        self._validate_reconstruction()
        
        final_points = len(self.points3d)
        final_images = len(self.images)
        
        logger.info(f"Cleanup complete: {initial_points} -> {final_points} points, "
                   f"{initial_images} -> {final_images} images")
    
    def _remove_high_error_points(self, max_reprojection_error: float = 4.0):
        """Remove 3D points with high reprojection error"""
        points_to_remove = []
        
        for point_id, point_data in self.points3d.items():
            total_error = 0.0
            observation_count = 0
            
            # Calculate reprojection error for this point across all images
            for img_path, img_data in self.images.items():
                point3d_ids = img_data['point3D_ids']
                keypoints = img_data['xys']
                
                for kp_idx, pid in enumerate(point3d_ids):
                    if pid == point_id:
                        # Calculate reprojection error
                        try:
                            observed = keypoints[kp_idx]
                            qvec = np.array(img_data['qvec'])
                            tvec = np.array(img_data['tvec'])
                            camera_params = self.cameras[img_data['camera_id']]['params']
                            
                            projected = self._project_point(
                                point_data['xyz'], qvec, tvec,
                                camera_params[0], camera_params[1], camera_params[2]
                            )
                            
                            error = np.linalg.norm(projected - observed)
                            total_error += error
                            observation_count += 1
                            
                        except Exception as e:
                            logger.debug(f"Error calculating reprojection for point {point_id}: {e}")
                            continue
            
            # Mark point for removal if average error is too high
            if observation_count > 0:
                avg_error = total_error / observation_count
                if avg_error > max_reprojection_error:
                    points_to_remove.append(point_id)
                    
                # Update point error
                point_data['error'] = avg_error
        
        # Remove high-error points
        for point_id in points_to_remove:
            del self.points3d[point_id]
            
            # Remove references from images
            for img_data in self.images.values():
                point3d_ids = img_data['point3D_ids']
                for i, pid in enumerate(point3d_ids):
                    if pid == point_id:
                        point3d_ids[i] = -1
        
        logger.info(f"Removed {len(points_to_remove)} high-error points")
    
    def _remove_poorly_observed_points(self, min_observations: int = 2):
        """Remove points observed by too few cameras"""
        points_to_remove = []
        
        for point_id, point_data in self.points3d.items():
            observation_count = 0
            
            # Count observations across all images
            for img_data in self.images.values():
                if point_id in img_data['point3D_ids']:
                    observation_count += 1
            
            if observation_count < min_observations:
                points_to_remove.append(point_id)
        
        # Remove poorly observed points
        for point_id in points_to_remove:
            del self.points3d[point_id]
            
            # Remove references from images
            for img_data in self.images.values():
                point3d_ids = img_data['point3D_ids']
                for i, pid in enumerate(point3d_ids):
                    if pid == point_id:
                        point3d_ids[i] = -1
        
        logger.info(f"Removed {len(points_to_remove)} poorly observed points")
    
    def _remove_poorly_registered_images(self, min_points: int = 10):
        """Remove images with too few 3D point observations"""
        images_to_remove = []
        
        for img_path, img_data in self.images.items():
            point_count = sum(1 for pid in img_data['point3D_ids'] 
                            if pid != -1 and pid in self.points3d)
            
            if point_count < min_points:
                images_to_remove.append(img_path)
        
        # Remove poorly registered images
        for img_path in images_to_remove:
            del self.images[img_path]
        
        logger.info(f"Removed {len(images_to_remove)} poorly registered images")
    
    def _validate_reconstruction(self):
        """Validate the final reconstruction"""
        if len(self.points3d) < 10:
            logger.warning("Very few 3D points in reconstruction - may be unreliable")
        
        if len(self.images) < 2:
            raise ValueError("Reconstruction needs at least 2 images")
        
        # Check for reasonable point distribution
        if self.points3d:
            points = np.array([p['xyz'] for p in self.points3d.values()])
            point_std = np.std(points, axis=0)
            
            if np.any(point_std < 0.01):
                logger.warning("Points are very close together - may indicate scale issues")
            
            if np.any(point_std > 1000):
                logger.warning("Points are very spread out - may indicate outliers")
        
        # Validate camera poses
        for img_path, img_data in self.images.items():
            qvec = np.array(img_data['qvec'])
            if abs(np.linalg.norm(qvec) - 1.0) > 0.1:
                logger.warning(f"Non-unit quaternion detected for {img_path}")
                
                # Normalize quaternion
                qvec = qvec / np.linalg.norm(qvec)
                img_data['qvec'] = qvec.tolist()
        
        logger.info("Reconstruction validation complete")
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        # Simplified conversion - in practice, use proper quaternion conversion
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
        
        return [w, x, y, z] 