"""
Geometric verification module with support for multiple RANSAC implementations
Supports OpenCV MAGSAC, pyransac, and GPU Advanced MAGSAC
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any, Literal
from enum import Enum
import time

# GPU modules - completely optional to avoid import issues
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

try:
    import pyransac
    PYRANSAC_AVAILABLE = True
except ImportError:
    PYRANSAC_AVAILABLE = False

logger = logging.getLogger(__name__)


class RANSACMethod(Enum):
    """Available RANSAC methods for geometric verification"""
    OPENCV_MAGSAC = "opencv_magsac"
    PYRANSAC = "pyransac"


class GeometricVerification:
    """
    Geometric verification with multiple RANSAC implementation options
    
    Supports:
    - OpenCV MAGSAC (default, most reliable)
    - pyransac (if available)
    - GPU Advanced MAGSAC (if GPU available)
    """
    
    def __init__(self, 
                 config_or_method = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        """
        Initialize geometric verification
        
        Args:
            config_or_method: Either a config dict or RANSACMethod enum
            device: GPU device for GPU-based methods
            **kwargs: Additional parameters for the chosen method
        """
        # Handle config dictionary or method enum
        if isinstance(config_or_method, dict):
            config = config_or_method
            method_str = config.get('geometric_method', 'opencv_magsac')
            # Safely get the enum value
            try:
                self.method = RANSACMethod(method_str)
            except ValueError:
                # If method_str is not a valid enum value, use default
                self.method = RANSACMethod.OPENCV_MAGSAC
            self.confidence = config.get('confidence', 0.95)  # More relaxed
            self.max_iterations = config.get('max_iterations', 10000)
            self.threshold = config.get('threshold', 3.0)  # More relaxed
            self.min_matches = config.get('min_matches', 8)
        elif config_or_method is None or isinstance(config_or_method, RANSACMethod):
            self.method = config_or_method or RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.95)  # More relaxed
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 3.0)  # More relaxed
            self.min_matches = kwargs.get('min_matches', 8)
        else:
            # Fallback to default
            self.method = RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.95)  # More relaxed
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 3.0)  # More relaxed
            self.min_matches = kwargs.get('min_matches', 8)
        
        self.device = device or (torch.device('cuda') if GPU_AVAILABLE else torch.device('cpu'))
        
        # Initialize the chosen method
        self._init_method(**kwargs)
        
        logger.info(f"Geometric verification initialized with method: {self.method.value}")
    
    def _init_method(self, **kwargs):
        """Initialize the chosen RANSAC method"""
        if self.method == RANSACMethod.PYRANSAC:
            if not PYRANSAC_AVAILABLE:
                logger.warning("pyransac not available, falling back to OpenCV MAGSAC")
                self.method = RANSACMethod.OPENCV_MAGSAC
        
        # OpenCV MAGSAC is always available as fallback
        if self.method == RANSACMethod.OPENCV_MAGSAC:
            logger.info("Using OpenCV MAGSAC")
    
    def find_essential_matrix(self, 
                            points1: np.ndarray, 
                            points2: np.ndarray,
                            intrinsics: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find essential matrix using the chosen RANSAC method with robust preprocessing
        
        Args:
            points1: Points in first image (Nx2)
            points2: Points in second image (Nx2)
            intrinsics: Camera intrinsics (optional)
            
        Returns:
            Essential matrix and inlier mask
        """
        start_time = time.time()
        
        if len(points1) < 8:
            logger.warning(f"Not enough points for essential matrix estimation: {len(points1)}")
            return None, None
        
        # Ensure points are float32
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        
        # Prefilter points
        points1, points2 = self._prefilter_points(points1, points2)
        
        if len(points1) < 8:
            logger.warning(f"Not enough points after preprocessing: {len(points1)}")
            return None, None
        
        try:
            if self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
                E, inliers = self._find_essential_pyransac(points1, points2, intrinsics)
            
            else:  # OpenCV MAGSAC (default)
                E, inliers = self._find_essential_opencv(points1, points2, intrinsics)
            
            elapsed_time = time.time() - start_time
            num_inliers = np.sum(inliers) if inliers is not None else 0
            logger.debug(f"Essential matrix found in {elapsed_time:.4f}s, "
                        f"inliers: {num_inliers}/{len(points1)}")
            
            return E, inliers
            
        except Exception as e:
            logger.error(f"Essential matrix estimation failed: {e}")
            return None, None
    
    def find_fundamental_matrix(self, 
                              points1: np.ndarray, 
                              points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find fundamental matrix using the chosen RANSAC method
        
        Args:
            points1: Points in first image (Nx2)
            points2: Points in second image (Nx2)
            
        Returns:
            Fundamental matrix and inlier mask
        """
        start_time = time.time()
        
        if len(points1) < 8:
            logger.warning(f"Not enough points for fundamental matrix estimation: {len(points1)}")
            return None, None
        
        # Ensure points are float32
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        
        try:
            if self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
                F, inliers = self._find_fundamental_pyransac(points1, points2)
            
            else:  # OpenCV MAGSAC (default)
                F, inliers = self._find_fundamental_opencv(points1, points2)
            
            elapsed_time = time.time() - start_time
            num_inliers = np.sum(inliers) if inliers is not None else 0
            logger.debug(f"Fundamental matrix found in {elapsed_time:.4f}s, "
                        f"inliers: {num_inliers}/{len(points1)}")
            
            return F, inliers
            
        except Exception as e:
            logger.error(f"Fundamental matrix estimation failed: {e}")
            return None, None
    
    def _find_essential_opencv(self, 
                             points1: np.ndarray, 
                             points2: np.ndarray,
                             intrinsics: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find essential matrix using OpenCV MAGSAC with adaptive parameters"""
        
        # Use identity camera matrix if not provided
        camera_matrix = intrinsics if intrinsics is not None else np.eye(3)
        
        # Use adaptive parameters
        threshold, confidence, max_iters = self._adapt_parameters(points1, points2)
        
        try:
            E, mask = cv2.findEssentialMat(
                points1, points2,
                cameraMatrix=camera_matrix,
                method=cv2.USAC_MAGSAC,
                prob=confidence,
                threshold=threshold,
                maxIters=max_iters
            )
            
            if E is None or E.size == 0:
                logger.debug("Essential matrix estimation returned empty, trying RANSAC fallback")
                E, mask = cv2.findEssentialMat(
                    points1, points2,
                    cameraMatrix=camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.90,
                    threshold=threshold * 1.5,
                    maxIters=max_iters
                )
            
            inliers = mask.flatten().astype(bool) if mask is not None else None
            return E, inliers
            
        except Exception as e:
            logger.warning(f"Essential matrix estimation failed: {e}")
            return None, None
    
    def _find_fundamental_opencv(self, 
                               points1: np.ndarray, 
                               points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find fundamental matrix using OpenCV MAGSAC with robust preprocessing"""
        
        # Validate input points
        if len(points1) < 8 or len(points2) < 8:
            logger.warning(f"Not enough points for fundamental matrix: {len(points1)}")
            return None, None
            
        # Ensure points are in the right format (N, 2)
        points1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
        points2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
        
        # Prefilter obvious outliers and invalid points
        points1, points2 = self._prefilter_points(points1, points2)
        
        # Check for degenerate cases after preprocessing
        if len(points1) < 8 or len(points2) < 8:
            logger.warning(f"Not enough points after preprocessing: {len(points1)}")
            return None, None
            
        unique_points1 = len(np.unique(points1, axis=0))
        unique_points2 = len(np.unique(points2, axis=0))
        if unique_points1 < 8 or unique_points2 < 8:
            logger.warning(f"Not enough unique points: {unique_points1}/{len(points1)}, {unique_points2}/{len(points2)}")
            return None, None
            
            
        # Use adaptive parameters based on data quality
        threshold, confidence, max_iters = self._adapt_parameters(points1, points2)
        
        try:
            F, mask = cv2.findFundamentalMat(
                points1, points2,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=threshold,
                confidence=confidence,
                maxIters=max_iters
            )
            
            # Check if F is valid
            if F is None or F.size == 0:
                logger.debug("OpenCV fundamental matrix estimation returned empty result, trying fallback")
                # Try with more relaxed parameters
                return self._find_fundamental_fallback(points1, points2)
                
            inliers = mask.flatten().astype(bool) if mask is not None else None
            
            # Validate results
            if inliers is not None and np.sum(inliers) < 8:
                logger.debug(f"Too few inliers ({np.sum(inliers)}), trying fallback")
                return self._find_fundamental_fallback(points1, points2)
                
            return F, inliers
            
        except cv2.error as e:
            logger.debug(f"OpenCV fundamental matrix failed: {e}, trying fallback")
            return self._find_fundamental_fallback(points1, points2)
    
    def _find_essential_pyransac(self, 
                               points1: np.ndarray, 
                               points2: np.ndarray,
                               intrinsics: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find essential matrix using pyransac"""
        
        # Convert to pyransac format if needed
        # Note: This is a placeholder - actual implementation depends on pyransac API
        try:
            # Placeholder implementation - adjust based on actual pyransac API
            model, inliers = pyransac.findEssentialMatrix(
                points1, points2,
                confidence=self.confidence,
                max_iterations=self.max_iterations,
                threshold=self.threshold
            )
            return model, inliers
        except Exception as e:
            logger.warning(f"pyransac essential matrix failed: {e}, falling back to OpenCV")
            return self._find_essential_opencv(points1, points2, intrinsics)
    
    def _find_fundamental_pyransac(self, 
                                 points1: np.ndarray, 
                                 points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find fundamental matrix using pyransac"""
        
        try:
            # Placeholder implementation - adjust based on actual pyransac API
            model, inliers = pyransac.findFundamentalMatrix(
                points1, points2,
                confidence=self.confidence,
                max_iterations=self.max_iterations,
                threshold=self.threshold
            )
            return model, inliers
        except Exception as e:
            logger.warning(f"pyransac fundamental matrix failed: {e}, falling back to OpenCV")
            return self._find_fundamental_opencv(points1, points2)
    
    def _find_essential_gpu_advanced(self, 
                                   points1: np.ndarray, 
                                   points2: np.ndarray,
                                   intrinsics: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find essential matrix using GPU Advanced MAGSAC"""
        
        try:
            E, inliers = self.gpu_magsac.find_essential_matrix(points1, points2, intrinsics)
            return E, inliers
        except Exception as e:
            logger.warning(f"GPU Advanced MAGSAC failed: {e}, falling back to OpenCV")
            return self._find_essential_opencv(points1, points2, intrinsics)
    
    def _find_fundamental_gpu_advanced(self, 
                                     points1: np.ndarray, 
                                     points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find fundamental matrix using GPU Advanced MAGSAC"""
        
        try:
            F, inliers = self.gpu_magsac.find_fundamental_matrix(points1, points2)
            return F, inliers
        except Exception as e:
            logger.warning(f"GPU Advanced MAGSAC failed: {e}, falling back to OpenCV")
            return self._find_fundamental_opencv(points1, points2)
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about the current method"""
        info = {
            'method': self.method.value,
            'confidence': self.confidence,
            'max_iterations': self.max_iterations,
            'threshold': self.threshold,
            'gpu_available': GPU_AVAILABLE,
            'pyransac_available': PYRANSAC_AVAILABLE
        }
        
        
        return info
    
    @staticmethod
    def get_available_methods() -> list:
        """Get list of available RANSAC methods"""
        methods = [RANSACMethod.OPENCV_MAGSAC]  # Always available
        
        if PYRANSAC_AVAILABLE:
            methods.append(RANSACMethod.PYRANSAC)
        
        
        return methods
    
    def verify_matches(self, feat1: Dict[str, Any], feat2: Dict[str, Any], match_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Verify matches between two images using geometric verification
        
        Args:
            feat1: Features from first image
            feat2: Features from second image
            match_data: Match data containing matches0, matches1, etc.
            
        Returns:
            Verified match data with inliers only, or None if verification fails
        """
        try:
            # Get matched keypoints
            kpts0 = feat1['keypoints']
            kpts1 = feat2['keypoints']
            matches0 = match_data['matches0']
            matches1 = match_data['matches1']
            
            # Extract corresponding points
            if len(matches0) < self.min_matches:
                logger.debug(f"Not enough matches: {len(matches0)} < {self.min_matches}")
                return None
                
            points0 = kpts0[matches0]
            points1 = kpts1[matches1]
            
            # Find fundamental matrix and inliers
            F, inliers = self.find_fundamental_matrix(points0, points1)
            
            if F is not None and inliers is not None and np.sum(inliers) >= self.min_matches:
                # Filter matches to keep only inliers
                inlier_matches0 = matches0[inliers]
                inlier_matches1 = matches1[inliers]
                
                # Update scores if they exist
                inlier_scores0 = match_data.get('mscores0', np.ones(len(matches0)))[inliers]
                inlier_scores1 = match_data.get('mscores1', np.ones(len(matches1)))[inliers]
                
                # Create verified match data
                verified_match = {
                    'keypoints0': kpts0,
                    'keypoints1': kpts1,
                    'matches0': inlier_matches0,
                    'matches1': inlier_matches1,
                    'mscores0': inlier_scores0,
                    'mscores1': inlier_scores1,
                    'fundamental_matrix': F,
                    'inliers': inliers,
                    'image_shape0': match_data.get('image_shape0'),
                    'image_shape1': match_data.get('image_shape1')
                }
                
                logger.debug(f"Verified matches: {len(inlier_matches0)}/{len(matches0)} inliers")
                return verified_match
            else:
                logger.debug("Geometric verification failed")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to verify matches: {e}")
            return None

    def verify(self, matches: Dict[Tuple[str, str], Any]) -> Dict[Tuple[str, str], Any]:
        """
        Verify matches geometrically using RANSAC-based methods
        
        Args:
            matches: Dictionary of matches from feature matching
            
        Returns:
            Dictionary of verified matches with inliers only
        """
        verified_matches = {}
        
        logger.info(f"Verifying {len(matches)} image pairs...")
        
        for pair, match_data in matches.items():
            try:
                # Get matched keypoints
                kpts0 = match_data['keypoints0']
                kpts1 = match_data['keypoints1']
                matches0 = match_data['matches0']
                matches1 = match_data['matches1']
                
                # Extract corresponding points
                if len(matches0) < self.min_matches:
                    logger.debug(f"Skipping pair {pair}: not enough matches ({len(matches0)})")
                    continue
                    
                points0 = kpts0[matches0]
                points1 = kpts1[matches1]
                
                # Find fundamental matrix and inliers
                F, inliers = self.find_fundamental_matrix(points0, points1)
                
                if F is not None and inliers is not None and np.sum(inliers) >= self.min_matches:
                    # Filter matches to keep only inliers
                    inlier_matches0 = matches0[inliers]
                    inlier_matches1 = matches1[inliers]
                    
                    # Update scores if they exist
                    inlier_scores0 = match_data.get('mscores0', np.ones(len(matches0)))[inliers]
                    inlier_scores1 = match_data.get('mscores1', np.ones(len(matches1)))[inliers]
                    
                    # Create verified match data
                    verified_match = {
                        'keypoints0': kpts0,
                        'keypoints1': kpts1,
                        'matches0': inlier_matches0,
                        'matches1': inlier_matches1,
                        'mscores0': inlier_scores0,
                        'mscores1': inlier_scores1,
                        'fundamental_matrix': F,
                        'inliers': inliers,
                        'image_shape0': match_data.get('image_shape0'),
                        'image_shape1': match_data.get('image_shape1')
                    }
                    
                    verified_matches[pair] = verified_match
                    
                    logger.debug(f"Pair {pair}: {len(inlier_matches0)}/{len(matches0)} inliers")
                else:
                    logger.debug(f"Pair {pair}: geometric verification failed")
                    
            except Exception as e:
                logger.warning(f"Failed to verify pair {pair}: {e}")
                continue
        
        logger.info(f"Verified {len(verified_matches)}/{len(matches)} pairs successfully")
        return verified_matches

    def _prefilter_points(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prefilter points to remove obvious outliers and invalid points"""
        # Remove NaN and infinite values
        valid_mask = (np.isfinite(points1).all(axis=1) & 
                     np.isfinite(points2).all(axis=1))
        
        points1_clean = points1[valid_mask]
        points2_clean = points2[valid_mask]
        
        if len(points1_clean) < 8:
            return points1_clean, points2_clean
        
        # Remove points that are too close (likely duplicates with noise)
        min_distance = 2.0  # pixels
        keep_mask = np.ones(len(points1_clean), dtype=bool)
        
        for i in range(len(points1_clean) - 1):
            if not keep_mask[i]:
                continue
            distances = np.linalg.norm(points1_clean[i+1:] - points1_clean[i], axis=1)
            too_close = distances < min_distance
            keep_mask[i+1:] = keep_mask[i+1:] & ~too_close
        
        points1_filtered = points1_clean[keep_mask]
        points2_filtered = points2_clean[keep_mask]
        
        logger.debug(f"Prefiltering: {len(points1)} -> {len(points1_filtered)} points")
        return points1_filtered, points2_filtered
    
    def _adapt_parameters(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[float, float, int]:
        """Adapt RANSAC parameters based on data characteristics"""
        num_points = len(points1)
        
        # Check point spread to detect low-quality matches
        spread1 = np.std(points1, axis=0).mean()
        spread2 = np.std(points2, axis=0).mean()
        avg_spread = (spread1 + spread2) / 2
        
        # Adaptive threshold based on point distribution
        if avg_spread < 50:  # Points are clustered
            threshold = self.threshold * 2
            confidence = 0.90
        elif num_points < 20:  # Very few points
            threshold = self.threshold * 1.5
            confidence = 0.90
        elif num_points < 50:  # Few points
            threshold = self.threshold * 1.2
            confidence = 0.93
        else:  # Normal case
            threshold = self.threshold
            confidence = self.confidence
        
        # Adaptive iterations
        max_iters = min(self.max_iterations, max(1000, num_points * 100))
        
        logger.debug(f"Adaptive params: threshold={threshold:.1f}, confidence={confidence:.2f}, iters={max_iters}")
        return threshold, confidence, max_iters
    
    def _find_fundamental_fallback(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fallback fundamental matrix estimation with multiple methods"""
        logger.debug("Trying fallback methods for fundamental matrix")
        
        # Method 1: LMEDS (good for high outlier ratio)
        try:
            F, mask = cv2.findFundamentalMat(
                points1, points2,
                method=cv2.FM_LMEDS,
                param1=3.0,  # LMEDS parameter
                param2=0.99
            )
            
            if F is not None and F.size > 0:
                inliers = mask.flatten().astype(bool) if mask is not None else None
                if inliers is not None and np.sum(inliers) >= 8:
                    logger.debug(f"LMEDS fallback successful: {np.sum(inliers)} inliers")
                    return F, inliers
        except:
            pass
        
        # Method 2: RANSAC with very relaxed parameters
        try:
            F, mask = cv2.findFundamentalMat(
                points1, points2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.85,
                maxIters=5000
            )
            
            if F is not None and F.size > 0:
                inliers = mask.flatten().astype(bool) if mask is not None else None
                if inliers is not None and np.sum(inliers) >= 8:
                    logger.debug(f"RANSAC fallback successful: {np.sum(inliers)} inliers")
                    return F, inliers
        except:
            pass
        
        # Method 3: 8-point algorithm if we have exactly the right conditions
        if len(points1) >= 8:
            try:
                F, _ = cv2.findFundamentalMat(
                    points1, points2,
                    method=cv2.FM_8POINT
                )
                
                if F is not None and F.size > 0:
                    # For 8-point, all points are "inliers"
                    inliers = np.ones(len(points1), dtype=bool)
                    logger.debug("8-point fallback successful")
                    return F, inliers
            except:
                pass
        
        logger.debug("All fallback methods failed")
        return None, None
    
    @staticmethod
    def recommend_method(num_points: int, has_gpu: bool = None) -> RANSACMethod:
        """
        Recommend best method based on available hardware and data size
        
        Args:
            num_points: Number of point correspondences
            has_gpu: Whether GPU is available (auto-detect if None)
            
        Returns:
            Recommended RANSAC method
        """
        if has_gpu is None:
            has_gpu = GPU_AVAILABLE
        
        # For small datasets, OpenCV is usually fastest
        if num_points < 1000:
            return RANSACMethod.OPENCV_MAGSAC
        
        # For medium datasets, OpenCV MAGSAC is still reliable
        return RANSACMethod.OPENCV_MAGSAC


# Convenience functions
def find_essential_matrix(points1: np.ndarray, 
                         points2: np.ndarray,
                         method: RANSACMethod = RANSACMethod.OPENCV_MAGSAC,
                         intrinsics: Optional[np.ndarray] = None,
                         **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convenience function to find essential matrix
    
    Args:
        points1: Points in first image (Nx2)
        points2: Points in second image (Nx2)
        method: RANSAC method to use
        intrinsics: Camera intrinsics (optional)
        **kwargs: Additional parameters
        
    Returns:
        Essential matrix and inlier mask
    """
    verifier = GeometricVerification(method=method, **kwargs)
    return verifier.find_essential_matrix(points1, points2, intrinsics)


def find_fundamental_matrix(points1: np.ndarray,
                          points2: np.ndarray,
                          method: RANSACMethod = RANSACMethod.OPENCV_MAGSAC,
                          **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convenience function to find fundamental matrix
    
    Args:
        points1: Points in first image (Nx2)
        points2: Points in second image (Nx2)
        method: RANSAC method to use
        **kwargs: Additional parameters
        
    Returns:
        Fundamental matrix and inlier mask
    """
    verifier = GeometricVerification(method=method, **kwargs)
    return verifier.find_fundamental_matrix(points1, points2)