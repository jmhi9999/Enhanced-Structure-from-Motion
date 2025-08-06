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
    try:
        # Only import if all GPU dependencies are available
        import cupy
        import faiss
        from .gpu_advanced_magsac import GPUAdvancedMAGSAC
    except (ImportError, AttributeError):
        GPU_AVAILABLE = False
        GPUAdvancedMAGSAC = None
except ImportError:
    GPU_AVAILABLE = False
    GPUAdvancedMAGSAC = None

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
    GPU_ADVANCED_MAGSAC = "gpu_advanced_magsac"


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
            self.confidence = config.get('confidence', 0.999)
            self.max_iterations = config.get('max_iterations', 10000)
            self.threshold = config.get('threshold', 1.0)
            self.min_matches = config.get('min_matches', 8)
        elif config_or_method is None or isinstance(config_or_method, RANSACMethod):
            self.method = config_or_method or RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.999)
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 1.0)
            self.min_matches = kwargs.get('min_matches', 8)
        else:
            # Fallback to default
            self.method = RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.999)
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 1.0)
            self.min_matches = kwargs.get('min_matches', 8)
        
        self.device = device or (torch.device('cuda') if GPU_AVAILABLE else torch.device('cpu'))
        
        # Initialize the chosen method
        self._init_method(**kwargs)
        
        logger.info(f"Geometric verification initialized with method: {self.method.value}")
    
    def _init_method(self, **kwargs):
        """Initialize the chosen RANSAC method"""
        if self.method == RANSACMethod.GPU_ADVANCED_MAGSAC:
            if not GPU_AVAILABLE or GPUAdvancedMAGSAC is None:
                logger.warning("GPU Advanced MAGSAC not available, falling back to OpenCV MAGSAC")
                self.method = RANSACMethod.OPENCV_MAGSAC
                self.gpu_magsac = None
            else:
                self.gpu_magsac = GPUAdvancedMAGSAC(self.device)
                logger.info("GPU Advanced MAGSAC initialized")
        
        elif self.method == RANSACMethod.PYRANSAC:
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
        Find essential matrix using the chosen RANSAC method
        
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
        
        try:
            if self.method == RANSACMethod.GPU_ADVANCED_MAGSAC and self.gpu_magsac is not None:
                E, inliers = self._find_essential_gpu_advanced(points1, points2, intrinsics)
            
            elif self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
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
            if self.method == RANSACMethod.GPU_ADVANCED_MAGSAC and self.gpu_magsac is not None:
                F, inliers = self._find_fundamental_gpu_advanced(points1, points2)
            
            elif self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
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
        """Find essential matrix using OpenCV MAGSAC"""
        
        # Use identity camera matrix if not provided
        camera_matrix = intrinsics if intrinsics is not None else np.eye(3)
        
        E, mask = cv2.findEssentialMat(
            points1, points2,
            cameraMatrix=camera_matrix,
            method=cv2.USAC_MAGSAC,
            prob=self.confidence,
            threshold=self.threshold,
            maxIters=self.max_iterations
        )
        
        inliers = mask.flatten().astype(bool) if mask is not None else None
        return E, inliers
    
    def _find_fundamental_opencv(self, 
                               points1: np.ndarray, 
                               points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Find fundamental matrix using OpenCV MAGSAC"""
        
        F, mask = cv2.findFundamentalMat(
            points1, points2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.threshold,
            confidence=self.confidence,
            maxIters=self.max_iterations
        )
        
        inliers = mask.flatten().astype(bool) if mask is not None else None
        return F, inliers
    
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
        
        if self.method == RANSACMethod.GPU_ADVANCED_MAGSAC and self.gpu_magsac is not None:
            info.update(self.gpu_magsac.get_performance_stats())
        
        return info
    
    @staticmethod
    def get_available_methods() -> list:
        """Get list of available RANSAC methods"""
        methods = [RANSACMethod.OPENCV_MAGSAC]  # Always available
        
        if PYRANSAC_AVAILABLE:
            methods.append(RANSACMethod.PYRANSAC)
        
        if GPU_AVAILABLE and GPUAdvancedMAGSAC is not None:
            methods.append(RANSACMethod.GPU_ADVANCED_MAGSAC)
        
        return methods
    
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
                
                if F is not None and inliers is not None:
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
        
        # For large datasets with GPU, use GPU Advanced MAGSAC
        if num_points > 5000 and has_gpu and GPUAdvancedMAGSAC is not None:
            return RANSACMethod.GPU_ADVANCED_MAGSAC
        
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