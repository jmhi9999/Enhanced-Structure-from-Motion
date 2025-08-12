"""
Geometric and Semantic verification module
Supports multiple RANSAC implementations and semantic filtering
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any, Literal
from enum import Enum
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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
    Geometric and Semantic verification with multiple RANSAC implementation options
    """
    
    def __init__(self, 
                 config_or_method = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        # ... (rest of the __init__ method remains the same as before)
        if isinstance(config_or_method, dict):
            config = config_or_method
            method_str = config.get('geometric_method', 'opencv_magsac')
            try:
                self.method = RANSACMethod(method_str)
            except ValueError:
                self.method = RANSACMethod.OPENCV_MAGSAC
            self.confidence = config.get('confidence', 0.95)
            self.max_iterations = config.get('max_iterations', 10000)
            self.threshold = config.get('threshold', 3.0)
            self.min_matches = config.get('min_matches', 8)
        elif config_or_method is None or isinstance(config_or_method, RANSACMethod):
            self.method = config_or_method or RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.95)
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 3.0)
            self.min_matches = kwargs.get('min_matches', 8)
        else:
            self.method = RANSACMethod.OPENCV_MAGSAC
            self.confidence = kwargs.get('confidence', 0.95)
            self.max_iterations = kwargs.get('max_iterations', 10000)
            self.threshold = kwargs.get('threshold', 3.0)
            self.min_matches = kwargs.get('min_matches', 8)
        
        self.device = device or (torch.device('cuda') if GPU_AVAILABLE else torch.device('cpu'))
        self._init_method(**kwargs)
        logger.info(f"Geometric verification initialized with method: {self.method.value}")

    def _init_method(self, **kwargs):
        if self.method == RANSACMethod.PYRANSAC:
            if not PYRANSAC_AVAILABLE:
                logger.warning("pyransac not available, falling back to OpenCV MAGSAC")
                self.method = RANSACMethod.OPENCV_MAGSAC
        if self.method == RANSACMethod.OPENCV_MAGSAC:
            logger.info("Using OpenCV MAGSAC")

    def filter_by_semantics(self, matches: Dict, features: Dict, semantic_masks: Dict) -> Dict:
        """
        Filters matches based on semantic consistency.

        Args:
            matches (Dict): The raw matches from the matcher.
            features (Dict): The features dictionary containing keypoints.
            semantic_masks (Dict): A dictionary mapping image paths to their semantic masks.

        Returns:
            Dict: A new dictionary containing only the semantically consistent matches.
        """
        logger.info("Starting semantic filtering of matches...")
        semantically_verified_matches = {}
        
        for pair_key, match_data in tqdm(matches.items(), desc="Semantic Filtering"):
            # Handle both tuple and string keys
            if isinstance(pair_key, tuple):
                img_path1, img_path2 = pair_key
            else:
                # Fallback for string keys
                img_path1_str, img_path2_str = pair_key.split('-')
                # Find the full path from the features dictionary keys
                img_path1 = next((p for p in features.keys() if Path(p).name == img_path1_str), None)
                img_path2 = next((p for p in features.keys() if Path(p).name == img_path2_str), None)

            if not img_path1 or not img_path2:
                logger.warning(f"Could not find image paths for pair {pair_key}")
                continue

            mask1 = semantic_masks.get(img_path1)
            mask2 = semantic_masks.get(img_path2)

            if mask1 is None or mask2 is None:
                logger.debug(f"Skipping semantic check for pair {pair_key} due to missing masks.")
                semantically_verified_matches[pair_key] = match_data
                continue

            kpts1 = features[img_path1]['keypoints']
            kpts2 = features[img_path2]['keypoints']
            
            matches0 = match_data['matches0']
            
            good_indices = []
            for i in range(len(matches0)):
                idx1 = matches0[i]
                idx2 = match_data['matches1'][i]

                pt1 = kpts1[idx1]
                pt2 = kpts2[idx2]

                # Get semantic label at keypoint locations
                # Ensure coordinates are within mask bounds
                y1, x1 = int(pt1[1]), int(pt1[0])
                y2, x2 = int(pt2[1]), int(pt2[0])
                
                if 0 <= y1 < mask1.shape[0] and 0 <= x1 < mask1.shape[1] and \
                   0 <= y2 < mask2.shape[0] and 0 <= x2 < mask2.shape[1]:
                    
                    label1 = mask1[y1, x1]
                    label2 = mask2[y2, x2]

                    # Keep match if labels are identical
                    if label1 == label2:
                        good_indices.append(i)

            if len(good_indices) < self.min_matches:
                logger.debug(f"Pair {pair_key} has too few semantically consistent matches: {len(good_indices)}")
                continue

            # Create a new match_data dictionary with filtered matches
            new_match_data = match_data.copy()
            new_match_data['matches0'] = match_data['matches0'][good_indices]
            new_match_data['matches1'] = match_data['matches1'][good_indices]
            if 'mscores0' in match_data:
                new_match_data['mscores0'] = match_data['mscores0'][good_indices]
            if 'mscores1' in match_data:
                new_match_data['mscores1'] = match_data['mscores1'][good_indices]
            
            semantically_verified_matches[pair_key] = new_match_data
            logger.debug(f"Pair {pair_key}: {len(good_indices)}/{len(matches0)} matches passed semantic check.")

        logger.info(f"Semantic filtering complete. {len(semantically_verified_matches)}/{len(matches)} pairs remain.")
        return semantically_verified_matches

    # ... (all other methods like find_essential_matrix, verify_matches, etc., remain here)
    def find_essential_matrix(self, 
                            points1: np.ndarray, 
                            points2: np.ndarray,
                            intrinsics: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find essential matrix using the chosen RANSAC method with robust preprocessing
        """
        # ... (implementation unchanged)
        start_time = time.time()
        if len(points1) < 8: return None, None
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        points1, points2 = self._prefilter_points(points1, points2)
        if len(points1) < 8: return None, None
        try:
            if self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
                E, inliers = self._find_essential_pyransac(points1, points2, intrinsics)
            else:
                E, inliers = self._find_essential_opencv(points1, points2, intrinsics)
            return E, inliers
        except Exception as e:
            logger.error(f"Essential matrix estimation failed: {e}")
            return None, None

    def find_fundamental_matrix(self, 
                              points1: np.ndarray, 
                              points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find fundamental matrix using the chosen RANSAC method
        """
        # ... (implementation unchanged)
        start_time = time.time()
        if len(points1) < 8: return None, None
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        try:
            if self.method == RANSACMethod.PYRANSAC and PYRANSAC_AVAILABLE:
                F, inliers = self._find_fundamental_pyransac(points1, points2)
            else:
                F, inliers = self._find_fundamental_opencv(points1, points2)
            return F, inliers
        except Exception as e:
            logger.error(f"Fundamental matrix estimation failed: {e}")
            return None, None

    def _find_essential_opencv(self, 
                             points1: np.ndarray, 
                             points2: np.ndarray,
                             intrinsics: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation unchanged)
        camera_matrix = intrinsics if intrinsics is not None else np.eye(3)
        threshold, confidence, max_iters = self._adapt_parameters(points1, points2)
        try:
            E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=camera_matrix, method=cv2.USAC_MAGSAC, prob=confidence, threshold=threshold, maxIters=max_iters)
            if E is None or E.size == 0:
                E, mask = cv2.findEssentialMat(points1, points2, cameraMatrix=camera_matrix, method=cv2.RANSAC, prob=0.90, threshold=threshold * 1.5, maxIters=max_iters)
            inliers = mask.flatten().astype(bool) if mask is not None else None
            return E, inliers
        except Exception as e:
            return None, None

    def _find_fundamental_opencv(self, 
                               points1: np.ndarray, 
                               points2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # ... (implementation unchanged)
        if len(points1) < 8 or len(points2) < 8: return None, None
        points1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
        points2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
        points1, points2 = self._prefilter_points(points1, points2)
        if len(points1) < 8 or len(points2) < 8: return None, None
        unique_points1 = len(np.unique(points1, axis=0))
        unique_points2 = len(np.unique(points2, axis=0))
        if unique_points1 < 8 or unique_points2 < 8: return None, None
        threshold, confidence, max_iters = self._adapt_parameters(points1, points2)
        try:
            F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.USAC_MAGSAC, ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters)
            if F is None or F.size == 0: return self._find_fundamental_fallback(points1, points2)
            inliers = mask.flatten().astype(bool) if mask is not None else None
            if inliers is not None and np.sum(inliers) < 8: return self._find_fundamental_fallback(points1, points2)
            return F, inliers
        except cv2.error as e:
            return self._find_fundamental_fallback(points1, points2)

    def _find_essential_pyransac(self, points1, points2, intrinsics): return self._find_essential_opencv(points1, points2, intrinsics)
    def _find_fundamental_pyransac(self, points1, points2): return self._find_fundamental_opencv(points1, points2)
    def _find_essential_gpu_advanced(self, points1, points2, intrinsics): return self._find_essential_opencv(points1, points2, intrinsics)
    def _find_fundamental_gpu_advanced(self, points1, points2): return self._find_fundamental_opencv(points1, points2)

    def get_method_info(self) -> Dict[str, Any]:
        # ... (implementation unchanged)
        return {'method': self.method.value, 'confidence': self.confidence, 'max_iterations': self.max_iterations, 'threshold': self.threshold, 'gpu_available': GPU_AVAILABLE, 'pyransac_available': PYRANSAC_AVAILABLE}

    @staticmethod
    def get_available_methods() -> list:
        # ... (implementation unchanged)
        methods = [RANSACMethod.OPENCV_MAGSAC]
        if PYRANSAC_AVAILABLE: methods.append(RANSACMethod.PYRANSAC)
        return methods

    def verify_matches(self, feat1, feat2, match_data):
        # ... (implementation unchanged)
        try:
            kpts0, kpts1 = feat1['keypoints'], feat2['keypoints']
            matches0, matches1 = match_data['matches0'], match_data['matches1']
            if len(matches0) < self.min_matches: return None
            points0, points1 = kpts0[matches0], kpts1[matches1]
            F, inliers = self.find_fundamental_matrix(points0, points1)
            if F is not None and inliers is not None and np.sum(inliers) >= self.min_matches:
                verified_match = {'matches0': matches0[inliers], 'matches1': matches1[inliers]}
                return verified_match
            return None
        except Exception:
            return None

    def verify(self, matches, features):
        verified_matches = {}
        logger.info(f"Verifying {len(matches)} image pairs geometrically...")
        for pair_key, match_data in tqdm(matches.items(), desc="Geometric Verification"):
            try:
                img_path1_str, img_path2_str = pair_key.split('-')
                img_path1 = next((p for p in features.keys() if Path(p).name == img_path1_str), None)
                img_path2 = next((p for p in features.keys() if Path(p).name == img_path2_str), None)
                if not img_path1 or not img_path2: continue

                kpts0 = features[img_path1]['keypoints']
                kpts1 = features[img_path2]['keypoints']
                matches0 = match_data['matches0']
                matches1 = match_data['matches1']
                
                if len(matches0) < self.min_matches: continue
                    
                points0 = kpts0[matches0]
                points1 = kpts1[matches1]
                
                F, inliers = self.find_fundamental_matrix(points0, points1)
                
                if F is not None and inliers is not None and np.sum(inliers) >= self.min_matches:
                    new_match_data = match_data.copy()
                    new_match_data['matches0'] = matches0[inliers]
                    new_match_data['matches1'] = matches1[inliers]
                    if 'mscores0' in match_data:
                        new_match_data['mscores0'] = match_data['mscores0'][inliers]
                    if 'mscores1' in match_data:
                        new_match_data['mscores1'] = match_data['mscores1'][inliers]
                    verified_matches[pair_key] = new_match_data
            except Exception as e:
                logger.warning(f"Failed to verify pair {pair_key}: {e}")
                continue
        
        logger.info(f"Geometrically verified {len(verified_matches)}/{len(matches)} pairs successfully")
        return verified_matches

    def _prefilter_points(self, points1, points2):
        # ... (implementation unchanged)
        valid_mask = (np.isfinite(points1).all(axis=1) & np.isfinite(points2).all(axis=1))
        points1, points2 = points1[valid_mask], points2[valid_mask]
        if len(points1) < 8: return points1, points2
        return points1, points2

    def _adapt_parameters(self, points1, points2):
        # ... (implementation unchanged)
        return self.threshold, self.confidence, self.max_iterations

    def _find_fundamental_fallback(self, points1, points2):
        # ... (implementation unchanged)
        try:
            F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_LMEDS)
            if F is not None: return F, mask.flatten().astype(bool)
        except: pass
        try:
            F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_RANSAC, ransacReprojThreshold=5.0, confidence=0.85)
            if F is not None: return F, mask.flatten().astype(bool)
        except: pass
        return None, None

# ... (convenience functions remain the same)
def find_essential_matrix(points1, points2, method=RANSACMethod.OPENCV_MAGSAC, intrinsics=None, **kwargs):
    verifier = GeometricVerification(method=method, **kwargs)
    return verifier.find_essential_matrix(points1, points2, intrinsics)

def find_fundamental_matrix(points1, points2, method=RANSACMethod.OPENCV_MAGSAC, **kwargs):
    verifier = GeometricVerification(method=method, **kwargs)
    return verifier.find_fundamental_matrix(points1, points2)
