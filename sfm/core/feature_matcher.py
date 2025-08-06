"""
Enhanced LightGlue feature matcher with GPU vocabulary tree integration
Designed to be faster than hloc with O(n log n) complexity
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

try:
    from lightglue import LightGlue, SuperPoint, ALIKED, DISK
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    LightGlue = SuperPoint = ALIKED = DISK = None

try:
    from .gpu_vocabulary_tree import GPUVocabularyTree
    GPU_VOCAB_AVAILABLE = True
except ImportError:
    GPU_VOCAB_AVAILABLE = False
    GPUVocabularyTree = None

logger = logging.getLogger(__name__)


class EnhancedLightGlueMatcher:
    """
    Enhanced LightGlue feature matcher with GPU acceleration
    
    Key improvements over hloc:
    1. O(n log n) complexity using vocabulary tree vs O(n²) brute force
    2. Parallel feature matching with threading
    3. GPU-accelerated similarity search
    4. Intelligent pair selection
    5. Memory-efficient batch processing
    """
    
    def __init__(self, device: torch.device, use_vocabulary_tree: bool = True, feature_type: str = 'superpoint', config: Dict[str, Any] = None):
        self.device = device
        self.matcher = None
        self.use_vocabulary_tree = use_vocabulary_tree
        self.feature_type = feature_type
        self.config = config or {}
        
        # Initialize vocabulary tree for efficient pair selection
        if self.use_vocabulary_tree:
            self.vocabulary_tree = GPUVocabularyTree(device, config)
        else:
            self.vocabulary_tree = None
        
        # Performance parameters
        self.max_pairs_per_image = self.config.get('max_pairs_per_image', 20)
        self.parallel_workers = min(self.config.get('parallel_workers', 8), torch.get_num_threads())
        self.batch_size = self.config.get('batch_size', 32)
        
        # Timing statistics
        self.timing_stats = {
            'pair_selection': [],
            'feature_matching': [],
            'total': []
        }
        
        self._setup_matcher()
    
    def _setup_matcher(self):
        """Setup LightGlue matcher"""
        try:
            # LightGlue supports multiple extractors dynamically
            self.matcher = LightGlue(features=self.feature_type).eval().to(self.device)
        except NameError:
            # Fallback implementation
            self.matcher = self._create_fallback_matcher()
    
    def _create_fallback_matcher(self):
        """Fallback matcher implementation"""
        class SimpleMatcher:
            def __init__(self):
                self.device = self.device
                
            def match(self, data):
                # Simple feature matching using OpenCV
                kpts0, kpts1 = data['keypoints0'], data['keypoints1']
                desc0, desc1 = data['descriptors0'], data['descriptors1']
                
                # Use FLANN matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(desc0, desc1, k=2)
                
                # Apply ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                
                return {
                    'matches0': [m.queryIdx for m in good_matches],
                    'matches1': [m.trainIdx for m in good_matches],
                    'mscores0': [1.0 - m.distance / 1000.0 for m in good_matches],
                    'mscores1': [1.0 - m.distance / 1000.0 for m in good_matches]
                }
        
        return SimpleMatcher()
    
    def match_features(self, features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """
        Enhanced feature matching with intelligent pair selection
        Much faster than hloc's O(n²) brute force approach
        """
        start_time = time.time()
        
        image_paths = list(features.keys())
        matches = {}
        
        logger.info(f"Enhanced feature matching for {len(image_paths)} images...")
        
        # Step 1: Intelligent pair selection using vocabulary tree
        pair_selection_start = time.time()
        
        if self.use_vocabulary_tree and len(image_paths) > 10:
            # O(n log n) complexity using vocabulary tree
            logger.info("Using vocabulary tree for efficient pair selection...")
            pairs = self.vocabulary_tree.get_image_pairs_for_matching(
                features, self.max_pairs_per_image
            )
            logger.info(f"Selected {len(pairs)} pairs (vs {len(image_paths)*(len(image_paths)-1)//2} brute force)")
        else:
            # Fallback to all pairs for small datasets
            pairs = []
            for i in range(len(image_paths)):
                for j in range(i + 1, len(image_paths)):
                    pairs.append((image_paths[i], image_paths[j]))
            logger.info(f"Using brute force for {len(pairs)} pairs")
        
        pair_selection_time = time.time() - pair_selection_start
        self.timing_stats['pair_selection'].append(pair_selection_time)
        
        # Step 2: Parallel feature matching
        matching_start = time.time()
        
        if len(pairs) > self.parallel_workers:
            # Use parallel processing for large number of pairs
            matches = self._match_pairs_parallel(features, pairs)
        else:
            # Sequential matching for small datasets
            matches = self._match_pairs_sequential(features, pairs)
        
        matching_time = time.time() - matching_start
        self.timing_stats['feature_matching'].append(matching_time)
        
        total_time = time.time() - start_time
        self.timing_stats['total'].append(total_time)
        
        logger.info(f"Feature matching completed in {total_time:.2f}s "
                   f"(pair selection: {pair_selection_time:.2f}s, matching: {matching_time:.2f}s)")
        logger.info(f"Successfully matched {len(matches)}/{len(pairs)} pairs")
        
        return matches
    
    def _match_pairs_parallel(self, features: Dict[str, Any], 
                            pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """Match pairs in parallel for better performance"""
        matches = {}
        
        # Process pairs in batches to manage memory
        batch_size = self.batch_size
        
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
                batch_pairs = pairs[batch_start:batch_end]
                
                # Submit batch of matching tasks
                futures = {
                    executor.submit(self._match_pair_safe, features[img1], features[img2], img1, img2): (img1, img2)
                    for img1, img2 in batch_pairs
                }
                
                # Collect results
                for future in tqdm(as_completed(futures), total=len(futures),
                                 desc=f"Matching batch {batch_start//batch_size + 1}",
                                 leave=False):
                    img1, img2 = futures[future]
                    try:
                        pair_matches = future.result()
                        if pair_matches is not None:
                            matches[(img1, img2)] = pair_matches
                    except Exception as e:
                        logger.warning(f"Failed to match {img1} and {img2}: {e}")
        
        return matches
    
    def _match_pairs_sequential(self, features: Dict[str, Any],
                              pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """Sequential matching for small datasets"""
        matches = {}
        
        for img1_path, img2_path in tqdm(pairs, desc="Matching features"):
            try:
                pair_matches = self._match_pair(
                    features[img1_path], 
                    features[img2_path]
                )
                
                if pair_matches is not None:
                    matches[(img1_path, img2_path)] = pair_matches
                    
            except Exception as e:
                logger.warning(f"Failed to match {img1_path} and {img2_path}: {e}")
                continue
        
        return matches
    
    def _match_pair_safe(self, feat1: Dict, feat2: Dict, img1_path: str, img2_path: str) -> Optional[Dict]:
        """Thread-safe wrapper for pair matching"""
        try:
            return self._match_pair(feat1, feat2)
        except Exception as e:
            logger.warning(f"Error matching {img1_path} and {img2_path}: {e}")
            return None
    
    def _match_pair(self, feat1: Dict, feat2: Dict) -> Optional[Dict]:
        """Match features between a single pair of images"""
        try:
            # Validate feature data
            required_keys = ['keypoints', 'descriptors', 'image_shape']
            for key in required_keys:
                if key not in feat1:
                    logger.error(f"Missing key '{key}' in feat1")
                    return None
                if key not in feat2:
                    logger.error(f"Missing key '{key}' in feat2")
                    return None
            
            # Prepare data for LightGlue
            # Create dummy image tensors (LightGlue might need them for some operations)
            h0, w0 = feat1['image_shape']
            h1, w1 = feat2['image_shape']
            
            # Validate feature shapes
            if len(feat1['keypoints']) == 0 or len(feat2['keypoints']) == 0:
                logger.debug("One of the feature sets is empty")
                return None
            
            # Convert features to tensors and ensure proper dimensions
            # Debug shapes before conversion
            logger.debug(f"feat1 keypoints shape: {feat1['keypoints'].shape}, descriptors shape: {feat1['descriptors'].shape}")
            logger.debug(f"feat2 keypoints shape: {feat2['keypoints'].shape}, descriptors shape: {feat2['descriptors'].shape}")
            
            kpts0 = torch.from_numpy(feat1['keypoints']).float().to(self.device)
            kpts1 = torch.from_numpy(feat2['keypoints']).float().to(self.device)
            desc0 = torch.from_numpy(feat1['descriptors']).float().to(self.device)
            desc1 = torch.from_numpy(feat2['descriptors']).float().to(self.device)
            
            # Debug tensor shapes after conversion
            logger.debug(f"kpts0 shape: {kpts0.shape}, desc0 shape: {desc0.shape}")
            logger.debug(f"kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}")
            
            # Ensure keypoints have correct shape: [N, 2]
            if kpts0.dim() == 1:
                kpts0 = kpts0.unsqueeze(0)
            if kpts1.dim() == 1:
                kpts1 = kpts1.unsqueeze(0)
            
            # Ensure descriptors have correct shape: [N, D] 
            if desc0.dim() == 1:
                desc0 = desc0.unsqueeze(0)
            if desc1.dim() == 1:
                desc1 = desc1.unsqueeze(0)
            
            # Final shape validation
            logger.debug(f"Final kpts0 shape: {kpts0.shape}, desc0 shape: {desc0.shape}")
            logger.debug(f"Final kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}")
            
            # LightGlue expects the data in this specific format
            data = {
                "image0": {
                    "keypoints": kpts0,
                    "descriptors": desc0,
                    "image_size": torch.tensor([w0, h0]).float().to(self.device),
                },
                "image1": {
                    "keypoints": kpts1, 
                    "descriptors": desc1,
                    "image_size": torch.tensor([w1, h1]).float().to(self.device),
                }
            }
            
            # Match features
            with torch.no_grad():
                try:
                    pred = self.matcher(data)
                except Exception as e:
                    logger.error(f"LightGlue matcher failed: {e}")
                    logger.error(f"Data shapes - kpts0: {data['keypoints0'].shape}, kpts1: {data['keypoints1'].shape}")
                    logger.error(f"Data shapes - desc0: {data['descriptors0'].shape}, desc1: {data['descriptors1'].shape}")
                    raise e
            
            # Convert to numpy arrays - handle tensor dimensions properly
            matches0_tensor = pred['matches0']
            matches1_tensor = pred['matches1']
            
            # Debug tensor shapes
            logger.debug(f"matches0 shape: {matches0_tensor.shape}, matches1 shape: {matches1_tensor.shape}")
            
            # Handle different tensor shapes that LightGlue might return
            # LightGlue returns matches as indices, typically shape [N] or [1, N]
            if matches0_tensor.dim() > 1:
                # Remove batch dimensions
                while matches0_tensor.dim() > 1:
                    matches0_tensor = matches0_tensor.squeeze(0)
            if matches1_tensor.dim() > 1:
                while matches1_tensor.dim() > 1:
                    matches1_tensor = matches1_tensor.squeeze(0)
            
            matches = {
                'keypoints0': feat1['keypoints'],
                'keypoints1': feat2['keypoints'],
                'matches0': matches0_tensor.cpu().numpy(),
                'matches1': matches1_tensor.cpu().numpy(),
                'mscores0': pred['mscores0'].cpu().numpy() if 'mscores0' in pred else None,
                'mscores1': pred['mscores1'].cpu().numpy() if 'mscores1' in pred else None,
                'image_shape0': feat1['image_shape'],
                'image_shape1': feat2['image_shape']
            }
            
            # Handle score tensors if they exist
            if matches['mscores0'] is not None and matches['mscores0'].ndim > 1:
                matches['mscores0'] = matches['mscores0'].flatten()
            if matches['mscores1'] is not None and matches['mscores1'].ndim > 1:
                matches['mscores1'] = matches['mscores1'].flatten()
            
            # Filter matches based on confidence
            if matches['mscores0'] is not None:
                confidence_threshold = 0.2
                confident_matches = matches['mscores0'] > confidence_threshold
                
                matches['matches0'] = matches['matches0'][confident_matches]
                matches['matches1'] = matches['matches1'][confident_matches]
                matches['mscores0'] = matches['mscores0'][confident_matches]
                matches['mscores1'] = matches['mscores1'][confident_matches]
            
            # Only return if we have enough matches
            if len(matches['matches0']) >= 8:
                return matches
            else:
                return None
                
        except Exception as e:
            # More detailed error logging
            import traceback
            error_details = f"Error in feature matching: {e}\n"
            error_details += f"Traceback: {traceback.format_exc()}"
            logger.error(error_details)
            print(f"Error in feature matching: {e}")
            return None
    
    def get_matched_keypoints(self, matches: Dict, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get matched keypoints for a specific image pair"""
        pair_key = (img1_path, img2_path)
        if pair_key not in matches:
            return np.array([]), np.array([])
        
        match_data = matches[pair_key]
        kpts0 = match_data['keypoints0']
        kpts1 = match_data['keypoints1']
        matches0 = match_data['matches0']
        matches1 = match_data['matches1']
        
        # Get matched keypoints
        matched_kpts0 = kpts0[matches0]
        matched_kpts1 = kpts1[matches1]
        
        return matched_kpts0, matched_kpts1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the matcher"""
        stats = {
            'avg_pair_selection_time': np.mean(self.timing_stats['pair_selection']) if self.timing_stats['pair_selection'] else 0.0,
            'avg_matching_time': np.mean(self.timing_stats['feature_matching']) if self.timing_stats['feature_matching'] else 0.0,
            'avg_total_time': np.mean(self.timing_stats['total']) if self.timing_stats['total'] else 0.0,
            'num_matching_sessions': len(self.timing_stats['total']),
            'using_vocabulary_tree': self.use_vocabulary_tree,
            'max_pairs_per_image': self.max_pairs_per_image,
            'parallel_workers': self.parallel_workers
        }
        
        if self.vocabulary_tree:
            vocab_stats = self.vocabulary_tree.get_performance_stats()
            stats.update({f'vocab_{k}': v for k, v in vocab_stats.items()})
        
        return stats


# Keep backward compatibility
LightGlueMatcher = EnhancedLightGlueMatcher


class FeatureMatcher:
    """Main feature matcher class that wraps the enhanced matcher"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.use_vocabulary_tree = config.get('use_vocabulary_tree', True)
        self.max_pairs_per_image = config.get('max_pairs_per_image', 20)
        
        # Create the actual matcher
        feature_type = config.get('feature_type', 'superpoint')
        self.matcher = EnhancedLightGlueMatcher(
            device=self.device,
            use_vocabulary_tree=self.use_vocabulary_tree,
            feature_type=feature_type,
            config=config
        )
    
    def match(self, features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """Match features between images"""
        return self.matcher.match_features(features) 