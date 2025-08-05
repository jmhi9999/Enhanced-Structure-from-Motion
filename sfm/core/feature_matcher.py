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
except ImportError:
    print("Warning: LightGlue not installed. Using fallback implementation.")

from .gpu_vocabulary_tree import GPUVocabularyTree

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
    
    def __init__(self, device: torch.device, use_vocabulary_tree: bool = True):
        self.device = device
        self.matcher = None
        self.use_vocabulary_tree = use_vocabulary_tree
        
        # Initialize vocabulary tree for efficient pair selection
        if self.use_vocabulary_tree:
            self.vocabulary_tree = GPUVocabularyTree(device)
        else:
            self.vocabulary_tree = None
        
        # Performance parameters
        self.max_pairs_per_image = 20  # Reduce from O(n²) to O(n)
        self.parallel_workers = min(8, torch.get_num_threads())
        self.batch_size = 32
        
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
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
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
            # Prepare data for LightGlue
            data = {
                'keypoints0': torch.from_numpy(feat1['keypoints']).float().to(self.device),
                'keypoints1': torch.from_numpy(feat2['keypoints']).float().to(self.device),
                'descriptors0': torch.from_numpy(feat1['descriptors']).float().to(self.device),
                'descriptors1': torch.from_numpy(feat2['descriptors']).float().to(self.device),
                'image_size0': torch.tensor(feat1['image_shape'][::-1]).float().to(self.device),
                'image_size1': torch.tensor(feat2['image_shape'][::-1]).float().to(self.device)
            }
            
            # Match features
            with torch.no_grad():
                pred = self.matcher(data)
            
            # Convert to numpy arrays
            matches = {
                'keypoints0': feat1['keypoints'],
                'keypoints1': feat2['keypoints'],
                'matches0': pred['matches0'].cpu().numpy(),
                'matches1': pred['matches1'].cpu().numpy(),
                'mscores0': pred['mscores0'].cpu().numpy() if 'mscores0' in pred else None,
                'mscores1': pred['mscores1'].cpu().numpy() if 'mscores1' in pred else None,
                'image_shape0': feat1['image_shape'],
                'image_shape1': feat2['image_shape']
            }
            
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