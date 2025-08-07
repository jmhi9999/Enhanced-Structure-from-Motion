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

try:
    from .gpu_brute_force_matcher import GPUBruteForceMatcher
    GPU_BRUTE_FORCE_AVAILABLE = True
except ImportError:
    GPU_BRUTE_FORCE_AVAILABLE = False
    GPUBruteForceMatcher = None

logger = logging.getLogger(__name__)


class EnhancedLightGlueMatcher:
    """
    Enhanced LightGlue feature matcher with GPU acceleration
    
    Key improvements over hloc:
    1. GPU tensor-based brute force matching for maximum performance
    2. Features kept in GPU memory as tensors (no file I/O)
    3. GPU-accelerated direct comparisons
    4. Memory-efficient batch processing
    5. Option to use vocabulary tree for large datasets
    """
    
    def __init__(self, device: torch.device, use_brute_force: bool = True, use_vocabulary_tree: bool = False, feature_type: str = 'superpoint', config: Dict[str, Any] = None):
        self.device = device
        self.matcher = None
        self.use_brute_force = use_brute_force
        self.use_vocabulary_tree = use_vocabulary_tree
        self.feature_type = feature_type
        self.config = config or {}
        
        # Initialize GPU brute force matcher (preferred method)
        if self.use_brute_force and GPU_BRUTE_FORCE_AVAILABLE:
            self.gpu_brute_force_matcher = GPUBruteForceMatcher(device, feature_type, config)
            logger.info("Using GPU brute force matcher")
        else:
            self.gpu_brute_force_matcher = None
        
        # Initialize vocabulary tree as fallback for very large datasets
        if self.use_vocabulary_tree and GPU_VOCAB_AVAILABLE:
            self.vocabulary_tree = GPUVocabularyTree(device, config)
            logger.info("Using vocabulary tree matcher")
        else:
            self.vocabulary_tree = None
        
        # Performance parameters
        self.max_pairs_per_image = self.config.get('max_pairs_per_image', 20)
        self.parallel_workers = min(self.config.get('parallel_workers', 8), torch.get_num_threads())
        self.batch_size = self.config.get('batch_size', 32)
        
        # Matching parameters - More permissive for reconstruction
        self.confidence_threshold = self.config.get('confidence_threshold', 0.1)
        self.min_matches = self.config.get('min_matches', 4)
        
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
            # LightGlue configuration matching the provided format
            lightglue_conf = {
                "features": self.feature_type,
                "depth_confidence": self.config.get('depth_confidence', 0.95),
                "width_confidence": self.config.get('width_confidence', 0.99),
                "compile": self.config.get('compile', False),
            }
            
            # LightGlue supports multiple extractors dynamically
            self.matcher = LightGlue(features=lightglue_conf["features"], 
                                   depth_confidence=lightglue_conf["depth_confidence"],
                                   width_confidence=lightglue_conf["width_confidence"]).eval().to(self.device)
            
            if lightglue_conf["compile"]:
                self.matcher.compile()
                
        except NameError:
            logger.error("LightGlue is not available. Please install it to use this matcher.")
            raise ImportError("LightGlue is not available. Please install it to use this matcher.")
    
    
    def match_features(self, features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """
        Enhanced feature matching using GPU tensor operations
        Maximum performance with features kept in GPU memory
        """
        start_time = time.time()
        
        image_paths = list(features.keys())
        matches = {}
        
        logger.info(f"GPU tensor-based feature matching for {len(image_paths)} images...")
        
        # Use GPU brute force matcher (preferred method)
        if self.gpu_brute_force_matcher is not None:
            logger.info("Using GPU brute force matcher with tensor operations...")
            
            # Load features to GPU memory as tensors
            self.gpu_brute_force_matcher.load_features(features)
            
            # Perform GPU brute force matching
            max_total_pairs = self.config.get('max_total_pairs', None)
            matches = self.gpu_brute_force_matcher.match_all_pairs(max_pairs=max_total_pairs)
            
        # Fallback to vocabulary tree for very large datasets
        elif self.vocabulary_tree is not None and len(image_paths) > 50:
            logger.info("Using vocabulary tree for large dataset...")
            
            # Get smart pairs using vocabulary tree
            pairs = self.vocabulary_tree.get_image_pairs_for_matching(
                features, self.max_pairs_per_image
            )
            logger.info(f"Selected {len(pairs)} pairs using vocabulary tree")
            
            # Match selected pairs
            matches = self._match_pairs_sequential(features, pairs)
            
        else:
            # Traditional brute force with all pairs (fallback)
            logger.info("Using traditional brute force matching...")
            pairs = []
            for i in range(len(image_paths)):
                for j in range(i + 1, len(image_paths)):
                    pairs.append((image_paths[i], image_paths[j]))
            
            # Match all pairs
            if len(pairs) > self.parallel_workers:
                matches = self._match_pairs_parallel(features, pairs)
            else:
                matches = self._match_pairs_sequential(features, pairs)
        
        total_time = time.time() - start_time
        self.timing_stats['total'].append(total_time)
        
        logger.info(f"Feature matching completed in {total_time:.2f}s")
        logger.info(f"Successfully matched {len(matches)} pairs")
        
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
                            logger.debug(f"Matched {img1} and {img2} successfully with {len(pair_matches['matches0'])} matches")
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
            
            # Ensure keypoints have correct shape: [1, N, 2] for LightGlue
            if kpts0.dim() == 1:
                kpts0 = kpts0.unsqueeze(0)
            if kpts0.dim() == 2:  # [N, 2] -> [1, N, 2]
                kpts0 = kpts0.unsqueeze(0)
                
            if kpts1.dim() == 1:
                kpts1 = kpts1.unsqueeze(0)
            if kpts1.dim() == 2:  # [N, 2] -> [1, N, 2]
                kpts1 = kpts1.unsqueeze(0)
            
            # Ensure descriptors have correct shape for LightGlue: [1, N, D] before transpose
            if desc0.dim() == 1:
                desc0 = desc0.unsqueeze(0)
            if desc0.dim() == 2:  
                if desc0.shape[0] > desc0.shape[1]:
                    # If shape is [N, D], add batch dimension -> [1, N, D]
                    desc0 = desc0.unsqueeze(0)
                else:
                    # If shape is [D, N], transpose and add batch dim -> [1, N, D]
                    desc0 = desc0.transpose(0, 1).unsqueeze(0)
                
            if desc1.dim() == 1:
                desc1 = desc1.unsqueeze(0)
            if desc1.dim() == 2:  
                if desc1.shape[0] > desc1.shape[1]:
                    # If shape is [N, D], add batch dimension -> [1, N, D]
                    desc1 = desc1.unsqueeze(0)
                else:
                    # If shape is [D, N], transpose and add batch dim -> [1, N, D]
                    desc1 = desc1.transpose(0, 1).unsqueeze(0)
            
            # Final shape validation
            logger.debug(f"Final kpts0 shape: {kpts0.shape}, desc0 shape: {desc0.shape}")
            logger.debug(f"Final kpts1 shape: {kpts1.shape}, desc1 shape: {desc1.shape}")
            
            # LightGlue expects nested format
            data = {
                "image0": {
                    "keypoints": kpts0,
                    "descriptors": desc0,  # Already in correct shape [1, N, D]
                },
                "image1": {
                    "keypoints": kpts1,
                    "descriptors": desc1,  # Already in correct shape [1, N, D]
                }
            }
            
            # Match features
            with torch.no_grad():
                try:
                    pred = self.matcher(data)
                except Exception as e:
                    logger.error(f"LightGlue matcher failed: {e}")
                    logger.error(f"Data shapes - kpts0: {data['image0']['keypoints'].shape}, kpts1: {data['image1']['keypoints'].shape}")
                    logger.error(f"Data shapes - desc0: {data['image0']['descriptors'].shape}, desc1: {data['image1']['descriptors'].shape}")
                    raise e
            
            # Convert to numpy arrays - handle LightGlue output format
            # LightGlue returns 'matches' (shape [S, 2]) and 'scores' (shape [S])
            if 'matches' in pred:
                matches_tensor = pred['matches']  # [S, 2] where S is number of matches
                scores_tensor = pred.get('scores', None)  # [S]
                
                # Handle case where matches_tensor might be a list
                if isinstance(matches_tensor, list):
                    if len(matches_tensor) > 0:
                        matches_tensor = torch.stack(matches_tensor) if isinstance(matches_tensor[0], torch.Tensor) else torch.tensor(matches_tensor)
                    else:
                        matches_tensor = torch.empty(0, 2)
                
                # Handle case where scores_tensor might be a list
                if isinstance(scores_tensor, list):
                    if len(scores_tensor) > 0:
                        scores_tensor = torch.stack(scores_tensor) if isinstance(scores_tensor[0], torch.Tensor) else torch.tensor(scores_tensor)
                    else:
                        scores_tensor = torch.empty(0)
                
                # Debug tensor shapes (now safe to call .shape)
                logger.debug(f"matches shape: {matches_tensor.shape}")
                if scores_tensor is not None:
                    logger.debug(f"scores shape: {scores_tensor.shape}")
                
                # Remove batch dimensions if they exist
                while matches_tensor.dim() > 2:
                    matches_tensor = matches_tensor.squeeze(0)
                if scores_tensor is not None:
                    while scores_tensor.dim() > 1:
                        scores_tensor = scores_tensor.squeeze(0)
                
                # Extract matches - LightGlue format is [match_idx0, match_idx1] pairs
                if matches_tensor.shape[0] > 0:  # Check if we have matches
                    matches0_indices = matches_tensor[:, 0].cpu().numpy()
                    matches1_indices = matches_tensor[:, 1].cpu().numpy()
                    scores = scores_tensor.cpu().numpy() if scores_tensor is not None else np.ones(len(matches0_indices))
                else:
                    matches0_indices = np.array([])
                    matches1_indices = np.array([])
                    scores = np.array([])
            else:
                # Fallback for older LightGlue versions
                matches0_data = pred.get('matches0', torch.tensor([]))
                matches1_data = pred.get('matches1', torch.tensor([]))
                scores_data = pred.get('mscores0', None)
                
                # Handle lists in fallback case too
                if isinstance(matches0_data, list):
                    matches0_data = torch.tensor(matches0_data) if matches0_data else torch.tensor([])
                if isinstance(matches1_data, list):
                    matches1_data = torch.tensor(matches1_data) if matches1_data else torch.tensor([])
                if isinstance(scores_data, list):
                    scores_data = torch.tensor(scores_data) if scores_data else None
                
                matches0_indices = matches0_data.cpu().numpy()
                matches1_indices = matches1_data.cpu().numpy()
                scores = scores_data.cpu().numpy() if scores_data is not None else np.ones(len(matches0_indices))
            
            matches = {
                'keypoints0': feat1['keypoints'],
                'keypoints1': feat2['keypoints'],
                'matches0': matches0_indices,
                'matches1': matches1_indices,
                'mscores0': scores,
                'mscores1': scores,  # Same scores for both
                'image_shape0': feat1['image_shape'],
                'image_shape1': feat2['image_shape']
            }
            
            # Filter matches based on confidence
            if len(scores) > 0:
                confident_matches = scores > self.confidence_threshold
                
                matches['matches0'] = matches['matches0'][confident_matches]
                matches['matches1'] = matches['matches1'][confident_matches]
                matches['mscores0'] = matches['mscores0'][confident_matches]
                matches['mscores1'] = matches['mscores1'][confident_matches]
            
            # Only return if we have enough matches
            if len(matches['matches0']) >= self.min_matches:
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
        self.use_brute_force = config.get('use_brute_force', True)
        self.use_vocabulary_tree = config.get('use_vocabulary_tree', False)  # Disabled by default, GPU brute force is preferred
        self.max_pairs_per_image = config.get('max_pairs_per_image', 20)
        
        # Create the actual matcher with GPU brute force as default
        feature_type = config.get('feature_type', 'superpoint')
        self.matcher = EnhancedLightGlueMatcher(
            device=self.device,
            use_brute_force=self.use_brute_force,
            use_vocabulary_tree=self.use_vocabulary_tree,
            feature_type=feature_type,
            config=config
        )
    
    def match(self, features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """Match features between images"""
        return self.matcher.match_features(features) 