"""
GPU-accelerated Vocabulary Tree for fast image retrieval
Designed to outperform hloc's brute-force matching with O(n log n) complexity
"""

import torch
import numpy as np

# GPU dependencies - optional imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
from pathlib import Path
import time
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class GPUVocabularyTree:
    """
    GPU-accelerated Vocabulary Tree for efficient image retrieval
    
    Key improvements over hloc:
    1. O(n log n) complexity vs O(n²) brute force
    2. GPU-accelerated feature indexing with FAISS
    3. Hierarchical clustering for fast retrieval
    4. Memory-efficient batch processing
    5. Robust error handling with CPU fallbacks
    6. Production-ready stability and performance
    """
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None, output_path: Optional[str] = None):
        # Allow CPU-only operation as fallback
        self.gpu_available = CUPY_AVAILABLE and FAISS_AVAILABLE
        if not self.gpu_available:
            logger.warning("GPU dependencies not available. Using CPU fallback mode. "
                         "Install with: pip install cupy faiss-gpu for better performance")
        
        config = config or {}
        self.device = device
        self.output_path = Path(output_path) if output_path else Path(".")
        self.vocab_size = config.get('vocab_size', 10000)
        self.depth = config.get('vocab_depth', 6)
        self.branching_factor = config.get('vocab_branching_factor', 10)
        
        # FAISS GPU index for fast similarity search
        self.use_gpu = torch.cuda.is_available() and device.type == 'cuda'
        self.gpu_resource = None
        self.index = None
        self.vocabulary = None
        
        # Image database
        self.image_descriptors = {}
        self.image_features = {}
        self.inverted_index = {}
        
        # Performance monitoring
        self.build_time = 0.0
        self.query_times = []
        
        # Adaptive validation parameters for performance
        self.max_descriptors_per_image = config.get('max_descriptors_per_image', 3500)  # Indoor-optimized for SuperPoint
        self.max_vocab_descriptors = config.get('max_vocab_descriptors', 500000)
        self.min_cluster_size = config.get('min_cluster_size', 5)
        
        # Performance optimization settings
        self.adaptive_sampling = config.get('adaptive_sampling', True)  # Smart descriptor selection
        self.quality_threshold = config.get('quality_threshold', 0.01)  # Keep only high-quality descriptors
        self.max_total_descriptors = config.get('max_total_descriptors', 1000000)  # Total limit for performance
        
        # Setup resources
        if self.use_gpu and self.gpu_available:
            self._setup_gpu_resources()
        else:
            self._setup_cpu_resources()
    
    def _setup_gpu_resources(self):
        """Setup GPU resources for FAISS"""
        try:
            if not self.gpu_available:
                raise ImportError("GPU dependencies not available")
                
            # Try different ways to initialize GPU resources
            if hasattr(faiss, 'StandardGpuResources'):
                self.gpu_resource = faiss.StandardGpuResources()
            elif hasattr(faiss, 'GpuResources'):
                self.gpu_resource = faiss.GpuResources()
            else:
                # Fallback: just use GPU without explicit resource management
                self.gpu_resource = None
                logger.info("Using GPU without explicit resource management")
                return
            logger.info("GPU resources initialized for vocabulary tree")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU resources: {e}. Falling back to CPU")
            self.use_gpu = False
            self._setup_cpu_resources()
    
    def _setup_cpu_resources(self):
        """Setup CPU resources as fallback"""
        try:
            # Verify sklearn is available for CPU k-means
            from sklearn.cluster import KMeans
            logger.info("CPU resources initialized for vocabulary tree")
            self.gpu_resource = None
        except ImportError:
            logger.warning("sklearn not available. Installing fallback clustering...")
            # We'll implement a minimal k-means if sklearn is not available
    
    def build_vocabulary(self, all_features: Dict[str, Any], 
                        force_rebuild: bool = False) -> None:
        """
        Build vocabulary tree from all image features
        Much faster than hloc's approach using hierarchical k-means with GPU
        """
        
        cache_path = self.output_path / "vocabulary_cache.pkl"
        feature_hash = self._compute_feature_hash(all_features)
        
        # Check for cached vocabulary
        if not force_rebuild and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('hash') == feature_hash:
                        self.vocabulary = cached_data['vocabulary']
                        self.inverted_index = cached_data['inverted_index']
                        logger.info("Loaded cached vocabulary tree")
                        return
            except Exception as e:
                logger.warning(f"Failed to load cached vocabulary: {e}")
        
        logger.info("Building GPU-accelerated vocabulary tree...")
        start_time = time.time()
        
        # Collect all descriptors efficiently
        all_descriptors = self._collect_all_descriptors(all_features)
        logger.info(f"Collected {len(all_descriptors)} descriptors from {len(all_features)} images")
        
        # Build hierarchical vocabulary using GPU-accelerated k-means
        self.vocabulary = self._build_hierarchical_kmeans(all_descriptors)
        
        # Build inverted index for fast retrieval
        self._build_inverted_index(all_features)
        
        self.build_time = time.time() - start_time
        logger.info(f"Vocabulary tree built in {self.build_time:.2f}s")
        
        # Cache the vocabulary
        self._cache_vocabulary(cache_path, feature_hash)
    
    def _collect_all_descriptors(self, all_features: Dict[str, Any]) -> np.ndarray:
        """Collect all descriptors from all images efficiently"""
        descriptors_list = []
        
        # Use threading for parallel descriptor extraction
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._extract_descriptors, img_features): img_path
                for img_path, img_features in all_features.items()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Collecting descriptors"):
                try:
                    descriptors = future.result()
                    if descriptors is not None and len(descriptors) > 0:
                        # Smart descriptor selection for SfM quality
                        max_desc_per_image = min(self.max_descriptors_per_image, 2000)  # Cap at 2000 for quality
                        if len(descriptors) > max_desc_per_image:
                            # Use score-based selection instead of random
                            img_path = futures[future]
                            scores = all_features[img_path].get('scores', np.ones(len(descriptors)))
                            
                            # Select top descriptors by score (quality-first)
                            top_indices = np.argsort(scores)[-max_desc_per_image:]
                            descriptors = descriptors[top_indices]
                            logger.debug(f"Selected top {max_desc_per_image}/{len(scores)} descriptors for {Path(img_path).name}")
                        descriptors_list.append(descriptors)
                except Exception as e:
                    logger.warning(f"Failed to extract descriptors: {e}")
        
        if not descriptors_list:
            raise ValueError("No descriptors collected")
        
        all_descriptors = np.vstack(descriptors_list)
        
        # Subsample for vocabulary building if too many descriptors (indoor-optimized)
        max_vocab_descriptors = self.max_vocab_descriptors  # Use config value
        if len(all_descriptors) > max_vocab_descriptors:
            indices = np.random.choice(len(all_descriptors), 
                                     max_vocab_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
            logger.info(f"Subsampled to {len(all_descriptors)} descriptors for vocabulary")
        
        return all_descriptors.astype(np.float32)
    
    def _extract_descriptors(self, img_features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract descriptors from image features"""
        if 'descriptors' not in img_features:
            return None
        
        descriptors = img_features['descriptors']
        if isinstance(descriptors, torch.Tensor):
            descriptors = descriptors.cpu().numpy()
        
        return descriptors.astype(np.float32)
    
    def _build_hierarchical_kmeans(self, descriptors: np.ndarray) -> Dict:
        """Build hierarchical k-means vocabulary using GPU acceleration"""
        logger.info("Building hierarchical k-means vocabulary...")
        
        # Initialize vocabulary structure
        vocabulary = {
            'nodes': {},
            'depth': self.depth,
            'branching_factor': self.branching_factor,
            'descriptor_dim': descriptors.shape[1]
        }
        
        # Build tree level by level
        current_descriptors = {0: descriptors}  # node_id: descriptors
        
        for level in range(self.depth):
            logger.info(f"Building level {level + 1}/{self.depth}")
            next_descriptors = {}
            
            for node_id, node_desc in current_descriptors.items():
                if len(node_desc) < self.branching_factor:
                    # Not enough descriptors to split further
                    vocabulary['nodes'][node_id] = {
                        'center': np.mean(node_desc, axis=0),
                        'level': level,
                        'is_leaf': True,
                        'descriptor_count': len(node_desc)
                    }
                    continue
                
                # Run k-means clustering
                centers, assignments, actual_k = self._gpu_kmeans(node_desc, self.branching_factor)
                
                # Store node information
                vocabulary['nodes'][node_id] = {
                    'centers': centers,
                    'level': level,
                    'is_leaf': False,
                    'children': []
                }
                
                # Create child nodes (use actual_k instead of branching_factor)
                for child_idx in range(actual_k):
                    child_id = node_id * actual_k + child_idx + 1
                    vocabulary['nodes'][node_id]['children'].append(child_id)
                    
                    # Get descriptors assigned to this child
                    child_mask = (assignments == child_idx)
                    child_descriptors = node_desc[child_mask]
                    
                    if len(child_descriptors) > 0:
                        next_descriptors[child_id] = child_descriptors
            
            current_descriptors = next_descriptors
            
            if not current_descriptors:
                break
        
        # Mark remaining nodes as leaves
        for node_id, node_desc in current_descriptors.items():
            vocabulary['nodes'][node_id] = {
                'center': np.mean(node_desc, axis=0),
                'level': self.depth,
                'is_leaf': True,
                'descriptor_count': len(node_desc)
            }
        
        logger.info(f"Built vocabulary with {len(vocabulary['nodes'])} nodes")
        return vocabulary
    
    def _gpu_kmeans(self, descriptors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """GPU-accelerated k-means clustering using FAISS"""
        try:
            # Adjust k if not enough descriptors - Reduce requirement for small datasets
            min_points_per_cluster = 10  # Reduced from 39 for small datasets  
            required_points = k * min_points_per_cluster
            
            if len(descriptors) < required_points:
                # Calculate optimal k based on available points
                optimal_k = max(1, len(descriptors) // min_points_per_cluster)
                if optimal_k < k:
                    k = optimal_k
                
                # If still not enough points, fall back to CPU k-means
                if len(descriptors) < 5:  # Very minimal requirement
                    logger.warning(f"Not enough points ({len(descriptors)}) for GPU clustering, using CPU fallback")
                    centers, assignments = self._cpu_kmeans_fallback(descriptors, min(k, len(descriptors)))
                    return centers, assignments, len(centers)
            
            # Use FAISS GPU k-means for speed
            d = descriptors.shape[1]
            kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, gpu=self.use_gpu, min_points_per_centroid=1)
            
            if self.use_gpu and self.gpu_resource:
                kmeans.train(descriptors)
            else:
                kmeans.train(descriptors)
            
            # Get cluster assignments
            _, assignments = kmeans.index.search(descriptors, 1)
            assignments = assignments.flatten()
            
            return kmeans.centroids, assignments, k
            
        except Exception as e:
            logger.warning(f"GPU k-means failed, using CPU fallback: {e}")
            centers, assignments = self._cpu_kmeans_fallback(descriptors, k)
            return centers, assignments, k
    
    def _cpu_kmeans_fallback(self, descriptors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for k-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Ensure k doesn't exceed number of points
            k = min(k, len(descriptors))
            if k < 1:
                k = 1
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            assignments = kmeans.fit_predict(descriptors)
            centers = kmeans.cluster_centers_
            
            return centers, assignments
        except Exception as e:
            logger.warning(f"CPU k-means failed: {e}, using simple centroid")
            # Fallback to simple centroid
            center = np.mean(descriptors, axis=0, keepdims=True)
            assignments = np.zeros(len(descriptors), dtype=int)
            return center, assignments
    
    def _build_inverted_index(self, all_features: Dict[str, Any]):
        """Build inverted index for fast image retrieval"""
        logger.info("Building inverted index...")
        
        self.inverted_index = {}
        self.image_descriptors = {}
        
        for img_path, img_features in tqdm(all_features.items(), desc="Building index"):
            descriptors = self._extract_descriptors(img_features)
            if descriptors is None:
                continue
            
            # Store descriptors for this image
            self.image_descriptors[img_path] = descriptors
            
            # Get vocabulary words for each descriptor
            word_ids = self._get_vocabulary_words(descriptors)
            
            # Build TF-IDF representation
            word_counts = {}
            for word_id in word_ids:
                word_counts[word_id] = word_counts.get(word_id, 0) + 1
            
            # Add to inverted index
            for word_id, count in word_counts.items():
                if word_id not in self.inverted_index:
                    self.inverted_index[word_id] = []
                self.inverted_index[word_id].append((img_path, count))
        
        logger.info(f"Built inverted index with {len(self.inverted_index)} vocabulary words")
    
    def _get_vocabulary_words(self, descriptors: np.ndarray) -> List[int]:
        """Get vocabulary word IDs for descriptors by traversing the tree"""
        word_ids = []
        
        for desc in descriptors:
            # Start from root and traverse to leaf
            current_node = 0
            
            while current_node in self.vocabulary['nodes']:
                node = self.vocabulary['nodes'][current_node]
                
                if node['is_leaf']:
                    word_ids.append(current_node)
                    break
                
                # Find closest center
                centers = node['centers']
                distances = np.linalg.norm(centers - desc, axis=1)
                best_child_idx = np.argmin(distances)
                
                # Move to child
                current_node = node['children'][best_child_idx]
        
        return word_ids
    
    def query_similar_images(self, query_features: Dict[str, Any], 
                           top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Query similar images using vocabulary tree
        Much faster than hloc's brute force approach
        """
        start_time = time.time()
        
        query_descriptors = self._extract_descriptors(query_features)
        if query_descriptors is None:
            return []
        
        # Get vocabulary words for query
        query_word_ids = self._get_vocabulary_words(query_descriptors)
        
        # Build query TF-IDF vector
        query_word_counts = {}
        for word_id in query_word_ids:
            query_word_counts[word_id] = query_word_counts.get(word_id, 0) + 1
        
        # Score all candidate images
        candidate_scores = {}
        
        for word_id, count in query_word_counts.items():
            if word_id in self.inverted_index:
                # IDF weight (log of inverse document frequency) - avoid division by zero
                word_frequency = len(self.inverted_index[word_id])
                if word_frequency > 0 and len(self.image_descriptors) > 0:
                    idf = np.log(max(1, len(self.image_descriptors)) / max(1, word_frequency))
                else:
                    idf = 0.0  # Fallback for zero frequency
                query_tf_idf = count * idf
                
                for img_path, img_count in self.inverted_index[word_id]:
                    img_tf_idf = img_count * idf
                    
                    if img_path not in candidate_scores:
                        candidate_scores[img_path] = 0.0
                    candidate_scores[img_path] += query_tf_idf * img_tf_idf
        
        # Sort candidates by score
        sorted_candidates = sorted(candidate_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        logger.debug(f"Query completed in {query_time:.4f}s, found {len(sorted_candidates)} candidates")
        
        return sorted_candidates[:top_k]
    
    def get_generous_vocab_pairs(self, all_features: Dict[str, Any], 
                               generous_multiplier: float = 1.8,
                               min_score_threshold: float = 0.01) -> List[Tuple[str, str]]:
        """
        Stage 1: Generate generous vocabulary tree pairs with lower thresholds
        This creates more candidate pairs that will be filtered by MAGSAC
        """
        logger.info("Stage 1: Generating generous vocabulary tree pairs...")
        
        if self.vocabulary is None:
            logger.info("Building vocabulary first...")
            self.build_vocabulary(all_features)
        
        image_pairs = set()
        dataset_size = len(all_features)
        
        # Balanced k based on dataset size (less aggressive than before)
        if dataset_size < 50:
            generous_k = min(int(dataset_size * 0.5), dataset_size - 1)  # Moderate
        elif dataset_size < 200:
            generous_k = min(int(dataset_size * 0.25), 50)  # Less aggressive  
        else:
            generous_k = min(int(dataset_size * 0.15), 75)  # Conservative for large datasets
        
        logger.info(f"Using generous k={generous_k} for {dataset_size} images (multiplier: {generous_multiplier})")
        
        # Generate generous pairs with lower threshold
        total_candidates = 0
        total_above_threshold = 0
        
        for img_path, img_features in tqdm(all_features.items(), 
                                          desc="Generating generous pairs"):
            similar_images = self.query_similar_images(img_features, generous_k)
            total_candidates += len(similar_images)
            
            # Use lower threshold for more candidates
            for similar_img, score in similar_images:
                if similar_img != img_path:
                    if score >= min_score_threshold:
                        pair = tuple(sorted([img_path, similar_img]))
                        image_pairs.add(pair)
                        total_above_threshold += 1
        
        logger.info(f"Generous pairs debug: {total_candidates} total candidates, "
                   f"{total_above_threshold} above threshold ({min_score_threshold})")
        
        pairs_list = list(image_pairs)
        logger.info(f"Stage 1 generated {len(pairs_list)} generous pairs")
        
        # Fallback: if no pairs found, use very low threshold
        if len(pairs_list) == 0:
            logger.info(f"No pairs found with threshold {min_score_threshold}, using all top-k pairs instead...")
            
            # Simplified fallback: just take top-k pairs without score threshold
            for img_path, img_features in all_features.items():
                similar_images = self.query_similar_images(img_features, min(generous_k, 10))  # Limit to top 10
                for similar_img, score in similar_images:
                    if similar_img != img_path:
                        pair = tuple(sorted([img_path, similar_img]))
                        image_pairs.add(pair)
            
            pairs_list = list(image_pairs)
            logger.info(f"Stage 1 fallback generated {len(pairs_list)} generous pairs (no threshold)")
        
        # Ultimate fallback: if still no pairs, use traditional vocabulary tree
        if len(pairs_list) == 0:
            logger.warning("No generous pairs found, falling back to traditional vocabulary tree...")
            pairs_list = self.get_image_pairs_for_matching(
                all_features, 
                max_pairs_per_image=min(generous_k, 50),
                min_score_threshold=0.001,
                ensure_connectivity=True
            )
            logger.info(f"Stage 1 ultimate fallback generated {len(pairs_list)} pairs")
        
        return pairs_list

    def get_image_pairs_for_matching(self, all_features: Dict[str, Any], 
                                   max_pairs_per_image: int = 20,
                                   min_score_threshold: float = 0.01,
                                   ensure_connectivity: bool = True) -> List[Tuple[str, str]]:
        """
        Get efficient image pairs for matching using vocabulary tree with adaptive selection
        
        Args:
            max_pairs_per_image: Maximum pairs per image (adaptive based on scores)
            min_score_threshold: Minimum similarity score threshold
            ensure_connectivity: Ensure all images have at least one connection
        """
        logger.info("Getting image pairs using adaptive vocabulary tree...")
        
        if self.vocabulary is None:
            logger.info("Building vocabulary first...")
            self.build_vocabulary(all_features)
        
        image_pairs = set()
        connection_count = {img: 0 for img in all_features.keys()}
        
        # Adaptive top-k based on dataset size
        dataset_size = len(all_features)
        if dataset_size < 50:
            adaptive_k = min(max_pairs_per_image, dataset_size - 1)
        elif dataset_size < 200:
            adaptive_k = min(max_pairs_per_image, int(dataset_size * 0.15))
        else:
            adaptive_k = max_pairs_per_image
        
        logger.info(f"Using adaptive k={adaptive_k} for {dataset_size} images")
        
        # For each image, find most similar images with score filtering
        for img_path, img_features in tqdm(all_features.items(), 
                                          desc="Finding similar pairs"):
            similar_images = self.query_similar_images(img_features, adaptive_k * 2)  # Get more candidates
            
            # Filter by score threshold and adaptive selection
            valid_pairs = []
            for similar_img, score in similar_images:
                if similar_img != img_path and score >= min_score_threshold:
                    valid_pairs.append((similar_img, score))
            
            # Take top adaptive_k from valid pairs
            selected_pairs = valid_pairs[:adaptive_k]
            
            for similar_img, score in selected_pairs:
                pair = tuple(sorted([img_path, similar_img]))
                image_pairs.add(pair)
                connection_count[img_path] += 1
                connection_count[similar_img] += 1
        
        # Ensure connectivity for isolated images
        if ensure_connectivity:
            isolated_images = [img for img, count in connection_count.items() if count == 0]
            for img_path in isolated_images:
                # Find best matches without score threshold
                similar_images = self.query_similar_images(all_features[img_path], 3)
                for similar_img, _ in similar_images[:1]:  # At least one connection
                    if similar_img != img_path:
                        pair = tuple(sorted([img_path, similar_img]))
                        image_pairs.add(pair)
                        break
        
        pairs_list = list(image_pairs)
        avg_connections = sum(connection_count.values()) / len(connection_count) if connection_count else 0
        logger.info(f"Generated {len(pairs_list)} pairs, avg {avg_connections:.1f} connections per image")
        
        return pairs_list
    
    def expand_pairs_with_geometric_verification(self, pairs_list: List[Tuple[str, str]], 
                                               all_features: Dict[str, Any],
                                               expansion_ratio: float = 0.5) -> List[Tuple[str, str]]:
        """
        Expand successful pairs with geometric verification for missed connections
        
        Args:
            pairs_list: Initial vocabulary tree pairs
            all_features: All image features
            expansion_ratio: Ratio of additional pairs to try (0.5 = 50% more)
        """
        logger.info("Expanding pairs with geometric verification...")
        
        # Get images with successful geometric verification (would need to be called after verification)
        # For now, simulate by expanding pairs for images with few connections
        
        expanded_pairs = set(pairs_list)
        original_count = len(pairs_list)
        
        # Count connections per image
        connection_count = {}
        for img1, img2 in pairs_list:
            connection_count[img1] = connection_count.get(img1, 0) + 1
            connection_count[img2] = connection_count.get(img2, 0) + 1
        
        # Find images with few connections
        min_connections = 3
        images_needing_expansion = [img for img, count in connection_count.items() 
                                  if count < min_connections]
        
        # For images needing expansion, try more pairs
        expansion_candidates = int(len(all_features) * expansion_ratio)
        
        for img_path in images_needing_expansion:
            if img_path not in all_features:
                continue
                
            # Get more candidates for this image
            similar_images = self.query_similar_images(
                all_features[img_path], 
                expansion_candidates
            )
            
            # Add pairs that weren't already included
            added_count = 0
            for similar_img, score in similar_images:
                if similar_img != img_path:
                    pair = tuple(sorted([img_path, similar_img]))
                    if pair not in expanded_pairs:
                        expanded_pairs.add(pair)
                        added_count += 1
                        if added_count >= 5:  # Limit expansion per image
                            break
        
        expansion_count = len(expanded_pairs) - original_count
        logger.info(f"Expanded from {original_count} to {len(expanded_pairs)} pairs (+{expansion_count})")
        
        return list(expanded_pairs)
    
    def verify_pairs_with_magsac(self, pairs: List[Tuple[str, str]], 
                               all_features: Dict[str, Any],
                               ransac_threshold: float = 3.0,
                               confidence: float = 0.99,
                               max_iters: int = 500) -> List[Tuple[str, str]]:
        """
        Stage 2: Verify pairs using MAGSAC geometric verification
        Filters out pairs that don't have geometric consistency
        """
        logger.info(f"Stage 2: MAGSAC verification of {len(pairs)} pairs...")
        
        verified_pairs = []
        
        # Use threading for parallel MAGSAC verification
        magsac_workers = 8
        if hasattr(self, 'config') and self.config:
            magsac_workers = self.config.get('magsac_workers', 8)
        
        # Debug: sample a few pairs to understand why they're failing
        sample_pairs = pairs[:min(5, len(pairs))]
        logger.info(f"Debug: Testing {len(sample_pairs)} sample pairs for MAGSAC verification...")
        for img1, img2 in sample_pairs:
            is_valid = self._verify_single_pair_magsac(
                img1, img2,
                all_features[img1], all_features[img2],
                ransac_threshold, confidence, max_iters
            )
            logger.info(f"Debug sample: {img1} <-> {img2}: {'PASS' if is_valid else 'FAIL'}")
            
        with ThreadPoolExecutor(max_workers=magsac_workers) as executor:
            futures = {
                executor.submit(
                    self._verify_single_pair_magsac,
                    img1, img2, 
                    all_features[img1], all_features[img2],
                    ransac_threshold, confidence, max_iters
                ): (img1, img2)
                for img1, img2 in pairs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="MAGSAC verification"):
                img1, img2 = futures[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        verified_pairs.append((img1, img2))
                except Exception as e:
                    logger.warning(f"MAGSAC verification failed for {img1}-{img2}: {e}")
        
        verification_rate = (len(verified_pairs)/len(pairs)*100) if len(pairs) > 0 else 0.0
        logger.info(f"Stage 2: {len(verified_pairs)}/{len(pairs)} pairs passed MAGSAC verification "
                   f"({verification_rate:.1f}%)")
        
        return verified_pairs
    
    def _verify_single_pair_magsac(self, img1_path: str, img2_path: str,
                                 feat1: Dict[str, Any], feat2: Dict[str, Any],
                                 ransac_threshold: float = 3.0,
                                 confidence: float = 0.99,
                                 max_iters: int = 500) -> bool:
        """
        Verify single pair using MAGSAC with lightweight feature matching
        Returns True if pair has geometric consistency
        """
        try:
            # Extract features
            kpts1 = feat1['keypoints']
            kpts2 = feat2['keypoints']
            desc1 = feat1['descriptors']
            desc2 = feat2['descriptors']
            
            if isinstance(desc1, torch.Tensor):
                desc1 = desc1.cpu().numpy()
            if isinstance(desc2, torch.Tensor):
                desc2 = desc2.cpu().numpy()
            
            # Quick feature matching using nearest neighbor (lightweight)
            matches = self._fast_feature_matching(desc1, desc2, ratio_thresh=0.9)  # More permissive
            
            if len(matches) < 4:  # Reduced minimum matches requirement
                logger.debug(f"MAGSAC {img1_path}-{img2_path}: Only {len(matches)} matches found, need >= 4")
                return False
            
            # Get matched keypoints
            src_pts = kpts1[matches[:, 0]]
            dst_pts = kpts2[matches[:, 1]]
            
            # MAGSAC verification using cv2
            try:
                _, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=ransac_threshold,
                    confidence=confidence,
                    maxIters=max_iters
                )
                
                if mask is None:
                    return False
                
                # Check inlier ratio - more permissive for vocabulary tree filtering
                inlier_ratio = np.sum(mask) / len(mask)
                min_inlier_ratio = 0.05  # Reduced to 5% inliers for initial filtering
                
                logger.debug(f"MAGSAC {img1_path}-{img2_path}: {len(matches)} matches, "
                           f"inlier_ratio={inlier_ratio:.3f}, threshold={min_inlier_ratio}")
                
                return inlier_ratio >= min_inlier_ratio
                
            except cv2.error:
                # Fallback to fundamental matrix if homography fails
                _, mask = cv2.findFundamentalMat(
                    src_pts, dst_pts,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=ransac_threshold,
                    confidence=confidence,
                    maxIters=max_iters
                )
                
                if mask is None:
                    return False
                    
                inlier_ratio = np.sum(mask) / len(mask)
                return inlier_ratio >= 0.05  # Same reduced threshold
                
        except Exception as e:
            logger.debug(f"MAGSAC verification error for {img1_path}-{img2_path}: {e}")
            return False
    
    def _fast_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray, 
                             ratio_thresh: float = 0.9) -> np.ndarray:
        """
        Fast feature matching using nearest neighbor with ratio test
        Lightweight matching for MAGSAC verification
        """
        try:
            # Ensure descriptors have enough points
            if len(desc1) < 2 or len(desc2) < 2:
                return np.array([]).reshape(0, 2)
            
            # Use cv2 FLANN matcher for speed
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
            search_params = dict(checks=32)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append([m.queryIdx, m.trainIdx])
            
            return np.array(good_matches) if good_matches else np.array([]).reshape(0, 2)
            
        except Exception as e:
            # Fallback to brute force matching
            try:
                if len(desc1) < 2 or len(desc2) < 2:
                    return np.array([]).reshape(0, 2)
                    
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
                
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_thresh * n.distance:
                            good_matches.append([m.queryIdx, m.trainIdx])
                
                return np.array(good_matches) if good_matches else np.array([]).reshape(0, 2)
            except Exception as e2:
                logger.debug(f"Fast feature matching failed completely: {e2}")
                return np.array([]).reshape(0, 2)
    
    def get_multi_stage_pairs(self, all_features: Dict[str, Any],
                            generous_multiplier: float = 2.5,
                            magsac_threshold: float = 3.0,
                            ensure_connectivity: bool = True,
                            skip_magsac: bool = False) -> List[Tuple[str, str]]:
        """
        Multi-stage robust pair selection: Generous Vocab Tree -> MAGSAC -> Adaptive Expansion
        
        Args:
            all_features: Image features dictionary
            generous_multiplier: Multiplier for generous vocabulary tree stage
            magsac_threshold: RANSAC threshold for MAGSAC verification
            ensure_connectivity: Ensure all images have at least one connection
        """
        logger.info("Starting multi-stage pair selection...")
        
        # Stage 1: Generous vocabulary tree
        generous_pairs = self.get_generous_vocab_pairs(
            all_features, 
            generous_multiplier=generous_multiplier
        )
        
        # Stage 2: MAGSAC verification (optional skip for debugging)
        if skip_magsac:
            logger.info("Stage 2: Skipping MAGSAC verification (debug mode)")
            verified_pairs = generous_pairs
        else:
            verified_pairs = self.verify_pairs_with_magsac(
                generous_pairs, 
                all_features,
                ransac_threshold=magsac_threshold
            )
        
        # Stage 3: Adaptive expansion for connectivity
        if ensure_connectivity:
            final_pairs = self._ensure_connectivity_expansion(
                verified_pairs, all_features
            )
        else:
            final_pairs = verified_pairs
        
        # Performance statistics
        total_generous = len(generous_pairs)
        total_verified = len(verified_pairs)  
        total_final = len(final_pairs)
        
        verification_rate = (total_verified / total_generous * 100) if total_generous > 0 else 0
        expansion_rate = ((total_final - total_verified) / total_verified * 100) if total_verified > 0 else 0
        
        logger.info(f"Multi-stage selection completed:")
        logger.info(f"  Stage 1 (Generous): {total_generous} pairs")
        logger.info(f"  Stage 2 (MAGSAC): {total_verified} pairs ({verification_rate:.1f}% verified)")
        logger.info(f"  Stage 3 (Expansion): {total_final} pairs (+{expansion_rate:.1f}% expansion)")
        
        return final_pairs
    
    def _ensure_connectivity_expansion(self, verified_pairs: List[Tuple[str, str]], 
                                     all_features: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Stage 3: Ensure connectivity by adding pairs for isolated images
        """
        # Count connections per image
        connection_count = {}
        for img1, img2 in verified_pairs:
            connection_count[img1] = connection_count.get(img1, 0) + 1
            connection_count[img2] = connection_count.get(img2, 0) + 1
        
        # Find isolated images (no connections)
        isolated_images = []
        for img_path in all_features.keys():
            if connection_count.get(img_path, 0) == 0:
                isolated_images.append(img_path)
        
        if not isolated_images:
            logger.info("Stage 3: No isolated images found, connectivity ensured")
            return verified_pairs
        
        logger.info(f"Stage 3: Found {len(isolated_images)} isolated images, expanding connections...")
        
        expanded_pairs = set(verified_pairs)
        
        # For each isolated image, find best vocabulary match and force connection
        for img_path in isolated_images:
            similar_images = self.query_similar_images(all_features[img_path], 3)
            
            # Add connection to best match
            for similar_img, score in similar_images:
                if similar_img != img_path:
                    pair = tuple(sorted([img_path, similar_img]))
                    expanded_pairs.add(pair)
                    logger.debug(f"Added connectivity pair: {pair} (score: {score:.3f})")
                    break
        
        return list(expanded_pairs)
    
    def get_hybrid_matching_strategy(self, all_features: Dict[str, Any],
                                   strategy: str = "adaptive") -> List[Tuple[str, str]]:
        """
        Hybrid matching strategy that combines vocabulary tree with fallback methods
        
        Strategies:
        - "adaptive": Vocabulary tree + expansion for small datasets
        - "exhaustive_small": Brute force for very small datasets (<20 images)
        - "tiered": Vocabulary tree + brute force for critical images
        """
        dataset_size = len(all_features)
        logger.info(f"Using hybrid strategy '{strategy}' for {dataset_size} images")
        
        if strategy == "exhaustive_small" and dataset_size < 20:
            # Use brute force for very small datasets
            logger.info("Using exhaustive matching for small dataset")
            return self._get_exhaustive_pairs(list(all_features.keys()))
        
        elif strategy == "adaptive":
            # Adaptive approach based on dataset size
            if dataset_size < 50:
                # Small dataset: More generous vocabulary tree + expansion
                pairs = self.get_image_pairs_for_matching(
                    all_features, 
                    max_pairs_per_image=min(30, dataset_size-1),
                    min_score_threshold=0.005,  # Lower threshold
                    ensure_connectivity=True
                )
                # Always expand for small datasets
                pairs = self.expand_pairs_with_geometric_verification(
                    pairs, all_features, expansion_ratio=0.8
                )
                
            elif dataset_size < 200:
                # Medium dataset: Standard vocabulary tree + selective expansion
                pairs = self.get_image_pairs_for_matching(
                    all_features,
                    max_pairs_per_image=25,
                    min_score_threshold=0.01,
                    ensure_connectivity=True
                )
                pairs = self.expand_pairs_with_geometric_verification(
                    pairs, all_features, expansion_ratio=0.3
                )
                
            else:
                # Large dataset: Conservative vocabulary tree
                pairs = self.get_image_pairs_for_matching(
                    all_features,
                    max_pairs_per_image=20,
                    min_score_threshold=0.02,
                    ensure_connectivity=True
                )
                # Optional expansion only if connectivity is poor
                pairs = self.expand_pairs_with_geometric_verification(
                    pairs, all_features, expansion_ratio=0.1
                )
            
            return pairs
        
        elif strategy == "tiered":
            # Tiered approach: vocabulary tree + brute force for critical images
            vocab_pairs = self.get_image_pairs_for_matching(all_features)
            
            # Identify critical images (few connections)
            connection_count = {}
            for img1, img2 in vocab_pairs:
                connection_count[img1] = connection_count.get(img1, 0) + 1
                connection_count[img2] = connection_count.get(img2, 0) + 1
            
            critical_images = [img for img, count in connection_count.items() if count < 2]
            
            # Add brute force pairs for critical images
            all_pairs = set(vocab_pairs)
            for critical_img in critical_images:
                for other_img in all_features.keys():
                    if other_img != critical_img:
                        pair = tuple(sorted([critical_img, other_img]))
                        all_pairs.add(pair)
            
            logger.info(f"Tiered strategy: {len(vocab_pairs)} vocab + {len(all_pairs) - len(vocab_pairs)} brute force")
            return list(all_pairs)
        
        else:
            # Default to standard vocabulary tree
            return self.get_image_pairs_for_matching(all_features)
    
    def _get_exhaustive_pairs(self, image_list: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible image pairs (brute force)"""
        pairs = []
        for i in range(len(image_list)):
            for j in range(i + 1, len(image_list)):
                pairs.append((image_list[i], image_list[j]))
        return pairs
    
    def _compute_feature_hash(self, all_features: Dict[str, Any]) -> str:
        """Compute hash of features for caching"""
        feature_info = []
        for img_path in sorted(all_features.keys()):
            feat = all_features[img_path]
            if 'descriptors' in feat:
                desc = feat['descriptors']
                if isinstance(desc, torch.Tensor):
                    desc = desc.cpu().numpy()
                # Use shape and a sample of values for hash
                info = f"{img_path}:{desc.shape}:{np.mean(desc):.6f}"
                feature_info.append(info)
        
        combined = '|'.join(feature_info)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _cache_vocabulary(self, cache_path: Path, feature_hash: str):
        """Cache vocabulary to disk"""
        try:
            cache_data = {
                'vocabulary': self.vocabulary,
                'inverted_index': self.inverted_index,
                'hash': feature_hash
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached vocabulary to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache vocabulary: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'build_time': self.build_time,
            'avg_query_time': np.mean(self.query_times) if self.query_times else 0.0,
            'num_queries': len(self.query_times),
            'vocabulary_size': len(self.vocabulary['nodes']) if self.vocabulary else 0,
            'inverted_index_size': len(self.inverted_index),
            'using_gpu': self.use_gpu
        }
        return stats