"""
Distributed processing for large-scale SfM datasets
Supports multi-node processing with load balancing and fault tolerance
"""

import os
import sys
import time
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing import Manager, Queue, Lock
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed processing"""
    
    # Node configuration
    num_nodes: int = 1
    node_id: int = 0
    num_workers_per_node: int = 4
    
    # Communication
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = "nccl"  # or "gloo" for CPU
    
    # Data distribution
    batch_size: int = 32
    chunk_size: int = 100  # Images per chunk
    
    # Fault tolerance
    max_retries: int = 3
    timeout: int = 300  # seconds
    
    # Storage
    shared_storage: str = "/tmp/sfm_distributed"
    checkpoint_interval: int = 50  # chunks


class DistributedProcessor:
    """Distributed processing for large-scale SfM"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.task_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.lock = self.manager.Lock()
        
        # Initialize distributed environment
        self._setup_distributed()
        
    def _setup_distributed(self):
        """Setup distributed processing environment"""
        
        if self.config.num_nodes > 1:
            # Setup distributed training
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = str(self.config.master_port)
            os.environ['WORLD_SIZE'] = str(self.config.num_nodes)
            os.environ['RANK'] = str(self.config.node_id)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.config.node_id,
                world_size=self.config.num_nodes
            )
            
            logger.info(f"Initialized distributed processing: node {self.config.node_id}/{self.config.num_nodes}")
    
    def distribute_images(self, image_paths: List[str]) -> List[List[str]]:
        """Distribute images across nodes and workers"""
        
        # Create chunks of images
        chunks = []
        for i in range(0, len(image_paths), self.config.chunk_size):
            chunk = image_paths[i:i + self.config.chunk_size]
            chunks.append(chunk)
        
        # Distribute chunks across nodes
        node_chunks = self._distribute_chunks_across_nodes(chunks)
        
        logger.info(f"Distributed {len(image_paths)} images into {len(chunks)} chunks across {self.config.num_nodes} nodes")
        
        return node_chunks
    
    def _distribute_chunks_across_nodes(self, chunks: List[List[str]]) -> List[List[str]]:
        """Distribute chunks across nodes using round-robin"""
        
        node_chunks = [[] for _ in range(self.config.num_nodes)]
        
        for i, chunk in enumerate(chunks):
            node_id = i % self.config.num_nodes
            node_chunks[node_id].extend(chunk)
        
        return node_chunks
    
    def process_feature_extraction(self, image_paths: List[str], 
                                 feature_extractor_config: Dict) -> Dict[str, Any]:
        """Distributed feature extraction"""
        
        logger.info(f"Starting distributed feature extraction for {len(image_paths)} images")
        
        # Split images into batches
        batches = self._create_batches(image_paths, self.config.batch_size)
        
        # Process batches in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=self.config.num_workers_per_node) as executor:
            # Submit tasks
            future_to_batch = {
                executor.submit(self._extract_features_batch, batch, feature_extractor_config): batch 
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout)
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"Feature extraction failed for batch: {e}")
                    # Retry failed batch
                    self._retry_batch(batch, feature_extractor_config, results)
        
        logger.info(f"Completed feature extraction: {len(results)} images processed")
        return results
    
    def _extract_features_batch(self, image_batch: List[str], 
                              config: Dict) -> Dict[str, Any]:
        """Extract features for a batch of images"""
        
        # Import here to avoid pickling issues
        from sfm.core.feature_extractor import FeatureExtractorFactory
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = FeatureExtractorFactory.create(
            config['feature_extractor'],
            device=device,
            max_keypoints=config.get('max_keypoints', 2048)
        )
        
        results = {}
        for img_path in image_batch:
            try:
                features = extractor.extract_features([img_path])
                if img_path in features:
                    results[img_path] = features[img_path]
            except Exception as e:
                logger.warning(f"Failed to extract features for {img_path}: {e}")
        
        return results
    
    def _retry_batch(self, batch: List[str], config: Dict, results: Dict):
        """Retry failed batch with exponential backoff"""
        
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(2 ** attempt)  # Exponential backoff
                batch_results = self._extract_features_batch(batch, config)
                results.update(batch_results)
                break
            except Exception as e:
                logger.error(f"Retry {attempt + 1} failed: {e}")
    
    def process_feature_matching(self, features: Dict[str, Any], 
                               image_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """Distributed feature matching"""
        
        logger.info(f"Starting distributed feature matching for {len(image_pairs)} pairs")
        
        # Split pairs into batches
        pair_batches = self._create_batches(image_pairs, self.config.batch_size)
        
        # Process batches in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=self.config.num_workers_per_node) as executor:
            # Submit tasks
            future_to_batch = {
                executor.submit(self._match_features_batch, batch, features): batch 
                for batch in pair_batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout)
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"Feature matching failed for batch: {e}")
                    # Retry failed batch
                    self._retry_matching_batch(batch, features, results)
        
        logger.info(f"Completed feature matching: {len(results)} pairs processed")
        return results
    
    def _match_features_batch(self, pair_batch: List[Tuple[str, str]], 
                            features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """Match features for a batch of image pairs"""
        
        # Import here to avoid pickling issues
        from sfm.core.feature_matcher import LightGlueMatcher
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matcher = LightGlueMatcher(device=device)
        
        results = {}
        for img1, img2 in pair_batch:
            try:
                if img1 in features and img2 in features:
                    match_result = matcher.match_features(features[img1], features[img2])
                    if match_result is not None:
                        results[(img1, img2)] = match_result
            except Exception as e:
                logger.warning(f"Failed to match features for {img1}-{img2}: {e}")
        
        return results
    
    def _retry_matching_batch(self, batch: List[Tuple[str, str]], 
                            features: Dict[str, Any], results: Dict):
        """Retry failed matching batch"""
        
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(2 ** attempt)
                batch_results = self._match_features_batch(batch, features)
                results.update(batch_results)
                break
            except Exception as e:
                logger.error(f"Retry {attempt + 1} failed: {e}")
    
    def process_depth_estimation(self, image_paths: List[str], 
                               depth_config: Dict) -> Dict[str, np.ndarray]:
        """Distributed depth estimation"""
        
        logger.info(f"Starting distributed depth estimation for {len(image_paths)} images")
        
        # Split images into batches
        batches = self._create_batches(image_paths, self.config.batch_size)
        
        # Process batches in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=self.config.num_workers_per_node) as executor:
            # Submit tasks
            future_to_batch = {
                executor.submit(self._estimate_depth_batch, batch, depth_config): batch 
                for batch in batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result(timeout=self.config.timeout)
                    results.update(batch_results)
                except Exception as e:
                    logger.error(f"Depth estimation failed for batch: {e}")
                    # Retry failed batch
                    self._retry_depth_batch(batch, depth_config, results)
        
        logger.info(f"Completed depth estimation: {len(results)} images processed")
        return results
    
    def _estimate_depth_batch(self, image_batch: List[str], 
                            config: Dict) -> Dict[str, np.ndarray]:
        """Estimate depth for a batch of images"""
        
        # Import here to avoid pickling issues
        from sfm.core.dense_depth import DenseDepthEstimator
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        estimator = DenseDepthEstimator(device=device)
        
        results = {}
        for img_path in image_batch:
            try:
                # This is a simplified call - in practice you'd need sparse points, cameras, etc.
                # For now, just return a placeholder
                depth_map = np.random.rand(600, 800)  # Placeholder
                results[img_path] = depth_map
            except Exception as e:
                logger.warning(f"Failed to estimate depth for {img_path}: {e}")
        
        return results
    
    def _retry_depth_batch(self, batch: List[str], config: Dict, results: Dict):
        """Retry failed depth estimation batch"""
        
        for attempt in range(self.config.max_retries):
            try:
                time.sleep(2 ** attempt)
                batch_results = self._estimate_depth_batch(batch, config)
                results.update(batch_results)
                break
            except Exception as e:
                logger.error(f"Retry {attempt + 1} failed: {e}")
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Create batches from a list of items"""
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def save_checkpoint(self, data: Dict[str, Any], checkpoint_id: int):
        """Save checkpoint for fault tolerance"""
        
        checkpoint_dir = Path(self.config.shared_storage) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved checkpoint {checkpoint_id} to {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_id: int) -> Optional[Dict[str, Any]]:
        """Load checkpoint for fault tolerance"""
        
        checkpoint_file = Path(self.config.shared_storage) / "checkpoints" / f"checkpoint_{checkpoint_id}.pkl"
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded checkpoint {checkpoint_id} from {checkpoint_file}")
            return data
        
        return None
    
    def cleanup(self):
        """Cleanup distributed resources"""
        
        if self.config.num_nodes > 1:
            dist.destroy_process_group()
        
        logger.info("Cleaned up distributed resources")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        stats = {
            'node_id': self.config.node_id,
            'num_workers': self.config.num_workers_per_node,
            'batch_size': self.config.batch_size,
            'chunk_size': self.config.chunk_size,
            'max_retries': self.config.max_retries,
            'timeout': self.config.timeout
        }
        
        return stats


class DistributedSfMPipeline:
    """Distributed SfM pipeline for large-scale datasets"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.processor = DistributedProcessor(config)
        
    def run_distributed_pipeline(self, image_paths: List[str], 
                               pipeline_config: Dict) -> Dict[str, Any]:
        """Run complete distributed SfM pipeline"""
        
        logger.info("Starting distributed SfM pipeline")
        
        # Step 1: Distribute images across nodes
        node_chunks = self.processor.distribute_images(image_paths)
        
        # Step 2: Distributed feature extraction
        all_features = {}
        for node_id, node_images in enumerate(node_chunks):
            if node_id == self.config.node_id:
                logger.info(f"Processing {len(node_images)} images on node {node_id}")
                features = self.processor.process_feature_extraction(
                    node_images, pipeline_config.get('feature_extraction', {})
                )
                all_features.update(features)
        
        # Step 3: Distributed feature matching
        image_pairs = self._generate_image_pairs(image_paths)
        matches = self.processor.process_feature_matching(all_features, image_pairs)
        
        # Step 4: Distributed depth estimation (if enabled)
        depth_maps = {}
        if pipeline_config.get('use_monocular_depth', False):
            depth_maps = self.processor.process_depth_estimation(
                image_paths, pipeline_config.get('depth_estimation', {})
            )
        
        # Step 5: Centralized reconstruction (on master node)
        reconstruction = {}
        if self.config.node_id == 0:
            reconstruction = self._run_centralized_reconstruction(
                all_features, matches, depth_maps, pipeline_config
            )
        
        # Step 6: Distribute results back to all nodes
        if self.config.num_nodes > 1:
            reconstruction = self._distribute_reconstruction_results(reconstruction)
        
        logger.info("Completed distributed SfM pipeline")
        
        return {
            'features': all_features,
            'matches': matches,
            'depth_maps': depth_maps,
            'reconstruction': reconstruction,
            'performance_stats': self.processor.get_performance_stats()
        }
    
    def _generate_image_pairs(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        """Generate image pairs for matching"""
        
        pairs = []
        for i, img1 in enumerate(image_paths):
            for img2 in image_paths[i+1:]:
                pairs.append((img1, img2))
        
        return pairs
    
    def _run_centralized_reconstruction(self, features: Dict[str, Any], 
                                      matches: Dict[Tuple[str, str], Any],
                                      depth_maps: Dict[str, np.ndarray],
                                      config: Dict) -> Dict[str, Any]:
        """Run centralized reconstruction on master node"""
        
        # Import here to avoid pickling issues
        from sfm.core.reconstruction import IncrementalSfM
        from sfm.core.gpu_bundle_adjustment import GPUBundleAdjustment
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Run incremental SfM
        reconstruction = IncrementalSfM(device=device)
        sparse_points, cameras, images = reconstruction.reconstruct(
            features=features,
            matches=matches,
            image_paths=list(features.keys())
        )
        
        # Run bundle adjustment if enabled
        if config.get('use_gpu_ba', False):
            gpu_ba = GPUBundleAdjustment(device=device)
            sparse_points, cameras, images = gpu_ba.optimize(
                cameras, images, sparse_points, matches
            )
        
        return {
            'sparse_points': sparse_points,
            'cameras': cameras,
            'images': images,
            'depth_maps': depth_maps
        }
    
    def _distribute_reconstruction_results(self, reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute reconstruction results to all nodes"""
        
        # This would use torch.distributed to broadcast results
        # For now, return the reconstruction as-is
        return reconstruction
    
    def cleanup(self):
        """Cleanup distributed pipeline"""
        
        self.processor.cleanup() 