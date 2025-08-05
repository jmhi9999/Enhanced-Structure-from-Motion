#!/usr/bin/env python3
"""
Distributed 3D Gaussian Splatting SfM Pipeline Runner
For large-scale datasets across multiple nodes
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

from sfm.core.distributed_processor import DistributedConfig, DistributedSfMPipeline


def setup_logging():
    """Setup logging for distributed pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_image_paths(input_dir: str) -> List[str]:
    """Load image paths from directory"""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_paths = []
    for img_file in input_path.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in image_extensions:
            image_paths.append(str(img_file))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    logging.info(f"Found {len(image_paths)} images in {input_dir}")
    return image_paths


def create_distributed_config(args) -> DistributedConfig:
    """Create distributed configuration from arguments"""
    
    return DistributedConfig(
        num_nodes=args.num_nodes,
        node_id=args.node_id,
        num_workers_per_node=args.num_workers,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        timeout=args.timeout,
        shared_storage=args.shared_storage
    )


def create_pipeline_config(args) -> Dict[str, Any]:
    """Create pipeline configuration from arguments"""
    
    return {
        'feature_extraction': {
            'feature_extractor': args.feature_extractor,
            'max_keypoints': args.max_keypoints,
            'max_image_size': args.max_image_size
        },
        'matching': {
            'matcher': 'lightglue',
            'use_vocab_tree': args.use_vocab_tree,
            'max_pairs_per_image': args.max_pairs_per_image
        },
        'depth_estimation': {
            'depth_model': args.depth_model,
            'fusion_weight': args.fusion_weight,
            'bilateral_filter': args.bilateral_filter
        },
        'reconstruction': {
            'use_gpu_ba': args.use_gpu_ba,
            'ba_max_iterations': args.ba_max_iterations,
            'scale_recovery': args.scale_recovery
        },
        'use_monocular_depth': args.use_monocular_depth,
        'use_gpu_ba': args.use_gpu_ba,
        'scale_recovery': args.scale_recovery
    }


def save_results(results: Dict[str, Any], output_dir: str, node_id: int):
    """Save distributed pipeline results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save node-specific results
    node_output = output_path / f"node_{node_id}"
    node_output.mkdir(exist_ok=True)
    
    # Save features
    if 'features' in results:
        import pickle
        with open(node_output / "features.pkl", 'wb') as f:
            pickle.dump(results['features'], f)
    
    # Save matches
    if 'matches' in results:
        import pickle
        with open(node_output / "matches.pkl", 'wb') as f:
            pickle.dump(results['matches'], f)
    
    # Save depth maps
    if 'depth_maps' in results:
        depth_dir = node_output / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        
        import numpy as np
        for img_path, depth_map in results['depth_maps'].items():
            if depth_map is not None:
                img_name = Path(img_path).stem
                np.save(depth_dir / f"{img_name}_depth.npy", depth_map)
    
    # Save reconstruction (only on master node)
    if node_id == 0 and 'reconstruction' in results:
        reconstruction_dir = output_path / "reconstruction"
        reconstruction_dir.mkdir(exist_ok=True)
        
        import pickle
        with open(reconstruction_dir / "reconstruction.pkl", 'wb') as f:
            pickle.dump(results['reconstruction'], f)
    
    # Save performance stats
    if 'performance_stats' in results:
        with open(node_output / "performance_stats.json", 'w') as f:
            json.dump(results['performance_stats'], f, indent=2)
    
    logging.info(f"Saved results for node {node_id} to {node_output}")


def main():
    """Main distributed pipeline runner"""
    
    parser = argparse.ArgumentParser(description="Distributed 3DGS SfM Pipeline")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    
    # Distributed configuration
    parser.add_argument("--num_nodes", type=int, default=1,
                       help="Number of nodes in cluster")
    parser.add_argument("--node_id", type=int, default=0,
                       help="ID of current node (0-based)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers per node")
    parser.add_argument("--master_addr", type=str, default="localhost",
                       help="Master node address")
    parser.add_argument("--master_port", type=int, default=29500,
                       help="Master node port")
    parser.add_argument("--backend", type=str, default="nccl",
                       choices=["nccl", "gloo"],
                       help="Distributed backend")
    
    # Data distribution
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--chunk_size", type=int, default=100,
                       help="Images per chunk")
    
    # Fault tolerance
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum retries for failed tasks")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout for tasks (seconds)")
    parser.add_argument("--shared_storage", type=str, default="/tmp/sfm_distributed",
                       help="Shared storage path")
    
    # Pipeline configuration
    parser.add_argument("--feature_extractor", type=str, default="superpoint",
                       choices=["superpoint", "aliked", "disk"],
                       help="Feature extractor to use")
    parser.add_argument("--max_keypoints", type=int, default=2048,
                       help="Maximum keypoints per image")
    parser.add_argument("--max_image_size", type=int, default=1600,
                       help="Maximum image size")
    parser.add_argument("--use_vocab_tree", action="store_true",
                       help="Use vocabulary tree for pair selection")
    parser.add_argument("--max_pairs_per_image", type=int, default=20,
                       help="Maximum pairs per image")
    parser.add_argument("--use_monocular_depth", action="store_true",
                       help="Use monocular depth estimation")
    parser.add_argument("--depth_model", type=str, default="dpt-large",
                       help="Depth estimation model")
    parser.add_argument("--fusion_weight", type=float, default=0.7,
                       help="SfM vs monocular depth fusion weight")
    parser.add_argument("--bilateral_filter", action="store_true",
                       help="Apply bilateral filtering to depth maps")
    parser.add_argument("--use_gpu_ba", action="store_true",
                       help="Use GPU bundle adjustment")
    parser.add_argument("--ba_max_iterations", type=int, default=200,
                       help="Bundle adjustment max iterations")
    parser.add_argument("--scale_recovery", action="store_true",
                       help="Enable scale recovery")
    
    # Performance
    parser.add_argument("--profile", action="store_true",
                       help="Enable performance profiling")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("DISTRIBUTED 3D GAUSSIAN SPLATTING SFM PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Node ID: {args.node_id}/{args.num_nodes}")
    logging.info(f"Input Directory: {args.input_dir}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Workers per Node: {args.num_workers}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Chunk Size: {args.chunk_size}")
    
    try:
        # Load image paths
        image_paths = load_image_paths(args.input_dir)
        
        # Create distributed configuration
        dist_config = create_distributed_config(args)
        
        # Create pipeline configuration
        pipeline_config = create_pipeline_config(args)
        
        # Create distributed pipeline
        pipeline = DistributedSfMPipeline(dist_config)
        
        # Run distributed pipeline
        start_time = time.time()
        
        results = pipeline.run_distributed_pipeline(image_paths, pipeline_config)
        
        pipeline_time = time.time() - start_time
        
        # Save results
        save_results(results, args.output_dir, args.node_id)
        
        # Print performance summary
        logging.info("=" * 80)
        logging.info("DISTRIBUTED PIPELINE COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Total time: {pipeline_time:.2f}s")
        logging.info(f"Images processed: {len(image_paths)}")
        
        if 'performance_stats' in results:
            stats = results['performance_stats']
            logging.info(f"Node ID: {stats['node_id']}")
            logging.info(f"Workers: {stats['num_workers']}")
            logging.info(f"Batch size: {stats['batch_size']}")
            logging.info(f"Chunk size: {stats['chunk_size']}")
        
        # Cleanup
        pipeline.cleanup()
        
        logging.info("=" * 80)
        logging.info("Distributed pipeline completed successfully!")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Distributed pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 