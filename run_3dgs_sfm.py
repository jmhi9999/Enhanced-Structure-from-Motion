#!/usr/bin/env python3
"""
3D Gaussian Splatting SfM Pipeline Runner
Optimized for high-quality camera poses and dense reconstruction
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

from config_3dgs import SfMConfig3DGS, create_3dgs_config, QualityMetrics3DGS, OutputFormats3DGS


def setup_logging():
    """Setup logging for 3DGS pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_input_directory(input_dir: str) -> bool:
    """Validate input directory for 3DGS processing"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if len(image_files) < 10:
        logging.warning(f"Only {len(image_files)} images found. 3DGS typically needs 20+ images.")
        return False
    
    logging.info(f"Found {len(image_files)} images in {input_dir}")
    return True


def create_output_structure(output_dir: str) -> Dict[str, str]:
    """Create output directory structure for 3DGS"""
    output_path = Path(output_dir)
    
    # Create main directories
    directories = {
        'colmap': output_path / "colmap",
        'depth_maps': output_path / "depth_maps", 
        'visualization': output_path / "visualization",
        'performance': output_path / "performance",
        '3dgs_data': output_path / "3dgs_data",
        'logs': output_path / "logs"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in directories.items()}


def run_3dgs_pipeline(config: SfMConfig3DGS) -> Dict[str, Any]:
    """Run the 3DGS-optimized SfM pipeline"""
    
    logging.info("=" * 80)
    logging.info("3D GAUSSIAN SPLATTING SFM PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Quality Level: {'Ultra' if config.max_keypoints >= 4096 else 'High' if config.max_keypoints >= 2048 else 'Balanced'}")
    logging.info(f"Input Directory: {config.input_dir}")
    logging.info(f"Output Directory: {config.output_dir}")
    logging.info(f"Feature Extractor: {config.feature_extractor}")
    logging.info(f"Max Keypoints: {config.max_keypoints}")
    logging.info(f"GPU Bundle Adjustment: {config.use_gpu_ba}")
    logging.info(f"Monocular Depth: {config.use_monocular_depth}")
    logging.info(f"Scale Recovery: {config.scale_recovery}")
    
    # Validate input
    if not validate_input_directory(config.input_dir):
        return {'success': False, 'error': 'Invalid input directory'}
    
    # Create output structure
    output_dirs = create_output_structure(config.output_dir)
    
    # Import pipeline components
    try:
        from sfm_pipeline import main as run_sfm_pipeline
        import sys
        
        # Prepare arguments for sfm_pipeline.py
        pipeline_args = config.get_pipeline_args()
        
        # Temporarily modify sys.argv to pass arguments
        original_argv = sys.argv
        sys.argv = ['sfm_pipeline.py'] + pipeline_args
        
        # Run the pipeline
        start_time = time.time()
        run_sfm_pipeline()
        pipeline_time = time.time() - start_time
        
        # Restore original argv
        sys.argv = original_argv
        
        return {
            'success': True,
            'pipeline_time': pipeline_time,
            'output_dirs': output_dirs,
            'config': config.to_dict()
        }
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        return {'success': False, 'error': str(e)}


def evaluate_3dgs_quality(output_dir: str) -> Dict[str, Any]:
    """Evaluate reconstruction quality for 3DGS"""
    logging.info("Evaluating 3DGS reconstruction quality...")
    
    quality_metrics = QualityMetrics3DGS()
    
    # Load reconstruction data
    try:
        import pickle
        with open(Path(output_dir) / "3dgs_data.pkl", 'rb') as f:
            reconstruction_data = pickle.load(f)
        
        # Evaluate quality
        metrics = quality_metrics.evaluate_reconstruction(reconstruction_data)
        
        logging.info("Quality Evaluation Results:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Quality evaluation failed: {str(e)}")
        return {'error': str(e)}


def generate_3dgs_report(results: Dict[str, Any], output_dir: str):
    """Generate comprehensive report for 3DGS"""
    logging.info("Generating 3DGS report...")
    
    report = {
        'pipeline_results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'quality_metrics': evaluate_3dgs_quality(output_dir),
        'output_structure': create_output_structure(output_dir)
    }
    
    # Save report
    import json
    report_file = Path(output_dir) / "3dgs_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logging.info(f"Report saved to: {report_file}")
    
    return report


def main():
    """Main 3DGS pipeline runner"""
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting SfM Pipeline")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for 3DGS results")
    
    # Quality levels
    parser.add_argument("--quality", type=str, default="high",
                       choices=["balanced", "high", "ultra"],
                       help="Quality level for 3DGS reconstruction")
    
    # Feature extraction
    parser.add_argument("--feature_extractor", type=str, default="superpoint",
                       choices=["superpoint", "aliked", "disk"],
                       help="Feature extractor to use")
    
    # Performance options
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for parallel processing")
    
    # Advanced options
    parser.add_argument("--custom_config", type=str,
                       help="Path to custom configuration file")
    parser.add_argument("--evaluate_only", action="store_true",
                       help="Only evaluate existing reconstruction")
    parser.add_argument("--generate_report", action="store_true",
                       help="Generate detailed 3DGS report")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create configuration
    if args.custom_config:
        # Load custom configuration
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_config", args.custom_config)
        custom_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config_module)
        config = custom_config_module.get_config()
    else:
        # Use quality-based configuration
        config = create_3dgs_config(args.quality)
        
        # Override with command line arguments
        config.input_dir = args.input_dir
        config.output_dir = args.output_dir
        config.feature_extractor = args.feature_extractor
        config.device = args.device
        config.num_workers = args.num_workers
    
    # Run pipeline or evaluate only
    if args.evaluate_only:
        logging.info("Running quality evaluation only...")
        results = evaluate_3dgs_quality(args.output_dir)
    else:
        logging.info("Running full 3DGS pipeline...")
        results = run_3dgs_pipeline(config)
    
    # Generate report if requested
    if args.generate_report or results.get('success', False):
        report = generate_3dgs_report(results, args.output_dir)
        
        # Print summary
        logging.info("=" * 80)
        logging.info("3DGS PIPELINE SUMMARY")
        logging.info("=" * 80)
        
        if results.get('success'):
            logging.info(f"✅ Pipeline completed successfully")
            logging.info(f"⏱️  Total time: {results.get('pipeline_time', 0):.2f}s")
            logging.info(f"📁 Output directory: {args.output_dir}")
            
            if 'quality_metrics' in report:
                metrics = report['quality_metrics']
                logging.info(f"📊 Camera count: {metrics.get('camera_count', 'N/A')}")
                logging.info(f"📊 Point count: {metrics.get('point_count', 'N/A')}")
                logging.info(f"📊 Overall quality: {metrics.get('overall_quality', 'N/A'):.3f}")
        else:
            logging.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
    
    logging.info("=" * 80)
    logging.info("3DGS pipeline completed!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main() 