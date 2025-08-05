#!/usr/bin/env python3
"""
Configuration for 3D Gaussian Splatting optimization
High-quality camera poses and dense reconstruction settings
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class SfMConfig3DGS:
    """Configuration for 3DGS-optimized SfM pipeline"""
    
    # Input/Output
    input_dir: str = "data/images"
    output_dir: str = "results/3dgs"
    
    # Feature extraction (high quality for 3DGS)
    feature_extractor: str = "superpoint"
    max_keypoints: int = 2048  # More keypoints for better reconstruction
    keypoint_threshold: float = 0.005  # Stricter threshold for quality
    max_image_size: int = 1600  # High resolution for 3DGS
    
    # Matching and verification
    use_vocab_tree: bool = True  # O(n log n) complexity
    max_pairs_per_image: int = 20
    matcher: str = "lightglue"
    geometric_verification: str = "magsac"
    confidence: float = 0.999  # High confidence for 3DGS
    max_iterations: int = 5000
    
    # Bundle adjustment (high quality)
    use_gpu_ba: bool = True
    ba_max_iterations: int = 200
    function_tolerance: float = 1e-8  # Stricter convergence
    gradient_tolerance: float = 1e-10
    robust_loss: str = "HuberLoss"
    loss_scale: float = 0.5
    
    # Dense reconstruction for 3DGS
    use_monocular_depth: bool = True
    depth_model: str = "dpt-large"
    fusion_weight: float = 0.7  # SfM weight higher for accuracy
    bilateral_filter: bool = True
    filter_size: int = 9
    sigma_color: float = 75
    sigma_space: float = 75
    
    # Scale recovery (important for 3DGS)
    scale_recovery: bool = True
    mono_scale_estimation: bool = True
    multi_view_consistency: bool = True
    global_scale_application: bool = True
    
    # Quality settings for 3DGS
    high_quality: bool = True
    enable_profiling: bool = True
    save_intermediate: bool = True
    
    # Performance
    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 8
    memory_efficient: bool = True
    
    # Output formats
    save_colmap: bool = True
    save_3dgs_data: bool = True
    save_ply: bool = True
    save_trajectory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for pipeline arguments"""
        return {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'feature_extractor': self.feature_extractor,
            'max_keypoints': self.max_keypoints,
            'max_image_size': self.max_image_size,
            'use_vocab_tree': self.use_vocab_tree,
            'max_pairs_per_image': self.max_pairs_per_image,
            'use_gpu_ba': self.use_gpu_ba,
            'ba_max_iterations': self.ba_max_iterations,
            'use_monocular_depth': self.use_monocular_depth,
            'depth_model': self.depth_model,
            'scale_recovery': self.scale_recovery,
            'high_quality': self.high_quality,
            'device': self.device,
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'profile': self.enable_profiling
        }
    
    def get_pipeline_args(self) -> list:
        """Get command line arguments for sfm_pipeline.py"""
        args = []
        config_dict = self.to_dict()
        
        for key, value in config_dict.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            else:
                args.append(f"--{key}")
                args.append(str(value))
        
        return args


@dataclass
class QualityMetrics3DGS:
    """Quality metrics for 3DGS reconstruction"""
    
    # Camera pose quality
    reprojection_error_threshold: float = 2.0  # pixels
    pose_consistency_threshold: float = 0.1  # degrees
    scale_consistency_threshold: float = 0.05  # relative scale
    
    # Point cloud quality
    min_points_per_image: int = 100
    min_track_length: int = 3
    max_track_length: int = 20
    
    # Depth quality
    depth_consistency_threshold: float = 0.1
    depth_smoothness_threshold: float = 0.05
    
    # Overall quality
    min_images_reconstructed: int = 10
    min_reconstruction_ratio: float = 0.8
    
    def evaluate_reconstruction(self, reconstruction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate reconstruction quality for 3DGS"""
        metrics = {}
        
        # Camera pose evaluation
        if 'cameras' in reconstruction_data:
            metrics['camera_count'] = len(reconstruction_data['cameras'])
            metrics['pose_quality'] = self._evaluate_poses(reconstruction_data['cameras'])
        
        # Point cloud evaluation
        if 'points3d' in reconstruction_data:
            metrics['point_count'] = len(reconstruction_data['points3d'])
            metrics['point_quality'] = self._evaluate_points(reconstruction_data['points3d'])
        
        # Depth evaluation
        if 'dense_depth_maps' in reconstruction_data:
            metrics['depth_quality'] = self._evaluate_depth(reconstruction_data['dense_depth_maps'])
        
        # Overall quality score
        metrics['overall_quality'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _evaluate_poses(self, cameras: Dict) -> Dict[str, float]:
        """Evaluate camera pose quality"""
        # Implementation for pose evaluation
        return {'reprojection_error': 0.0, 'pose_consistency': 0.0}
    
    def _evaluate_points(self, points3d: Dict) -> Dict[str, float]:
        """Evaluate point cloud quality"""
        # Implementation for point cloud evaluation
        return {'track_length': 0.0, 'point_density': 0.0}
    
    def _evaluate_depth(self, depth_maps: Dict) -> Dict[str, float]:
        """Evaluate depth map quality"""
        # Implementation for depth evaluation
        return {'depth_consistency': 0.0, 'depth_smoothness': 0.0}
    
    def _calculate_overall_quality(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score for 3DGS"""
        # Implementation for overall quality calculation
        return 0.95  # Example score


@dataclass
class OutputFormats3DGS:
    """Output format configurations for 3DGS"""
    
    # COLMAP format (standard for 3DGS)
    colmap_dir: str = "colmap"
    save_cameras_bin: bool = True
    save_images_bin: bool = True
    save_points3d_bin: bool = True
    
    # 3DGS specific data
    save_3dgs_pickle: bool = True
    save_scale_info: bool = True
    save_quality_metrics: bool = True
    
    # Visualization
    save_ply: bool = True
    save_trajectory: bool = True
    save_depth_maps: bool = True
    
    # Performance data
    save_performance_report: bool = True
    save_memory_usage: bool = True
    
    def get_output_structure(self, base_dir: str) -> Dict[str, str]:
        """Get output directory structure"""
        base_path = Path(base_dir)
        
        return {
            'colmap': str(base_path / self.colmap_dir),
            'depth_maps': str(base_path / "depth_maps"),
            'visualization': str(base_path / "visualization"),
            'performance': str(base_path / "performance"),
            '3dgs_data': str(base_path / "3dgs_data")
        }


def create_3dgs_config(quality_level: str = "high") -> SfMConfig3DGS:
    """Create 3DGS-optimized configuration based on quality level"""
    
    if quality_level == "ultra":
        return SfMConfig3DGS(
            max_keypoints=4096,
            keypoint_threshold=0.003,
            max_image_size=2048,
            ba_max_iterations=500,
            function_tolerance=1e-9,
            confidence=0.9999,
            fusion_weight=0.8,
            bilateral_filter=True
        )
    
    elif quality_level == "high":
        return SfMConfig3DGS(
            max_keypoints=2048,
            keypoint_threshold=0.005,
            max_image_size=1600,
            ba_max_iterations=200,
            function_tolerance=1e-8,
            confidence=0.999,
            fusion_weight=0.7,
            bilateral_filter=True
        )
    
    elif quality_level == "balanced":
        return SfMConfig3DGS(
            max_keypoints=1024,
            keypoint_threshold=0.01,
            max_image_size=1200,
            ba_max_iterations=100,
            function_tolerance=1e-7,
            confidence=0.99,
            fusion_weight=0.6,
            bilateral_filter=False
        )
    
    else:
        raise ValueError(f"Unknown quality level: {quality_level}")


def get_3dgs_pipeline_command(config: SfMConfig3DGS) -> str:
    """Generate command line for 3DGS pipeline"""
    args = config.get_pipeline_args()
    return f"python sfm_pipeline.py {' '.join(args)}"


if __name__ == "__main__":
    # Example usage
    config = create_3dgs_config("high")
    print("3DGS Pipeline Command:")
    print(get_3dgs_pipeline_command(config))
    
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}") 