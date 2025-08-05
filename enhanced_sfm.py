#!/usr/bin/env python3
"""
Enhanced SfM Pipeline API
Clean interface for 3D Gaussian Splatting SfM processing
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Import core components directly
from sfm.core import (
    FeatureExtractor,
    FeatureMatcher,
    GeometricVerification,
    Reconstruction,
    DenseDepthEstimator,
    ScaleRecovery
)


class EnhancedSfM:
    """Enhanced SfM Pipeline for 3D Gaussian Splatting"""
    
    def __init__(self, 
                 feature_extractor: str = "superpoint",
                 max_keypoints: int = 4096,
                 max_image_size: int = 1600,
                 use_vocab_tree: bool = True,
                 max_pairs_per_image: int = 20,
                 use_gpu_ba: bool = True,
                 ba_max_iterations: int = 200,
                 use_monocular_depth: bool = True,
                 depth_model: str = "dpt-large",
                 fusion_weight: float = 0.7,
                 bilateral_filter: bool = True,
                 scale_recovery: bool = True,
                 high_quality: bool = True,
                 device: str = "auto",
                 num_workers: int = 4,
                 batch_size: int = 8,
                 profile: bool = False):
        """
        Initialize Enhanced SfM Pipeline
        
        Args:
            feature_extractor: Feature extractor to use ('superpoint', 'aliked', 'disk')
            max_keypoints: Maximum keypoints per image
            max_image_size: Maximum image size for processing
            use_vocab_tree: Use vocabulary tree for O(n log n) matching
            max_pairs_per_image: Maximum pairs per image for vocabulary tree
            use_gpu_ba: Use GPU-accelerated bundle adjustment
            ba_max_iterations: Maximum iterations for bundle adjustment
            use_monocular_depth: Use monocular depth estimation
            depth_model: Depth estimation model ('dpt-large', etc.)
            fusion_weight: Weight for SfM vs monocular depth fusion
            bilateral_filter: Apply bilateral filtering to depth maps
            scale_recovery: Enable scale recovery for 3DGS consistency
            high_quality: Enable high-quality mode
            device: Device to use ('auto', 'cpu', 'cuda')
            num_workers: Number of workers for parallel processing
            batch_size: Batch size for feature extraction
            profile: Enable performance profiling
        """
        
        self.config = {
            'feature_extractor': feature_extractor,
            'max_keypoints': max_keypoints,
            'max_image_size': max_image_size,
            'use_vocab_tree': use_vocab_tree,
            'max_pairs_per_image': max_pairs_per_image,
            'use_gpu_ba': use_gpu_ba,
            'ba_max_iterations': ba_max_iterations,
            'use_monocular_depth': use_monocular_depth,
            'depth_model': depth_model,
            'fusion_weight': fusion_weight,
            'bilateral_filter': bilateral_filter,
            'scale_recovery': scale_recovery,
            'high_quality': high_quality,
            'device': device,
            'num_workers': num_workers,
            'batch_size': batch_size,
            'profile': profile
        }
        
        # Process device setting
        if device == "auto":
            import torch
            self.config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process images with Enhanced SfM Pipeline
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing reconstruction results
        """
        
        self.logger.info("Starting Enhanced SfM Pipeline")
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline using core components
        try:
            # Initialize components
            feature_extractor = FeatureExtractor(self.config)
            feature_matcher = FeatureMatcher(self.config)
            geometric_verifier = GeometricVerification(self.config)
            reconstruction = Reconstruction(self.config)
            
            # Load images
            image_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
            if not image_paths:
                raise ValueError(f"No images found in {input_dir}")
            
            self.logger.info(f"Processing {len(image_paths)} images")
            
            # Extract features
            features = feature_extractor.extract(image_paths)
            
            # Match features
            matches = feature_matcher.match(features)
            
            # Verify geometry
            verified_matches = geometric_verifier.verify(matches)
            
            # 3D reconstruction
            reconstruction_result = reconstruction.reconstruct(features, verified_matches)
            
            # Optional: Dense depth estimation
            if self.config.get('use_monocular_depth', True):
                depth_estimator = DenseDepthEstimator(self.config)
                depth_maps = depth_estimator.estimate(image_paths)
                reconstruction_result['depth_maps'] = depth_maps
                
                # Optional: Scale recovery
                if self.config.get('scale_recovery', True):
                    scale_recovery = ScaleRecovery(self.config)
                    scaled_result = scale_recovery.recover(
                        reconstruction_result['points3d'], 
                        depth_maps
                    )
                    reconstruction_result.update(scaled_result)
            
            self.logger.info("Enhanced SfM Pipeline completed successfully!")
            return reconstruction_result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def process_with_custom_config(self, input_dir: str, output_dir: str, 
                                 custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process images with custom configuration
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            custom_config: Custom configuration dictionary
            
        Returns:
            Dictionary containing reconstruction results
        """
        
        # Merge custom config with default config
        config = self.config.copy()
        config.update(custom_config)
        
        self.logger.info("Starting Enhanced SfM Pipeline with custom configuration")
        self.logger.info(f"Custom config: {custom_config}")
        
        # Run pipeline with custom config using core components
        try:
            # Initialize components with custom config
            feature_extractor = FeatureExtractor(config)
            feature_matcher = FeatureMatcher(config)
            geometric_verifier = GeometricVerification(config)
            reconstruction = Reconstruction(config)
            
            # Load images
            image_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
            if not image_paths:
                raise ValueError(f"No images found in {input_dir}")
            
            self.logger.info(f"Processing {len(image_paths)} images with custom config")
            
            # Extract features
            features = feature_extractor.extract(image_paths)
            
            # Match features
            matches = feature_matcher.match(features)
            
            # Verify geometry
            verified_matches = geometric_verifier.verify(matches)
            
            # 3D reconstruction
            reconstruction_result = reconstruction.reconstruct(features, verified_matches)
            
            # Optional: Dense depth estimation
            if config.get('use_monocular_depth', True):
                depth_estimator = DenseDepthEstimator(config)
                depth_maps = depth_estimator.estimate(image_paths)
                reconstruction_result['depth_maps'] = depth_maps
                
                # Optional: Scale recovery
                if config.get('scale_recovery', True):
                    scale_recovery = ScaleRecovery(config)
                    scaled_result = scale_recovery.recover(
                        reconstruction_result['points3d'], 
                        depth_maps
                    )
                    reconstruction_result.update(scaled_result)
            
            self.logger.info("Enhanced SfM Pipeline completed successfully!")
            return reconstruction_result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        self.config.update(kwargs)
        self.logger.info(f"Updated configuration: {kwargs}")


# Convenience functions for quick usage

def quick_sfm(input_dir: str, output_dir: str, 
              use_monocular_depth: bool = True,
              scale_recovery: bool = True) -> Dict[str, Any]:
    """
    Quick SfM processing with default settings
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        use_monocular_depth: Enable monocular depth estimation
        scale_recovery: Enable scale recovery
        
    Returns:
        Dictionary containing reconstruction results
    """
    
    sfm = EnhancedSfM(
        use_monocular_depth=use_monocular_depth,
        scale_recovery=scale_recovery
    )
    
    return sfm.process(input_dir, output_dir)


def high_quality_sfm(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    High-quality SfM processing with optimal settings for 3DGS
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing reconstruction results
    """
    
    sfm = EnhancedSfM(
        feature_extractor="superpoint",
        max_keypoints=4096,
        max_image_size=2048,
        use_vocab_tree=True,
        max_pairs_per_image=30,
        use_gpu_ba=True,
        ba_max_iterations=500,
        use_monocular_depth=True,
        depth_model="dpt-large",
        fusion_weight=0.8,
        bilateral_filter=True,
        scale_recovery=True,
        high_quality=True,
        device="auto",
        num_workers=8,
        batch_size=16,
        profile=True
    )
    
    return sfm.process(input_dir, output_dir)


def fast_sfm(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Fast SfM processing with balanced settings
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing reconstruction results
    """
    
    sfm = EnhancedSfM(
        feature_extractor="superpoint",
        max_keypoints=1024,
        max_image_size=1200,
        use_vocab_tree=True,
        max_pairs_per_image=15,
        use_gpu_ba=False,
        ba_max_iterations=100,
        use_monocular_depth=False,
        scale_recovery=False,
        high_quality=False,
        device="auto",
        num_workers=4,
        batch_size=4,
        profile=False
    )
    
    return sfm.process(input_dir, output_dir)


# Example usage
if __name__ == "__main__":
    # Quick usage
    results = quick_sfm("data/images", "results/quick")
    
    # High-quality usage
    results = high_quality_sfm("data/images", "results/high_quality")
    
    # Custom configuration
    sfm = EnhancedSfM()
    sfm.update_config(max_keypoints=2048, use_gpu_ba=True)
    results = sfm.process("data/images", "results/custom") 