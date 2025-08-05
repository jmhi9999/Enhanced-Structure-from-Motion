#!/usr/bin/env python3
"""
Example usage of the SfM Pipeline with OpenCV USAC_MAGSAC
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from sfm.core.feature_extractor import FeatureExtractorFactory
from sfm.core.feature_matcher import LightGlueMatcher
from sfm.core.geometric_verification import GeometricVerification, RANSACMethod
from sfm.core.reconstruction import IncrementalSfM
from sfm.core.dense_depth import DenseDepthEstimator
from sfm.utils.image_utils import load_images, resize_images
from sfm.utils.io_utils import save_colmap_format


def create_test_images(output_dir: str, num_images: int = 5):
    """Create synthetic test images for demonstration"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images with known geometry
    for i in range(num_images):
        # Create a simple pattern
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some geometric patterns
        cv2.rectangle(img, (100, 100), (300, 200), (255, 0, 0), -1)
        cv2.circle(img, (400, 300), 80, (0, 255, 0), -1)
        cv2.line(img, (50, 400), (550, 400), (0, 0, 255), 5)
        
        # Add some text
        cv2.putText(img, f"Image {i+1}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save image
        img_path = output_path / f"test_image_{i+1:02d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    print(f"Created {num_images} test images in {output_dir}")


def run_sfm_example():
    """Run a complete SfM example"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create test images
    test_dir = "test_images"
    create_test_images(test_dir, num_images=5)
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    image_paths = load_images(test_dir)
    processed_images = resize_images(image_paths, max_size=1600)
    
    # Initialize feature extractor
    print("Initializing SuperPoint feature extractor...")
    feature_extractor = FeatureExtractorFactory.create(
        extractor_type="superpoint",
        device=device
    )
    
    # Initialize feature matcher
    print("Initializing LightGlue matcher...")
    feature_matcher = LightGlueMatcher(device=device)
    
    # Extract features
    print("Extracting features...")
    features = feature_extractor.extract_features(
        images=processed_images,
        batch_size=4
    )
    
    # Match features
    print("Matching features...")
    matches = feature_matcher.match_features(features)
    
    # Initialize reconstruction with OpenCV USAC_MAGSAC
    print("Initializing reconstruction with OpenCV USAC_MAGSAC...")
    reconstruction = IncrementalSfM(
        device=device,
        max_image_size=1600
    )
    
    # Display geometric verification method info
    method_info = reconstruction.geometric_verifier.get_method_info()
    print(f"Geometric verification method: {method_info['method']}")
    
    # Run incremental SfM with OpenCV USAC_MAGSAC
    print("Running incremental SfM with OpenCV USAC_MAGSAC...")
    try:
        sparse_points, cameras, images = reconstruction.reconstruct(
            features=features,
            matches=matches,
            image_paths=image_paths
        )
        
        # Generate dense depth maps
        print("Generating dense depth maps...")
        dense_estimator = DenseDepthEstimator(device=device)
        dense_depth_maps = dense_estimator.estimate_dense_depth(
            sparse_points=sparse_points,
            cameras=cameras,
            images=images,
            features=features
        )
        
        # Save results
        print("Saving results...")
        output_dir = "sfm_results"
        save_colmap_format(
            output_dir=output_dir,
            cameras=cameras,
            images=images,
            points3d=sparse_points,
            dense_depth_maps=dense_depth_maps
        )
        
        print(f"SfM pipeline with OpenCV USAC_MAGSAC completed successfully!")
        print(f"Results saved to {output_dir}")
        print(f"Number of cameras: {len(cameras)}")
        print(f"Number of images: {len(images)}")
        print(f"Number of 3D points: {len(sparse_points)}")
        
    except Exception as e:
        print(f"Error during SfM reconstruction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_sfm_example() 