#!/usr/bin/env python3
"""
Simple example showing how to use Enhanced SfM core functions directly
Similar to hloc usage pattern
"""

import os
from pathlib import Path
import cv2
import numpy as np

# Import core functions directly
from sfm import (
    extract_features,
    match_features,
    verify_geometry,
    reconstruct_3d,
    estimate_dense_depth,
    recover_scale
)

def simple_sfm_example():
    """Simple SfM pipeline using core functions directly"""
    
    # 1. Load images
    image_dir = "data/images"
    if not os.path.exists(image_dir):
        print(f"⚠️  {image_dir} not found. Please create it and add some images.")
        return
    
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_paths:
        print(f"⚠️  No images found in {image_dir}")
        return
    
    print(f"📷 Found {len(image_paths)} images")
    
    # 2. Extract features
    print("🔍 Extracting features...")
    features = extract_features(image_paths)
    print(f"✅ Extracted features from {len(features)} images")
    
    # 3. Match features
    print("🔗 Matching features...")
    matches = match_features(features)
    print(f"✅ Found {len(matches)} feature matches")
    
    # 4. Verify geometry
    print("✅ Verifying geometry...")
    verified_matches = verify_geometry(matches)
    print(f"✅ Verified {len(verified_matches)} matches")
    
    # 5. 3D reconstruction
    print("🏗️  Performing 3D reconstruction...")
    reconstruction = reconstruct_3d(features, verified_matches)
    print(f"✅ Reconstructed {len(reconstruction['points3d'])} 3D points")
    
    # 6. Optional: Dense depth estimation
    print("🏔️  Estimating dense depth...")
    depth_maps = estimate_dense_depth(image_paths)
    print(f"✅ Generated {len(depth_maps)} depth maps")
    
    # 7. Optional: Scale recovery
    if depth_maps:
        print("📏 Recovering scale...")
        scaled_reconstruction = recover_scale(
            reconstruction['points3d'], 
            depth_maps
        )
        print(f"✅ Scale recovered: {scaled_reconstruction['scale_factor']:.3f}")
    
    print("🎉 SfM pipeline completed!")

def custom_config_example():
    """Example with custom configuration"""
    
    # Custom configuration for each step
    feature_config = {
        'max_keypoints': 2048,
        'feature_extractor': 'superpoint'
    }
    
    match_config = {
        'matcher': 'lightglue',
        'max_ratio': 0.8
    }
    
    # Use with custom config
    features = extract_features(["image1.jpg", "image2.jpg"], feature_config)
    matches = match_features(features, match_config)
    
    print("✅ Custom configuration applied")

if __name__ == "__main__":
    print("🚀 Enhanced SfM - Simple Usage Example")
    print("=" * 50)
    
    simple_sfm_example() 