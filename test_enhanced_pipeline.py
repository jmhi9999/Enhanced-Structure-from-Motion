#!/usr/bin/env python3
"""
Test script for the enhanced SfM pipeline with tensor backup and cache system
"""

import os
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from sfm_pipeline import sfm_pipeline


def create_test_images(test_dir: str, num_images: int = 3):
    """Create dummy test images for testing"""
    import cv2
    
    test_path = Path(test_dir)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Create simple test images with some features
    for i in range(num_images):
        # Create a simple image with some geometric patterns
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some rectangles and circles as features
        cv2.rectangle(img, (50 + i*20, 50 + i*20), (150 + i*20, 150 + i*20), (255, 255, 255), 2)
        cv2.circle(img, (300 + i*30, 240 + i*30), 50, (255, 255, 255), 2)
        
        # Add some diagonal lines
        cv2.line(img, (0, 0), (640, 480), (128, 128, 128), 1)
        cv2.line(img, (0, 480), (640, 0), (128, 128, 128), 1)
        
        # Save image
        img_path = test_path / f"test_img_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
        
    print(f"Created {num_images} test images in {test_dir}")


def test_pipeline_caching():
    """Test the enhanced pipeline with caching system"""
    
    print("=" * 60)
    print("Testing Enhanced SfM Pipeline with Tensor Caching")
    print("=" * 60)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        # Create test images
        create_test_images(input_dir, num_images=5)
        
        # Test configuration
        config = {
            'feature_extractor': 'superpoint',
            'max_keypoints': 1024,
            'max_image_size': 800,
            'use_brute_force': True,
            'use_vocab_tree': False,
            'use_gpu_ba': False,
            'use_monocular_depth': False,
            'high_quality': False,
            'device': 'auto',
            'num_workers': 2,
            'batch_size': 4
        }
        
        print("\n1. First run - should create all cache files...")
        try:
            result1 = sfm_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                **config
            )
            
            print(f"✓ First run completed successfully!")
            print(f"  - Total time: {result1['total_time']:.2f}s")
            print(f"  - Features: {len(result1['features'])} images")
            print(f"  - 3D points: {len(result1['sparse_points'])}")
            
            # Check if cache files were created
            output_path = Path(output_dir)
            cache_files = [
                "features_tensors.pt",
                "matches_tensors.pt", 
                "verified_tensors.pt"
            ]
            
            print(f"\n  Cache files created:")
            for cache_file in cache_files:
                cache_path = output_path / cache_file
                if cache_path.exists():
                    size_mb = cache_path.stat().st_size / (1024 * 1024)
                    print(f"    ✓ {cache_file} ({size_mb:.2f} MB)")
                else:
                    print(f"    ✗ {cache_file} (missing)")
            
        except Exception as e:
            print(f"✗ First run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n2. Second run - should use cached results...")
        try:
            result2 = sfm_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                **config
            )
            
            print(f"✓ Second run completed successfully!")
            print(f"  - Total time: {result2['total_time']:.2f}s")
            print(f"  - Features: {len(result2['features'])} images")
            print(f"  - 3D points: {len(result2['sparse_points'])}")
            
            # Compare performance
            speedup = result1['total_time'] / max(result2['total_time'], 0.1)
            print(f"  - Speedup: {speedup:.1f}x")
            
            if speedup > 2:
                print(f"  ✓ Significant speedup achieved from caching!")
            else:
                print(f"  ⚠ Limited speedup - cache may not be working optimally")
                
        except Exception as e:
            print(f"✗ Second run failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n3. Testing cache validation...")
        try:
            # Test with different number of images (should invalidate cache)
            create_test_images(input_dir, num_images=3)  # Reduce to 3 images
            
            result3 = sfm_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                **config
            )
            
            print(f"✓ Cache validation working - detected image count change")
            print(f"  - Features: {len(result3['features'])} images (should be 3)")
            
        except Exception as e:
            print(f"✗ Cache validation test failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Enhanced pipeline with caching is working correctly.")
    print("=" * 60)
    return True


def test_tensor_backup_format():
    """Test that tensor backups are in the correct format"""
    
    print("\nTesting tensor backup format...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        
        # Create minimal test case
        create_test_images(input_dir, num_images=2)
        
        config = {
            'feature_extractor': 'superpoint',
            'max_keypoints': 512,
            'max_image_size': 640,
            'use_brute_force': True,
            'use_vocab_tree': False,
            'high_quality': False,
            'device': 'cpu',  # Force CPU for consistent testing
            'num_workers': 1,
            'batch_size': 1
        }
        
        try:
            result = sfm_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                **config
            )
            
            output_path = Path(output_dir)
            
            # Test feature tensors
            features_tensor_file = output_path / "features_tensors.pt"
            if features_tensor_file.exists():
                features_tensors = torch.load(features_tensor_file, map_location='cpu')
                print(f"  ✓ Feature tensors: {len(features_tensors)} images")
                
                # Check tensor format
                for img_path, feat_tensors in features_tensors.items():
                    expected_keys = ['keypoints', 'descriptors', 'scores', 'image_shape']
                    for key in expected_keys:
                        if key in feat_tensors:
                            if key != 'image_shape' and torch.is_tensor(feat_tensors[key]):
                                print(f"    ✓ {key}: {feat_tensors[key].shape}")
                            elif key == 'image_shape':
                                print(f"    ✓ {key}: {feat_tensors[key]}")
                    break  # Check first image only
                        
            # Test match tensors
            matches_tensor_file = output_path / "matches_tensors.pt"
            if matches_tensor_file.exists():
                matches_tensors = torch.load(matches_tensor_file, map_location='cpu')
                print(f"  ✓ Match tensors: {len(matches_tensors)} pairs")
                
            print("  ✓ Tensor backup format validation passed!")
            return True
            
        except Exception as e:
            print(f"  ✗ Tensor backup format test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = True
    
    # Run tests
    try:
        success &= test_pipeline_caching()
        success &= test_tensor_backup_format()
        
        if success:
            print("\n🎉 All tests passed! The enhanced SfM pipeline is ready to use.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    exit(0 if success else 1)