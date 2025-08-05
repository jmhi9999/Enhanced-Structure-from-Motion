#!/usr/bin/env python3
"""
Test script for 3D Gaussian Splatting SfM Pipeline
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from sfm_pipeline import main as run_sfm_pipeline
from sfm.utils.quality_metrics import QualityMetrics3DGS


def create_test_images(output_dir: str, num_images: int = 20):
    """Create synthetic test images for pipeline testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic images with known patterns
    for i in range(num_images):
        # Create a simple pattern image
        img = Image.new('RGB', (800, 600), color=(100, 150, 200))
        
        # Add some synthetic features (circles, lines)
        import random
        for _ in range(50):
            x = random.randint(50, 750)
            y = random.randint(50, 550)
            radius = random.randint(5, 20)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw circle
            for dx in range(-radius, radius):
                for dy in range(-radius, radius):
                    if dx*dx + dy*dy <= radius*radius:
                        px, py = x + dx, y + dy
                        if 0 <= px < 800 and 0 <= py < 600:
                            img.putpixel((px, py), color)
        
        # Save image
        img_path = output_path / f"test_image_{i:03d}.jpg"
        img.save(img_path, quality=95)
    
    print(f"Created {num_images} test images in {output_dir}")


def test_basic_pipeline():
    """Test basic pipeline functionality"""
    
    print("Testing basic 3DGS pipeline...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "test_images")
        output_dir = os.path.join(temp_dir, "results")
        
        # Create test images
        create_test_images(input_dir, num_images=15)
        
        # Test basic pipeline
        try:
            # Prepare arguments
            sys.argv = [
                'sfm_pipeline.py',
                '--input_dir', input_dir,
                '--output_dir', output_dir,
                '--feature_extractor', 'superpoint',
                '--max_keypoints', '1024',
                '--max_image_size', '800',
                '--device', 'cpu'  # Use CPU for testing
            ]
            
            # Run pipeline
            run_sfm_pipeline()
            
            # Check outputs
            output_path = Path(output_dir)
            assert output_path.exists(), "Output directory not created"
            
            # Check for COLMAP files
            colmap_dir = output_path / "colmap"
            if colmap_dir.exists():
                print("✅ COLMAP output directory created")
            
            # Check for 3DGS data
            gs_data_file = output_path / "3dgs_data.pkl"
            if gs_data_file.exists():
                print("✅ 3DGS data file created")
            
            print("✅ Basic pipeline test passed")
            return True
            
        except Exception as e:
            print(f"❌ Basic pipeline test failed: {e}")
            return False


def test_quality_metrics():
    """Test quality metrics functionality"""
    
    print("Testing quality metrics...")
    
    # Create mock reconstruction data
    mock_data = {
        'cameras': {
            '1': {'params': [800, 600, 400, 400, 300, 300]},
            '2': {'params': [800, 600, 400, 400, 300, 300]}
        },
        'images': {
            'img1.jpg': {
                'camera_id': '1',
                'qvec': [1, 0, 0, 0],
                'tvec': [0, 0, 0],
                'xys': [[100, 100], [200, 200]],
                'point3D_ids': [1, 2]
            },
            'img2.jpg': {
                'camera_id': '2',
                'qvec': [1, 0, 0, 0],
                'tvec': [1, 0, 0],
                'xys': [[150, 150], [250, 250]],
                'point3D_ids': [1, 2]
            }
        },
        'sparse_points': {
            1: {'xyz': [0, 0, 1], 'track': [1, 2]},
            2: {'xyz': [1, 0, 1], 'track': [1, 2]}
        },
        'dense_depth_maps': {
            'img1.jpg': np.random.rand(600, 800),
            'img2.jpg': np.random.rand(600, 800)
        }
    }
    
    try:
        # Test quality metrics
        metrics = QualityMetrics3DGS()
        results = metrics.evaluate_reconstruction(mock_data)
        
        # Check that metrics were calculated
        assert 'camera_count' in results
        assert 'point_count' in results
        assert 'overall_quality' in results
        
        print("✅ Quality metrics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quality metrics test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and fallbacks"""
    
    print("Testing error handling...")
    
    # Test with non-existent directory
    try:
        sys.argv = [
            'sfm_pipeline.py',
            '--input_dir', '/non/existent/path',
            '--output_dir', '/tmp/test_output',
            '--device', 'cpu'
        ]
        
        # This should handle the error gracefully
        run_sfm_pipeline()
        
    except Exception as e:
        print(f"✅ Error handling test passed (caught expected error: {e})")
        return True
    
    print("❌ Error handling test failed (should have caught error)")
    return False


def test_memory_management():
    """Test memory management with large images"""
    
    print("Testing memory management...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "large_images")
        output_dir = os.path.join(temp_dir, "results")
        
        # Create large test images
        os.makedirs(input_dir, exist_ok=True)
        for i in range(5):
            # Create large image
            img = Image.new('RGB', (2048, 1536), color=(100, 150, 200))
            img_path = os.path.join(input_dir, f"large_image_{i}.jpg")
            img.save(img_path, quality=95)
        
        try:
            # Test with memory-efficient settings
            sys.argv = [
                'sfm_pipeline.py',
                '--input_dir', input_dir,
                '--output_dir', output_dir,
                '--max_image_size', '1024',
                '--max_keypoints', '512',
                '--device', 'cpu',
                '--batch_size', '2'
            ]
            
            run_sfm_pipeline()
            
            print("✅ Memory management test passed")
            return True
            
        except Exception as e:
            print(f"❌ Memory management test failed: {e}")
            return False


def main():
    """Run all tests"""
    
    print("🧪 Running 3DGS Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Quality Metrics", test_quality_metrics),
        ("Error Handling", test_error_handling),
        ("Memory Management", test_memory_management)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! 3DGS pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 