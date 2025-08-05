#!/usr/bin/env python3
"""
Example usage of Enhanced SfM Pipeline
Demonstrates different ways to use the pipeline
"""

import os
from pathlib import Path

# Import the enhanced SfM module
from enhanced_sfm import EnhancedSfM, quick_sfm, high_quality_sfm, fast_sfm


def example_1_quick_usage():
    """Example 1: Quick usage with default settings"""
    print("=" * 60)
    print("Example 1: Quick SfM Processing")
    print("=" * 60)
    
    # Simple one-liner for quick processing
    results = quick_sfm(
        input_dir="data/images",
        output_dir="results/quick",
        use_monocular_depth=True,
        scale_recovery=True
    )
    
    print(f"✅ Quick SfM completed!")
    print(f"📊 Results: {len(results['sparse_points'])} 3D points")
    print(f"📷 Cameras: {len(results['cameras'])}")
    print(f"⏱️  Total time: {results['total_time']:.2f}s")


def example_2_high_quality_usage():
    """Example 2: High-quality processing for 3DGS"""
    print("\n" + "=" * 60)
    print("Example 2: High-Quality SfM for 3DGS")
    print("=" * 60)
    
    # High-quality processing with optimal settings
    results = high_quality_sfm(
        input_dir="data/images",
        output_dir="results/high_quality"
    )
    
    print(f"✅ High-quality SfM completed!")
    print(f"📊 Results: {len(results['sparse_points'])} 3D points")
    print(f"📷 Cameras: {len(results['cameras'])}")
    print(f"⏱️  Total time: {results['total_time']:.2f}s")
    
    # Show performance breakdown
    if 'stage_times' in results:
        print("\n📈 Performance Breakdown:")
        for stage, time_taken in results['stage_times'].items():
            print(f"  {stage}: {time_taken:.2f}s")


def example_3_custom_configuration():
    """Example 3: Custom configuration"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)
    
    # Create custom SfM instance
    sfm = EnhancedSfM(
        feature_extractor="superpoint",
        max_keypoints=2048,
        use_vocab_tree=True,
        use_gpu_ba=True,
        use_monocular_depth=True,
        scale_recovery=True,
        high_quality=True
    )
    
    # Update configuration
    sfm.update_config(
        max_image_size=1800,
        num_workers=8,
        profile=True
    )
    
    # Process with custom settings
    results = sfm.process(
        input_dir="data/images",
        output_dir="results/custom"
    )
    
    print(f"✅ Custom SfM completed!")
    print(f"📊 Results: {len(results['sparse_points'])} 3D points")
    print(f"📷 Cameras: {len(results['cameras'])}")
    print(f"⏱️  Total time: {results['total_time']:.2f}s")


def example_4_fast_processing():
    """Example 4: Fast processing for large datasets"""
    print("\n" + "=" * 60)
    print("Example 4: Fast SfM Processing")
    print("=" * 60)
    
    # Fast processing with minimal settings
    results = fast_sfm(
        input_dir="data/images",
        output_dir="results/fast"
    )
    
    print(f"✅ Fast SfM completed!")
    print(f"📊 Results: {len(results['sparse_points'])} 3D points")
    print(f"📷 Cameras: {len(results['cameras'])}")
    print(f"⏱️  Total time: {results['total_time']:.2f}s")


def example_5_batch_processing():
    """Example 5: Batch processing multiple datasets"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    # List of datasets to process
    datasets = [
        ("data/dataset1", "results/dataset1"),
        ("data/dataset2", "results/dataset2"),
        ("data/dataset3", "results/dataset3")
    ]
    
    # Create SfM instance
    sfm = EnhancedSfM(
        use_monocular_depth=True,
        scale_recovery=True,
        profile=True
    )
    
    # Process each dataset
    for input_dir, output_dir in datasets:
        if os.path.exists(input_dir):
            print(f"🔄 Processing {input_dir}...")
            
            try:
                results = sfm.process(input_dir, output_dir)
                print(f"✅ Completed {input_dir}: {len(results['sparse_points'])} points")
            except Exception as e:
                print(f"❌ Failed {input_dir}: {e}")
        else:
            print(f"⚠️  Skipping {input_dir}: directory not found")


def example_6_advanced_usage():
    """Example 6: Advanced usage with custom configuration"""
    print("\n" + "=" * 60)
    print("Example 6: Advanced Usage")
    print("=" * 60)
    
    # Create SfM with advanced settings
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
    
    # Custom configuration for specific use case
    custom_config = {
        'max_keypoints': 3072,
        'fusion_weight': 0.75,
        'bilateral_filter': True
    }
    
    # Process with custom configuration
    results = sfm.process_with_custom_config(
        input_dir="data/images",
        output_dir="results/advanced",
        custom_config=custom_config
    )
    
    print(f"✅ Advanced SfM completed!")
    print(f"📊 Results: {len(results['sparse_points'])} 3D points")
    print(f"📷 Cameras: {len(results['cameras'])}")
    print(f"⏱️  Total time: {results['total_time']:.2f}s")
    
    # Show detailed results
    if results.get('scale_info'):
        print(f"📏 Scale info: {results['scale_info']}")
    
    if results.get('dense_depth_maps'):
        print(f"🏔️  Depth maps: {len(results['dense_depth_maps'])} generated")


def main():
    """Run all examples"""
    
    print("🚀 Enhanced SfM Pipeline Examples")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists("data/images"):
        print("⚠️  Warning: data/images directory not found")
        print("   Please create the directory and add some images")
        print("   Or modify the input paths in the examples")
        return
    
    try:
        # Run examples
        example_1_quick_usage()
        example_2_high_quality_usage()
        example_3_custom_configuration()
        example_4_fast_processing()
        example_5_batch_processing()
        example_6_advanced_usage()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("   Make sure you have the required dependencies installed")


if __name__ == "__main__":
    main() 