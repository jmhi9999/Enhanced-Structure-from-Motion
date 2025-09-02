#!/usr/bin/env python3
"""
Test script for simplified SuperPoint-based SfM pipeline
Core flow: Feature extraction -> Pair selection -> Matching -> MAGSAC -> COLMAP reconstruction
"""

from sfm import sfm_pipeline

def test_simplified_sfm():   
    """Test Simplified SfM with SuperPoint (InLoc model by default)"""
    print("Testing Simplified SuperPoint-based SfM pipeline...")
    
    try:
        result = sfm_pipeline(
            "images_Dohyeon_Lee",
            "output_v2/large_dataset_sfm",
            feature_extractor="superpoint",  # Uses InLoc (indoor model) by default
            use_vocab_tree=True,
            use_brute_force=False,
            high_quality=True,
            max_keypoints=2500,  # Balanced for 330 images
            magsac_threshold=2.0,  # Balanced threshold for large dataset
            max_pairs_per_image=20,  # Optimized for large datasets
            batch_size=16,  # Larger batch for efficiency
            device="cuda"
        )
        print("✅ Simplified SfM pipeline completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Simplified SfM pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_simplified_sfm()

        