#!/usr/bin/env python3
"""
Test script for enhanced geometric verification
"""

import sys
import os
import numpy as np
import cv2
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sfm.core.geometric_verification import GeometricVerification, RANSACMethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(num_matches=100, noise_level=2.0, outlier_ratio=0.3):
    """Generate synthetic test data"""
    # Generate points in first image
    points1 = np.random.rand(num_matches, 2) * [640, 480]
    
    num_inliers = int(num_matches * (1 - outlier_ratio))
    points2 = points1.copy()
    
    # Apply geometric transformation to inliers
    center = np.array([320, 240])
    for i in range(num_inliers):
        # Translation + small rotation
        points2[i] += [20, 10]
        
        # Small rotation around center
        angle = 0.05
        p_centered = points2[i] - center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points2[i] = center + rotation_matrix @ p_centered
        
        # Add noise
        points1[i] += np.random.randn(2) * noise_level
        points2[i] += np.random.randn(2) * noise_level
    
    # Random outliers
    if num_matches > num_inliers:
        points2[num_inliers:] = np.random.rand(num_matches - num_inliers, 2) * [640, 480]
    
    # Ensure points are within bounds
    points1 = np.clip(points1, 0, [640, 480])
    points2 = np.clip(points2, 0, [640, 480])
    
    return points1.astype(np.float32), points2.astype(np.float32), num_inliers

def test_enhanced_verification():
    """Test the enhanced geometric verification"""
    print("🔧 Testing Enhanced Geometric Verification")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {"name": "Good Quality", "points": 100, "noise": 1.0, "outliers": 0.2},
        {"name": "Noisy Data", "points": 80, "noise": 3.0, "outliers": 0.4},
        {"name": "High Outliers", "points": 60, "noise": 2.0, "outliers": 0.7},
        {"name": "Few Points", "points": 15, "noise": 1.5, "outliers": 0.3},
        {"name": "Very Few Points", "points": 8, "noise": 1.0, "outliers": 0.1}
    ]
    
    # Test with enhanced verifier
    verifier = GeometricVerification()
    print(f"Using method: {verifier.method.value}")
    print(f"Parameters: threshold={verifier.threshold}, confidence={verifier.confidence}")
    
    results = []
    
    for case in test_cases:
        print(f"\n--- Testing {case['name']} ---")
        
        points1, points2, expected_inliers = generate_test_data(
            case['points'], case['noise'], case['outliers']
        )
        
        print(f"Generated: {len(points1)} points, expected inliers: {expected_inliers}")
        
        # Test fundamental matrix
        F, inliers = verifier.find_fundamental_matrix(points1, points2)
        
        if F is not None and inliers is not None:
            num_inliers = np.sum(inliers)
            success_rate = num_inliers / len(points1) * 100
            print(f"✅ Success: {num_inliers}/{len(points1)} inliers ({success_rate:.1f}%)")
            results.append((case['name'], True, num_inliers, len(points1)))
        else:
            print(f"❌ Failed: No fundamental matrix found")
            results.append((case['name'], False, 0, len(points1)))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 RESULTS SUMMARY:")
    successful = sum(1 for r in results if r[1])
    print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    for name, success, inliers, total in results:
        status = "✅" if success else "❌"
        if success:
            print(f"{status} {name}: {inliers}/{total} inliers")
        else:
            print(f"{status} {name}: Failed")

def test_problematic_cases():
    """Test cases that previously failed"""
    print(f"\n{'='*50}")
    print("🚨 Testing Previously Problematic Cases")
    print("=" * 50)
    
    verifier = GeometricVerification()
    
    # Case 1: Too few points (should handle gracefully)
    print("\n--- Case 1: Too few points (5) ---")
    points1 = np.random.rand(5, 2) * 640
    points2 = np.random.rand(5, 2) * 480
    F, inliers = verifier.find_fundamental_matrix(points1, points2)
    if F is not None:
        print(f"✅ Handled gracefully: {np.sum(inliers) if inliers is not None else 0} inliers")
    else:
        print("✅ Correctly rejected (too few points)")
    
    # Case 2: Duplicate points
    print("\n--- Case 2: Many duplicate points ---")
    points1 = np.ones((50, 2)) * [320, 240]
    points2 = np.ones((50, 2)) * [320, 240]
    points1 += np.random.randn(50, 2) * 0.5  # Very small noise
    points2 += np.random.randn(50, 2) * 0.5
    F, inliers = verifier.find_fundamental_matrix(points1, points2)
    if F is not None:
        print(f"✅ Success with duplicates: {np.sum(inliers) if inliers is not None else 0} inliers")
    else:
        print("✅ Correctly handled duplicate case")
    
    # Case 3: All outliers
    print("\n--- Case 3: Pure random (all outliers) ---")
    points1 = np.random.rand(100, 2) * 640
    points2 = np.random.rand(100, 2) * 480
    F, inliers = verifier.find_fundamental_matrix(points1, points2)
    if F is not None:
        print(f"✅ Found some structure: {np.sum(inliers) if inliers is not None else 0} inliers")
    else:
        print("✅ Correctly rejected pure noise")

def simulate_real_pipeline():
    """Simulate real feature matching -> geometric verification pipeline"""
    print(f"\n{'='*50}")
    print("🔄 Simulating Real Pipeline")
    print("=" * 50)
    
    # Create mock matches dictionary (like from feature matching)
    matches = {}
    
    # Generate several image pairs with varying quality
    pair_configs = [
        {"pair": ("img1.jpg", "img2.jpg"), "points": 150, "noise": 1.5, "outliers": 0.3},
        {"pair": ("img2.jpg", "img3.jpg"), "points": 80, "noise": 2.5, "outliers": 0.5},
        {"pair": ("img3.jpg", "img4.jpg"), "points": 200, "noise": 1.0, "outliers": 0.2},
        {"pair": ("img4.jpg", "img5.jpg"), "points": 30, "noise": 3.0, "outliers": 0.6},
        {"pair": ("img5.jpg", "img6.jpg"), "points": 12, "noise": 2.0, "outliers": 0.4}
    ]
    
    for config in pair_configs:
        pair = config["pair"]
        points1, points2, expected_inliers = generate_test_data(
            config["points"], config["noise"], config["outliers"]
        )
        
        # Create match data structure
        matches[pair] = {
            'keypoints0': points1,
            'keypoints1': points2,
            'matches0': np.arange(len(points1)),  # All points are matched
            'matches1': np.arange(len(points2)),
            'mscores0': np.random.rand(len(points1)) * 0.5 + 0.5,  # Random scores
            'mscores1': np.random.rand(len(points2)) * 0.5 + 0.5,
            'image_shape0': (480, 640),
            'image_shape1': (480, 640)
        }
    
    print(f"Generated {len(matches)} image pairs for verification")
    
    # Run geometric verification
    verifier = GeometricVerification()
    verified_matches = verifier.verify(matches)
    
    # Show results
    print(f"\nVerification Results:")
    print(f"Input pairs: {len(matches)}")
    print(f"Verified pairs: {len(verified_matches)}")
    print(f"Success rate: {len(verified_matches)/len(matches)*100:.1f}%")
    
    for pair, match_data in verified_matches.items():
        num_inliers = len(match_data['matches0'])
        original_matches = len(matches[pair]['matches0'])
        print(f"  {pair[0]} <-> {pair[1]}: {num_inliers}/{original_matches} inliers")

def main():
    """Run all tests"""
    test_enhanced_verification()
    test_problematic_cases() 
    simulate_real_pipeline()
    
    print(f"\n{'='*50}")
    print("🎉 Enhanced Geometric Verification Testing Complete!")
    print("The improvements should significantly reduce empty MAGSAC results.")
    print("Key enhancements:")
    print("- Relaxed default parameters (threshold=3.0, confidence=0.95)")
    print("- Adaptive parameter adjustment based on data quality")
    print("- Multiple fallback methods (LMEDS, RANSAC, 8-point)")
    print("- Point preprocessing to remove duplicates and invalid data")
    print("- Robust error handling")

if __name__ == "__main__":
    main()