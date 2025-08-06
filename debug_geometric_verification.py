#!/usr/bin/env python3
"""
Debug script to diagnose geometric verification issues
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_point_distribution(points1, points2):
    """Analyze the distribution and quality of matched points"""
    print(f"\n=== POINT DISTRIBUTION ANALYSIS ===")
    print(f"Number of points: {len(points1)}")
    
    # Check for duplicates
    unique_p1 = np.unique(points1, axis=0)
    unique_p2 = np.unique(points2, axis=0)
    print(f"Unique points1: {len(unique_p1)}/{len(points1)} ({len(unique_p1)/len(points1)*100:.1f}%)")
    print(f"Unique points2: {len(unique_p2)}/{len(points2)} ({len(unique_p2)/len(points2)*100:.1f}%)")
    
    # Check point spread
    p1_std = np.std(points1, axis=0)
    p2_std = np.std(points2, axis=0)
    print(f"Points1 spread (std): x={p1_std[0]:.1f}, y={p1_std[1]:.1f}")
    print(f"Points2 spread (std): x={p2_std[0]:.1f}, y={p2_std[1]:.1f}")
    
    # Check for degenerate configurations
    # Colinearity check
    if len(points1) >= 3:
        # Check if points lie on a line
        p1_centered = points1 - np.mean(points1, axis=0)
        U, S, Vt = np.linalg.svd(p1_centered)
        ratio = S[1] / S[0] if S[0] > 0 else 0
        print(f"Points1 colinearity ratio: {ratio:.4f} (< 0.1 = colinear)")
        
        p2_centered = points2 - np.mean(points2, axis=0)
        U, S, Vt = np.linalg.svd(p2_centered)
        ratio = S[1] / S[0] if S[0] > 0 else 0
        print(f"Points2 colinearity ratio: {ratio:.4f} (< 0.1 = colinear)")
    
    # Check distances between consecutive points
    if len(points1) > 1:
        dists1 = np.linalg.norm(points1[1:] - points1[:-1], axis=1)
        dists2 = np.linalg.norm(points2[1:] - points2[:-1], axis=1)
        print(f"Mean distance between points1: {np.mean(dists1):.1f}")
        print(f"Mean distance between points2: {np.mean(dists2):.1f}")
        print(f"Min distance between points1: {np.min(dists1):.1f}")
        print(f"Min distance between points2: {np.min(dists2):.1f}")

def test_opencv_fundamental_matrix(points1, points2):
    """Test OpenCV fundamental matrix with different parameters"""
    print(f"\n=== OPENCV FUNDAMENTAL MATRIX TESTS ===")
    
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    
    # Test different methods and parameters
    test_configs = [
        {"method": cv2.USAC_MAGSAC, "threshold": 1.0, "confidence": 0.999, "name": "MAGSAC (strict)"},
        {"method": cv2.USAC_MAGSAC, "threshold": 3.0, "confidence": 0.95, "name": "MAGSAC (relaxed)"},
        {"method": cv2.USAC_MAGSAC, "threshold": 5.0, "confidence": 0.90, "name": "MAGSAC (very relaxed)"},
        {"method": cv2.FM_RANSAC, "threshold": 3.0, "confidence": 0.99, "name": "RANSAC"},
        {"method": cv2.FM_LMEDS, "threshold": 0, "confidence": 0.99, "name": "LMEDS"},
        {"method": cv2.FM_8POINT, "threshold": 0, "confidence": 0, "name": "8-point (no RANSAC)"}
    ]
    
    results = []
    
    for config in test_configs:
        try:
            if config["method"] == cv2.FM_8POINT:
                # 8-point doesn't use RANSAC parameters
                F, mask = cv2.findFundamentalMat(points1, points2, method=config["method"])
            else:
                F, mask = cv2.findFundamentalMat(
                    points1, points2,
                    method=config["method"],
                    ransacReprojThreshold=config["threshold"],
                    confidence=config["confidence"],
                    maxIters=10000
                )
            
            if F is not None and F.size > 0:
                num_inliers = np.sum(mask) if mask is not None else 0
                print(f"✅ {config['name']}: F matrix found, {num_inliers} inliers")
                results.append((config["name"], F, mask, num_inliers))
            else:
                print(f"❌ {config['name']}: Failed to find F matrix")
                results.append((config["name"], None, None, 0))
                
        except Exception as e:
            print(f"❌ {config['name']}: Exception - {e}")
            results.append((config["name"], None, None, 0))
    
    return results

def generate_synthetic_matches(num_matches=50, noise_level=1.0, outlier_ratio=0.3):
    """Generate synthetic point matches for testing"""
    print(f"\n=== GENERATING SYNTHETIC MATCHES ===")
    print(f"Matches: {num_matches}, Noise: {noise_level}, Outliers: {outlier_ratio}")
    
    # Simpler approach - generate points that follow epipolar geometry
    # Generate points in first image
    points1 = np.random.rand(num_matches, 2) * [640, 480]
    
    # Simulate a simple transformation (translation + small rotation)
    # This ensures the points follow proper epipolar geometry
    num_inliers = int(num_matches * (1 - outlier_ratio))
    
    # For inliers, apply geometric transformation
    points2 = points1.copy()
    
    # Apply translation and small rotation to inliers
    center = np.array([320, 240])
    for i in range(num_inliers):
        # Translate
        points2[i] += [20, 10]
        
        # Small rotation around center
        angle = 0.05  # Small angle in radians
        p_centered = points2[i] - center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points2[i] = center + rotation_matrix @ p_centered
        
        # Add noise
        points1[i] += np.random.randn(2) * noise_level
        points2[i] += np.random.randn(2) * noise_level
    
    # For outliers, use random points
    if num_matches > num_inliers:
        points2[num_inliers:] = np.random.rand(num_matches - num_inliers, 2) * [640, 480]
    
    # Ensure points are within image bounds
    points1 = np.clip(points1, 0, [640, 480])
    points2 = np.clip(points2, 0, [640, 480])
    
    return points1.astype(np.float32), points2.astype(np.float32), num_inliers

def test_with_real_matches():
    """Test with potentially problematic real matches"""
    print(f"\n=== TESTING WITH PROBLEMATIC CASES ===")
    
    # Case 1: Too few points
    points1 = np.random.rand(5, 2) * 640
    points2 = np.random.rand(5, 2) * 480
    print("\n--- Case 1: Too few points (5) ---")
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Case 2: All points identical (duplicate)
    points1 = np.ones((50, 2)) * [320, 240]
    points2 = np.ones((50, 2)) * [320, 240]
    points1 += np.random.randn(50, 2) * 0.1  # Tiny noise
    points2 += np.random.randn(50, 2) * 0.1
    print("\n--- Case 2: Nearly identical points ---")
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Case 3: Colinear points
    t = np.linspace(0, 1, 50)
    points1 = np.column_stack([t * 640, t * 480])  # Line from (0,0) to (640,480)
    points2 = np.column_stack([t * 640, t * 480])  # Same line
    points1 += np.random.randn(50, 2) * 2  # Small noise
    points2 += np.random.randn(50, 2) * 2
    print("\n--- Case 3: Colinear points ---")
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Case 4: All outliers
    points1 = np.random.rand(100, 2) * 640
    points2 = np.random.rand(100, 2) * 480
    print("\n--- Case 4: Pure random (all outliers) ---")
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)

def main():
    print("🔍 Geometric Verification Debugging Tool")
    print("=" * 50)
    
    # Test 1: Good synthetic data
    print("\n🟢 TEST 1: Good synthetic data")
    points1, points2, expected_inliers = generate_synthetic_matches(100, 1.0, 0.2)
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Test 2: Noisy synthetic data
    print("\n🟡 TEST 2: Noisy synthetic data")
    points1, points2, expected_inliers = generate_synthetic_matches(100, 3.0, 0.5)
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Test 3: Very noisy synthetic data
    print("\n🔴 TEST 3: Very noisy synthetic data")
    points1, points2, expected_inliers = generate_synthetic_matches(50, 5.0, 0.8)
    analyze_point_distribution(points1, points2)
    results = test_opencv_fundamental_matrix(points1, points2)
    
    # Test 4: Problematic real cases
    test_with_real_matches()
    
    print(f"\n{'='*50}")
    print("🎯 RECOMMENDATIONS:")
    print("1. Check input match quality before geometric verification")
    print("2. Use relaxed MAGSAC parameters for noisy data:")
    print("   - threshold: 3.0-5.0 instead of 1.0") 
    print("   - confidence: 0.95 instead of 0.999")
    print("3. Prefilter matches to remove obvious outliers")
    print("4. Check for degenerate point configurations")
    print("5. Use adaptive parameters based on point distribution")

if __name__ == "__main__":
    main()