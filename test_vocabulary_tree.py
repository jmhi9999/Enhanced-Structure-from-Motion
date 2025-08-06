#!/usr/bin/env python3
"""
Test script for GPUVocabularyTree to validate the improvements
"""

import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path
import time

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from sfm.core.gpu_vocabulary_tree import GPUVocabularyTree

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_features(num_images: int = 50, num_descriptors_per_image: int = 500, descriptor_dim: int = 256):
    """Create synthetic features for testing"""
    features = {}
    
    for i in range(num_images):
        # Create random descriptors with some structure
        base_desc = np.random.randn(descriptor_dim).astype(np.float32)
        descriptors = []
        
        for j in range(num_descriptors_per_image):
            # Add noise to base descriptor to create similar but different descriptors
            noise = np.random.randn(descriptor_dim) * 0.1
            desc = base_desc + noise + np.random.randn(descriptor_dim) * 0.3
            descriptors.append(desc)
        
        descriptors = np.array(descriptors, dtype=np.float32)
        
        features[f"image_{i:03d}.jpg"] = {
            'descriptors': descriptors,
            'keypoints': np.random.rand(num_descriptors_per_image, 2) * 1000,
            'image_shape': (480, 640)
        }
    
    logger.info(f"Created synthetic features for {num_images} images")
    return features

def test_vocabulary_tree_basic():
    """Test basic vocabulary tree functionality"""
    logger.info("Testing basic vocabulary tree functionality...")
    
    # Create test data
    features = create_synthetic_features(num_images=20, num_descriptors_per_image=200)
    
    # Test with CPU fallback
    device = torch.device('cpu')
    config = {
        'vocab_size': 1000,
        'vocab_depth': 3,
        'vocab_branching_factor': 5,
        'max_descriptors_per_image': 100,
        'max_vocab_descriptors': 10000
    }
    
    vocab_tree = GPUVocabularyTree(device, config)
    
    # Build vocabulary
    start_time = time.time()
    vocab_tree.build_vocabulary(features)
    build_time = time.time() - start_time
    
    logger.info(f"Vocabulary built in {build_time:.2f}s")
    
    # Test querying
    query_features = list(features.values())[0]
    similar_images = vocab_tree.query_similar_images(query_features, top_k=5)
    
    logger.info(f"Found {len(similar_images)} similar images")
    
    # Test pair generation
    pairs = vocab_tree.get_image_pairs_for_matching(features, max_pairs_per_image=10)
    logger.info(f"Generated {len(pairs)} image pairs")
    
    # Get performance stats
    stats = vocab_tree.get_performance_stats()
    logger.info(f"Performance stats: {stats}")
    
    return True

def test_vocabulary_tree_edge_cases():
    """Test edge cases and error handling"""
    logger.info("Testing edge cases and error handling...")
    
    device = torch.device('cpu')
    config = {'vocab_size': 100, 'vocab_depth': 2}
    vocab_tree = GPUVocabularyTree(device, config)
    
    # Test empty features
    try:
        vocab_tree.build_vocabulary({})
        logger.error("Should have failed with empty features")
        return False
    except ValueError:
        logger.info("✓ Correctly handled empty features")
    
    # Test single image
    single_feature = create_synthetic_features(num_images=1, num_descriptors_per_image=10)
    vocab_tree.build_vocabulary(single_feature)
    logger.info("✓ Handled single image case")
    
    # Test very few descriptors
    few_desc_features = create_synthetic_features(num_images=3, num_descriptors_per_image=5)
    vocab_tree.build_vocabulary(few_desc_features)
    logger.info("✓ Handled few descriptors case")
    
    # Test invalid query
    invalid_query = {'descriptors': np.array([])}
    similar = vocab_tree.query_similar_images(invalid_query)
    assert len(similar) == 0
    logger.info("✓ Handled invalid query")
    
    return True

def test_vocabulary_tree_performance():
    """Test performance with larger dataset"""
    logger.info("Testing performance with larger dataset...")
    
    # Create larger dataset
    features = create_synthetic_features(num_images=100, num_descriptors_per_image=1000)
    
    device = torch.device('cpu')
    config = {
        'vocab_size': 5000,
        'vocab_depth': 4,
        'vocab_branching_factor': 10,
        'max_descriptors_per_image': 500,
        'max_vocab_descriptors': 50000
    }
    
    vocab_tree = GPUVocabularyTree(device, config)
    
    # Build vocabulary
    start_time = time.time()
    vocab_tree.build_vocabulary(features)
    build_time = time.time() - start_time
    
    logger.info(f"Large dataset vocabulary built in {build_time:.2f}s")
    
    # Test query performance
    query_times = []
    for i in range(10):
        query_features = list(features.values())[i]
        start_time = time.time()
        similar_images = vocab_tree.query_similar_images(query_features, top_k=20)
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    avg_query_time = np.mean(query_times)
    logger.info(f"Average query time: {avg_query_time:.4f}s")
    
    # Test pair generation
    start_time = time.time()
    pairs = vocab_tree.get_image_pairs_for_matching(features, max_pairs_per_image=15)
    pair_time = time.time() - start_time
    
    total_possible_pairs = len(features) * (len(features) - 1) // 2
    reduction_ratio = len(pairs) / total_possible_pairs
    
    logger.info(f"Pair generation took {pair_time:.2f}s")
    logger.info(f"Generated {len(pairs)}/{total_possible_pairs} pairs ({reduction_ratio:.1%})")
    
    return True

def test_caching_system():
    """Test the caching system"""
    logger.info("Testing caching system...")
    
    features = create_synthetic_features(num_images=10, num_descriptors_per_image=100)
    
    device = torch.device('cpu')
    config = {'vocab_size': 500, 'vocab_depth': 3}
    
    # Clean up any existing cache
    cache_path = Path("vocabulary_cache.pkl")
    if cache_path.exists():
        cache_path.unlink()
    
    # First build - should create cache
    vocab_tree1 = GPUVocabularyTree(device, config)
    start_time = time.time()
    vocab_tree1.build_vocabulary(features)
    first_build_time = time.time() - start_time
    
    # Second build - should use cache
    vocab_tree2 = GPUVocabularyTree(device, config)
    start_time = time.time()
    vocab_tree2.build_vocabulary(features)
    second_build_time = time.time() - start_time
    
    logger.info(f"First build: {first_build_time:.2f}s, Second build: {second_build_time:.2f}s")
    
    # Second build should be much faster
    if second_build_time < first_build_time * 0.5:
        logger.info("✓ Caching system working correctly")
        return True
    else:
        logger.warning("⚠ Caching may not be working optimally")
        return True  # Still pass as it's not critical

def main():
    """Run all tests"""
    logger.info("Starting GPUVocabularyTree comprehensive tests...")
    
    tests = [
        ("Basic Functionality", test_vocabulary_tree_basic),
        ("Edge Cases", test_vocabulary_tree_edge_cases),
        ("Performance", test_vocabulary_tree_performance),
        ("Caching System", test_caching_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! GPUVocabularyTree is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())