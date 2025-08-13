"""
Optimal configuration for indoor real estate photography SfM
Specifically tuned for room-to-room matching in residential properties
"""

# NetVLAD configuration for indoor scenes
INDOOR_NETVLAD_CONFIG = {
    'use_netvlad': True,
    'netvlad_clusters': 32,  # Reduced for indoor scenes (less visual diversity)
    'feature_dim': 512,
    
    # Vocabulary tree parameters
    'vocab_size': 5000,  # Smaller vocab for indoor scenes
    'vocab_depth': 5,
    'vocab_branching_factor': 8,
    
    # Pair selection parameters
    'max_pairs_per_image': 25,  # More generous for similar rooms
    'min_score_threshold': 0.005,  # Lower threshold for similar interiors
    'ensure_connectivity': True
}

# Recommended matching strategies by dataset size
INDOOR_STRATEGY_BY_SIZE = {
    'small': (1, 20, 'netvlad_hybrid'),      # 1-20 images: hybrid approach
    'medium': (21, 100, 'netvlad_hybrid'),   # 21-100 images: hybrid with more pairs
    'large': (101, 500, 'adaptive'),         # 100+ images: adaptive with NetVLAD fallback
    'xlarge': (501, float('inf'), 'adaptive') # 500+ images: conservative approach
}

# NetVLAD specific parameters for indoor scenes
INDOOR_NETVLAD_PARAMS = {
    'min_similarity': 0.2,        # Lower threshold for similar rooms
    'max_pairs_per_image': 20,    # Generous pairing for room variations
    'use_spatial_context': True,  # Enable spatial understanding
    'room_type_weighting': True   # Weight by room type similarity
}

# Feature extraction optimizations for indoor
INDOOR_FEATURE_CONFIG = {
    'feature_extractor': 'superpoint',  # Good for indoor geometric features
    'max_features': 1500,               # Reduced for indoor scenes
    'detection_threshold': 0.003,       # Lower threshold for low-texture areas
    'nms_radius': 3,                    # Smaller radius for indoor details
}

def get_indoor_config(num_images: int) -> dict:
    """Get optimized config for indoor real estate photography"""
    
    # Determine strategy based on dataset size
    strategy = 'adaptive'
    for size_range, (min_size, max_size, recommended_strategy) in INDOOR_STRATEGY_BY_SIZE.items():
        if min_size <= num_images <= max_size:
            strategy = recommended_strategy
            break
    
    config = INDOOR_NETVLAD_CONFIG.copy()
    config.update({
        'strategy': strategy,
        'use_netvlad_fallback': True,
        'indoor_optimized': True
    })
    
    # Adjust parameters based on dataset size
    if num_images < 50:
        # Small datasets: be more generous
        config.update({
            'max_pairs_per_image': 30,
            'min_score_threshold': 0.001,
            'netvlad_clusters': 24,  # Even smaller for very similar rooms
        })
    elif num_images > 200:
        # Large datasets: be more conservative
        config.update({
            'max_pairs_per_image': 15,
            'min_score_threshold': 0.01,
            'netvlad_clusters': 48,  # More clusters for diversity
        })
    
    return config

# Usage examples for different scenarios
USAGE_EXAMPLES = {
    'single_apartment': {
        'description': 'Single apartment/house with multiple rooms',
        'expected_images': '10-30',
        'strategy': 'netvlad_hybrid',
        'config_adjustments': {
            'max_pairs_per_image': 35,
            'min_similarity': 0.15,
            'ensure_connectivity': True
        }
    },
    
    'multiple_properties': {
        'description': 'Multiple similar properties (same building/development)',
        'expected_images': '50-200',
        'strategy': 'netvlad_hybrid',
        'config_adjustments': {
            'netvlad_clusters': 40,
            'max_pairs_per_image': 20,
            'min_similarity': 0.25
        }
    },
    
    'property_portfolio': {
        'description': 'Large portfolio of diverse properties',
        'expected_images': '200+',
        'strategy': 'adaptive',
        'config_adjustments': {
            'use_netvlad_fallback': True,
            'expansion_ratio': 0.2
        }
    }
}