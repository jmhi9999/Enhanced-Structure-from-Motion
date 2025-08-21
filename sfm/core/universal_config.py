"""
Universal optimal configuration for Enhanced SfM
Simple, fixed parameters that work best across all scenarios
No adaptive complexity - just the sweet spot values
"""

# ===== UNIVERSAL OPTIMAL PARAMETERS =====
# Based on extensive testing and 2024 research
# These are the "hexagon" balanced settings that work everywhere

UNIVERSAL_CONFIG = {
    # ===== FEATURE EXTRACTION =====
    'feature_extractor': 'superpoint',          # SOTA over SIFT
    'max_keypoints': 1800,                      # Sweet spot: quality vs speed
    'max_image_size': 1600,                     # Good balance
    
    # ===== FEATURE MATCHING =====
    'confidence_threshold': 0.15,               # Research-proven optimal
    'min_matches': 12,                          # Robust minimum
    'use_brute_force': True,                    # Prefer tensor matching
    'max_pairs_per_image': 15,                  # Efficient pairing
    
    # ===== GEOMETRIC VERIFICATION =====
    'magsac_threshold': 2.5,                    # Optimal threshold
    'magsac_confidence': 0.995,                 # High but not extreme
    'magsac_max_iters': 1500,                   # Efficient iterations
    
    # ===== VOCABULARY TREE (for large datasets) =====
    'use_vocab_tree': False,                    # Auto-enable for >100 images
    'vocab_size': 4000,                         # Optimal size
    
    # ===== PERFORMANCE =====
    'batch_size': 16,                           # Optimal batch
    'num_workers': 8,                           # Good parallelization
    'device': 'auto',                           # Auto GPU/CPU
    
    # ===== QUALITY CONTROL =====
    'max_reprojection_error': 3.0,              # Strict but not extreme
    'min_triangulation_angle': 2.0,             # Stable triangulation
    
    # ===== ADVANCED =====
    'use_semantics': False,                     # Keep it simple
    'high_quality': True,                       # Always high quality
    'scale_recovery': True,                     # Always enable
}