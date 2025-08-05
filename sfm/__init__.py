"""
Enhanced Structure-from-Motion Package
"""

__version__ = "0.1.0"

# Import the main classes and functions
from ..enhanced_sfm import (
    EnhancedSfM,
    quick_sfm,
    high_quality_sfm,
    fast_sfm
)

# Import core functions for direct access
from .core import (
    FeatureExtractor,
    FeatureMatcher,
    GeometricVerification,
    GPUAdvancedMAGSAC,
    GPUBundleAdjustment,
    GPUVocabularyTree,
    Reconstruction,
    DenseDepthEstimator,
    ScaleRecovery,
    DistributedProcessor,
    extract_features,
    match_features,
    verify_geometry,
    bundle_adjustment,
    reconstruct_3d,
    estimate_dense_depth,
    recover_scale
)

__all__ = [
    # High-level API
    "EnhancedSfM",
    "quick_sfm", 
    "high_quality_sfm",
    "fast_sfm",
    # Core components
    "FeatureExtractor",
    "FeatureMatcher",
    "GeometricVerification", 
    "GPUAdvancedMAGSAC",
    "GPUBundleAdjustment",
    "GPUVocabularyTree",
    "Reconstruction",
    "DenseDepthEstimator",
    "ScaleRecovery",
    "DistributedProcessor",
    # Convenience functions
    "extract_features",
    "match_features",
    "verify_geometry",
    "bundle_adjustment",
    "reconstruct_3d",
    "estimate_dense_depth",
    "recover_scale"
] 