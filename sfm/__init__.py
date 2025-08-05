"""
Enhanced Structure-from-Motion Package
"""

__version__ = "0.1.0"

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