"""
Core SfM components
"""

# Import all core functions for direct access
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .geometric_verification import GeometricVerification
from .gpu_advanced_magsac import GPUAdvancedMAGSAC
from .gpu_bundle_adjustment import GPUBundleAdjustment
from .gpu_vocabulary_tree import GPUVocabularyTree
from .reconstruction import Reconstruction
from .dense_depth import DenseDepthEstimator
from .scale_recovery import ScaleRecovery
from .distributed_processor import DistributedProcessor

# Convenience functions for direct usage
def extract_features(images, config=None):
    """Extract features from images"""
    extractor = FeatureExtractor(config or {})
    return extractor.extract(images)

def match_features(features, config=None):
    """Match features between images"""
    matcher = FeatureMatcher(config or {})
    return matcher.match(features)

def verify_geometry(matches, config=None):
    """Verify geometric consistency"""
    verifier = GeometricVerification(config or {})
    return verifier.verify(matches)

def bundle_adjustment(points3d, cameras, config=None):
    """Perform bundle adjustment"""
    ba = GPUBundleAdjustment(config or {})
    return ba.optimize(points3d, cameras)

def reconstruct_3d(features, matches, config=None):
    """Perform 3D reconstruction"""
    reconstruction = Reconstruction(config or {})
    return reconstruction.reconstruct(features, matches)

def estimate_dense_depth(images, config=None):
    """Estimate dense depth maps"""
    depth_estimator = DenseDepthEstimator(config or {})
    return depth_estimator.estimate(images)

def recover_scale(points3d, depth_maps, config=None):
    """Recover scale from depth maps"""
    scale_recovery = ScaleRecovery(config or {})
    return scale_recovery.recover(points3d, depth_maps)

__all__ = [
    # Classes
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