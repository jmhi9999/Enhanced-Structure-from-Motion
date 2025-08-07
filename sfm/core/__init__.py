"""
Core SfM components
"""

# Import core functions - some conditionally
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .geometric_verification import GeometricVerification
# Dense depth - optional import to avoid transformer dependency issues
try:
    from .dense_depth import DenseDepthEstimator
    DENSE_DEPTH_AVAILABLE = True
except (ImportError, TypeError, AttributeError):
    DENSE_DEPTH_AVAILABLE = False
    DenseDepthEstimator = None
from .scale_recovery import ScaleRecovery

# GPU modules - optional imports

try:
    from .gpu_bundle_adjustment import GPUBundleAdjustment
    GPU_BA_AVAILABLE = True
except ImportError:
    GPU_BA_AVAILABLE = False
    GPUBundleAdjustment = None

try:
    from .gpu_vocabulary_tree import GPUVocabularyTree
    GPU_VOCAB_AVAILABLE = True
except ImportError:
    GPU_VOCAB_AVAILABLE = False
    GPUVocabularyTree = None

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
    if not GPU_BA_AVAILABLE:
        raise ImportError("GPU Bundle Adjustment not available. Install with: pip install -e .[gpu]")
    ba = GPUBundleAdjustment(config or {})
    return ba.optimize(points3d, cameras)


def estimate_dense_depth(images, config=None):
    """Estimate dense depth maps"""
    if not DENSE_DEPTH_AVAILABLE:
        raise ImportError("Dense depth estimation not available. Install with: pip install transformers")
    depth_estimator = DenseDepthEstimator(config or {})
    return depth_estimator.estimate(images)

def recover_scale(points3d, depth_maps, config=None):
    """Recover scale from depth maps"""
    scale_recovery = ScaleRecovery(config or {})
    return scale_recovery.recover(points3d, depth_maps)

# Build __all__ list dynamically based on available modules
__all__ = [
    # Core classes (always available)
    "FeatureExtractor",
    "FeatureMatcher", 
    "GeometricVerification",
    "ScaleRecovery",
    # Convenience functions
    "extract_features",
    "match_features", 
    "verify_geometry",
    "bundle_adjustment",
    "recover_scale"
]

# Add optional modules if available
if DENSE_DEPTH_AVAILABLE:
    __all__.extend(["DenseDepthEstimator", "estimate_dense_depth"])

# Add GPU classes if available
if GPU_BA_AVAILABLE:
    __all__.append("GPUBundleAdjustment")
if GPU_VOCAB_AVAILABLE:
    __all__.append("GPUVocabularyTree") 