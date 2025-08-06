"""
Enhanced Structure-from-Motion Package
GPU-accelerated SfM pipeline with modern feature extractors and matchers
"""

__version__ = "0.1.0"

# Import main pipeline function
try:
    from ..sfm_pipeline import sfm_pipeline
except ImportError:
    import sys
    from pathlib import Path
    # Add parent directory to path to import sfm_pipeline
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sfm_pipeline import sfm_pipeline

# Import core components
from .core.feature_extractor import FeatureExtractorFactory
from .core.feature_matcher import EnhancedLightGlueMatcher
from .core.geometric_verification import GeometricVerification, RANSACMethod
from .core.reconstruction import IncrementalSfM
from .core.gpu_bundle_adjustment import GPUBundleAdjustment
from .core.dense_depth import DenseDepthEstimator
from .core.gpu_vocabulary_tree import GPUVocabularyTree
from .core.scale_recovery import ScaleRecovery

# Import utilities
from .utils.io_utils import save_colmap_format, load_images, save_features, save_matches
from .utils.image_utils import resize_image
from .utils.quality_metrics import QualityMetrics

__all__ = [
    # Main pipeline
    "sfm_pipeline",
    
    # Core components
    "FeatureExtractorFactory",
    "EnhancedLightGlueMatcher", 
    "GeometricVerification",
    "RANSACMethod",
    "IncrementalSfM",
    "GPUBundleAdjustment",
    "DenseDepthEstimator",
    "GPUVocabularyTree",
    "ScaleRecovery",
    
    # Utilities
    "save_colmap_format",
    "load_images", 
    "save_features",
    "save_matches",
    "resize_image",
    "QualityMetrics"
] 