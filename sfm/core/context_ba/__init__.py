"""
Context-Aware Bundle Adjustment module

This module provides a drop-in replacement for COLMAP's traditional Bundle Adjustment
by leveraging global scene understanding through scene graphs and confidence weighting.

Key Features:
- Scene graph construction from features and matches
- Rule-based and hybrid confidence computation
- Context-aware weighted optimization
- Backward compatible with existing pipeline

Usage:
    from sfm.core.context_ba import ContextAwareBundleAdjustment

    ba = ContextAwareBundleAdjustment(config)
    cameras, images, points3d = ba.optimize(features, matches, image_dir)
"""

from .config import ContextBAConfig
from .scene_graph import SceneGraph, SceneGraphBuilder
from .confidence import (
    ConfidenceCalculator,
    RuleBasedConfidence,
    HybridConfidence,
    HYBRID_AVAILABLE
)
from .optimizer import ContextAwareBundleAdjustment

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "ContextBAConfig",

    # Scene graph
    "SceneGraph",
    "SceneGraphBuilder",

    # Confidence
    "ConfidenceCalculator",
    "RuleBasedConfidence",
    "HybridConfidence",
    "HYBRID_AVAILABLE",

    # Main optimizer
    "ContextAwareBundleAdjustment",
]
