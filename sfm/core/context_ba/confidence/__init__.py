"""
Confidence computation module

Provides different strategies for computing camera and point confidence scores:
- Rule-based: Hand-crafted heuristics (no training)
- Hybrid: Rule-based features + learned MLP combiner (lightweight)
"""

from .base import ConfidenceCalculator
from .rule_based import RuleBasedConfidence

# Optional hybrid confidence (requires torch)
try:
    from .hybrid import HybridConfidence
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    HybridConfidence = None

__all__ = [
    "ConfidenceCalculator",
    "RuleBasedConfidence",
    "HybridConfidence",
    "HYBRID_AVAILABLE",
]
