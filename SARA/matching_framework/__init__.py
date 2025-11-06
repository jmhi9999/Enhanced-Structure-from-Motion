"""
Modular Feature Matching Framework

A professional, hloc-style framework for feature extraction, matching, and pair selection.

Components:
- Extractors: SIFT, ORB, SuperPoint, ALIKED, DISK
- Matchers: NearestNeighbor, LightGlue
- Pair Selectors: Exhaustive, VocabTree, MPA

Usage:
    from matching_framework.extractors import ExtractorFactory, ExtractorConfig
    from matching_framework.matchers import MatcherFactory, MatcherConfig
    from matching_framework.pair_selectors import PairSelectorFactory, PairSelectorConfig

    # Create extractor
    config = ExtractorConfig(max_keypoints=4096, device="cuda")
    extractor = ExtractorFactory.create("superpoint", config)

    # Extract features
    features = extractor.extract(image_path)

    # Create matcher
    matcher_config = MatcherConfig(device="cuda")
    matcher = MatcherFactory.create("lightglue", matcher_config, extractor.get_config_dict())

    # Match
    matches = matcher.match(features0, features1)
"""

__version__ = "1.0.0"
__author__ = "Enhanced SfM Team"

from .extractors import ExtractorFactory, ExtractorConfig, FeatureData
from .matchers import MatcherFactory, MatcherConfig, MatchData
from .pair_selectors import PairSelectorFactory, PairSelectorConfig

__all__ = [
    "ExtractorFactory",
    "ExtractorConfig",
    "FeatureData",
    "MatcherFactory",
    "MatcherConfig",
    "MatchData",
    "PairSelectorFactory",
    "PairSelectorConfig",
]
