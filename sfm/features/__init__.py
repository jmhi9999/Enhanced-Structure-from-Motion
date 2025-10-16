"""
Feature modules for the Enhanced SfM pipeline.

Currently exposes the unified DINO feature extractor capable of loading
DINOv3 (default) or DINOv2 backbones as described in the proposal.
"""

from .dino_feature_extractor import DINOFeatureExtractor

__all__ = ["DINOFeatureExtractor"]
