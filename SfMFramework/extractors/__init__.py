"""Feature extractors with factory."""

from .base import BaseFeatureExtractor, ExtractorConfig, FeatureData
from .sift import SIFTExtractor
from .superpoint import SuperPointExtractor
from .aliked import ALIKEDExtractor
from .disk import DISKExtractor

__all__ = [
    "BaseFeatureExtractor",
    "ExtractorConfig",
    "FeatureData",
    "ExtractorFactory",
]


class ExtractorFactory:
    """Factory for creating feature extractors."""

    _extractors = {
        "superpoint": SuperPointExtractor,
        "aliked": ALIKEDExtractor,
        "disk": DISKExtractor,
        "sift": SIFTExtractor,
    }

    @classmethod
    def create(cls, name: str, config: ExtractorConfig) -> BaseFeatureExtractor:
        """Create extractor by name."""
        if name not in cls._extractors:
            available = ", ".join(cls._extractors.keys())
            raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
        return cls._extractors[name](config)

    @classmethod
    def list_extractors(cls):
        """List available extractors."""
        return list(cls._extractors.keys())
