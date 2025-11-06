"""Feature matchers with factory."""

from .base import BaseMatcher, MatcherConfig, MatchData
from .nn import NNMatcher
from .lightglue import LightGlueMatcher

__all__ = [
    "BaseMatcher",
    "MatcherConfig",
    "MatchData",
    "MatcherFactory",
]


class MatcherFactory:
    """Factory for creating matchers."""

    _matchers = {
        "nn": NNMatcher,
        "lightglue": LightGlueMatcher,
    }

    @classmethod
    def create(
        cls,
        name: str,
        config: MatcherConfig,
        extractor_config: dict
    ) -> BaseMatcher:
        """Create matcher by name."""
        if name not in cls._matchers:
            available = ", ".join(cls._matchers.keys())
            raise ValueError(f"Unknown matcher '{name}'. Available: {available}")
        return cls._matchers[name](config, extractor_config)

    @classmethod
    def list_matchers(cls):
        """List available matchers."""
        return list(cls._matchers.keys())
