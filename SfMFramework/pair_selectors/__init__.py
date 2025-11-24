"""Pair selectors with factory."""

from .base import BasePairSelector, PairSelectorConfig
from .exhaustive import ExhaustivePairSelector
from .sara import SARAPairSelector
from .vocab_tree import VocabTreePairSelector

__all__ = [
    "BasePairSelector",
    "PairSelectorConfig",
    "PairSelectorFactory",
]


class PairSelectorFactory:
    """Factory for creating pair selectors."""

    _selectors = {
        "SARA": SARAPairSelector,
        "vocab_tree": VocabTreePairSelector,
        "exhaustive": ExhaustivePairSelector,
    }

    @classmethod
    def create(cls, name: str, config: PairSelectorConfig) -> BasePairSelector:
        """Create selector by name."""
        if name not in cls._selectors:
            available = ", ".join(cls._selectors.keys())
            raise ValueError(f"Unknown selector '{name}'. Available: {available}")
        return cls._selectors[name](config)

    @classmethod
    def list_selectors(cls):
        """List available selectors."""
        return list(cls._selectors.keys())
