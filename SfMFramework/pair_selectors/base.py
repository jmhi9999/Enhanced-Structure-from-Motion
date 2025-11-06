"""
Base class for pair selection strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class PairSelectorConfig:
    """Configuration for pair selection."""

    # Common parameters
    max_pairs_per_image: int = 20
    min_pairs_per_image: int = 1

    # Device
    device: str = "cuda"

    # Selector-specific
    extra: Dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BasePairSelector(ABC):
    """
    Abstract base class for pair selection strategies.

    All selectors must implement:
    - _select_impl(): Core selection logic
    """

    def __init__(self, config: PairSelectorConfig):
        self.config = config
        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize selector."""
        pass

    @abstractmethod
    def _select_impl(
        self,
        image_list: List[Path],
        features: Dict[Path, any] = None
    ) -> List[Tuple[Path, Path]]:
        """
        Core pair selection implementation.

        Args:
            image_list: List of image paths
            features: Optional dictionary of features for similarity-based selection

        Returns:
            List of image pairs (path0, path1)
        """
        pass

    @property
    def name(self) -> str:
        """Selector name."""
        return self.__class__.__name__.replace("PairSelector", "").lower()

    def select(
        self,
        image_list: List[Path],
        features: Dict[Path, any] = None
    ) -> List[Tuple[Path, Path]]:
        """
        Select image pairs for matching.

        Args:
            image_list: List of image paths
            features: Optional dictionary of features

        Returns:
            List of unique image pairs
        """
        if len(image_list) < 2:
            raise ValueError("Need at least 2 images for pair selection")

        # Select pairs
        pairs = self._select_impl(image_list, features)

        # Ensure uniqueness and canonical ordering
        pairs = self._canonicalize_pairs(pairs)

        # Validate pair counts
        self._validate_pairs(image_list, pairs)

        return pairs

    def _canonicalize_pairs(
        self,
        pairs: List[Tuple[Path, Path]]
    ) -> List[Tuple[Path, Path]]:
        """Ensure pairs are unique and canonically ordered."""
        seen = set()
        canonical_pairs = []

        for p0, p1 in pairs:
            # Canonical ordering: smaller path first
            if p0 > p1:
                p0, p1 = p1, p0

            # Skip duplicates
            pair_key = (p0, p1)
            if pair_key in seen:
                continue

            seen.add(pair_key)
            canonical_pairs.append(pair_key)

        return canonical_pairs

    def _validate_pairs(
        self,
        image_list: List[Path],
        pairs: List[Tuple[Path, Path]]
    ):
        """Validate that pair selection meets requirements."""
        from collections import Counter

        # Count pairs per image
        pair_counts = Counter()
        for p0, p1 in pairs:
            pair_counts[p0] += 1
            pair_counts[p1] += 1

        # Check minimum connectivity
        for img in image_list:
            count = pair_counts.get(img, 0)
            if count < self.config.min_pairs_per_image:
                print(
                    f"Warning: Image {img.name} has only {count} pairs "
                    f"(minimum: {self.config.min_pairs_per_image})"
                )

    def get_statistics(
        self,
        image_list: List[Path],
        pairs: List[Tuple[Path, Path]]
    ) -> Dict:
        """Get statistics about pair selection."""
        from collections import Counter

        # Count pairs per image
        pair_counts = Counter()
        for p0, p1 in pairs:
            pair_counts[p0] += 1
            pair_counts[p1] += 1

        counts = list(pair_counts.values())

        return {
            "total_images": len(image_list),
            "total_pairs": len(pairs),
            "avg_pairs_per_image": np.mean(counts) if counts else 0,
            "min_pairs_per_image": min(counts) if counts else 0,
            "max_pairs_per_image": max(counts) if counts else 0,
            "median_pairs_per_image": np.median(counts) if counts else 0,
        }
