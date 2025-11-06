"""
Exhaustive (brute force) pair selection.
"""

from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pair_selectors.base import BasePairSelector, PairSelectorConfig


class ExhaustivePairSelector(BasePairSelector):
    """
    Exhaustive pair selection - match all possible pairs.

    Good for small datasets (<100 images).
    O(n²) complexity.
    """

    def _setup(self):
        """No setup needed for exhaustive selection."""
        pass

    def _select_impl(
        self,
        image_list: List[Path],
        features: Dict[Path, any] = None
    ) -> List[Tuple[Path, Path]]:
        """Select all possible pairs."""
        pairs = []

        n = len(image_list)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((image_list[i], image_list[j]))

        return pairs
