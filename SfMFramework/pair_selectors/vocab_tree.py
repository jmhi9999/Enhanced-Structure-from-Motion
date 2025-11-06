"""
Vocabulary-tree based pair selector.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from pair_selectors.base import BasePairSelector, PairSelectorConfig

logger = logging.getLogger(__name__)


class VocabTreePairSelector(BasePairSelector):
    """
    Pair selector that uses the GPUVocabularyTree retrieval backend to choose
    geometrically promising image pairs.
    """

    def _setup(self) -> None:
        try:
            from sfm.core.gpu_vocabulary_tree import GPUVocabularyTree  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - defensive
            logger.warning(
                "GPUVocabularyTree unavailable (%s); vocab_tree selector will fall back to exhaustive pairing.",
                exc,
            )
            self._tree_cls = None
        else:
            self._tree_cls = GPUVocabularyTree

        self._tree = None
        self._min_score_threshold = self.config.extra.get("min_score_threshold", 0.01)
        self._ensure_connectivity = self.config.extra.get("ensure_connectivity", True)

    def _ensure_tree(self) -> None:
        """Instantiate the vocabulary tree once."""
        if self._tree is not None or self._tree_cls is None:
            return

        import torch

        device = (
            torch.device(self.config.device)
            if self.config.device != "cpu" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        tree_config = {
            "max_descriptors_per_image": self.config.extra.get(
                "max_descriptors_per_image", 4000
            ),
            "max_vocab_descriptors": self.config.extra.get(
                "max_vocab_descriptors", 800000
            ),
            "vocab_size": self.config.extra.get("vocab_size", 10000),
            "vocab_depth": self.config.extra.get("vocab_depth", 6),
            "vocab_branching_factor": self.config.extra.get(
                "vocab_branching_factor", 10
            ),
        }

        self._tree = self._tree_cls(device=device, config=tree_config)

    def _select_impl(
        self,
        image_list: List[Path],
        features: Dict[Path, "FeatureData"] = None,
    ) -> List[Tuple[Path, Path]]:
        if len(image_list) < 2:
            return []

        if self._tree_cls is None:
            logger.info("Falling back to exhaustive pairing (vocabulary tree unavailable).")
            return list(combinations(image_list, 2))

        if features is None or not features:
            logger.warning(
                "VocabTree selector requires feature descriptors; falling back to exhaustive pairs."
            )
            return list(combinations(image_list, 2))

        converted = self._convert_features(image_list, features)
        if len(converted) < 2:
            logger.warning(
                "Insufficient features for vocabulary tree (got %d images); using exhaustive pairs.",
                len(converted),
            )
            return list(combinations(image_list, 2))

        self._ensure_tree()

        if self._tree is None:
            logger.info("Vocabulary tree instantiation failed; returning exhaustive pairs.")
            return list(combinations(image_list, 2))

        pair_strings = self._tree.get_image_pairs_for_matching(
            converted,
            max_pairs_per_image=self.config.max_pairs_per_image,
            min_score_threshold=self._min_score_threshold,
            ensure_connectivity=self._ensure_connectivity,
        )

        return self._to_path_pairs(pair_strings)

    def _convert_features(
        self,
        image_list: Iterable[Path],
        features: Dict[Path, "FeatureData"],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        converted = {}
        missing = 0

        for path in image_list:
            feature = features.get(path)
            if feature is None:
                missing += 1
                continue

            descriptors = np.asarray(feature.descriptors, dtype=np.float32)
            if descriptors.size == 0:
                missing += 1
                continue

            converted[str(path)] = {
                "descriptors": descriptors,
                "keypoints": np.asarray(feature.keypoints, dtype=np.float32),
                "scores": np.asarray(feature.scores, dtype=np.float32),
            }

        if missing:
            logger.debug(
                "Vocabulary tree skipped %d images without usable features.", missing
            )

        return converted

    @staticmethod
    def _to_path_pairs(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[Path, Path]]:
        unique_pairs = []
        seen = set()
        for left, right in pairs:
            p0 = Path(left)
            p1 = Path(right)
            key = tuple(sorted((p0, p1)))
            if key in seen:
                continue
            seen.add(key)
            unique_pairs.append(key)
        return unique_pairs
