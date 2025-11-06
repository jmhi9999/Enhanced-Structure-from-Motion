"""
Base class for feature matchers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import FeatureData


@dataclass
class MatcherConfig:
    """Configuration for matchers."""

    # Matching parameters
    distance_threshold: float = 0.8  # Lowe's ratio or distance threshold (0.8 matches COLMAP default)
    mutual_check: bool = True
    max_matches: Optional[int] = None

    # Device
    device: str = "cuda"

    # Matcher-specific
    extra: Dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class MatchData:
    """Container for match results."""

    matches0: np.ndarray       # [M] indices into keypoints0
    matches1: np.ndarray       # [M] indices into keypoints1
    scores: np.ndarray         # [M] match confidence scores

    def __len__(self) -> int:
        return len(self.matches0)

    def validate(self) -> bool:
        """Validate match data consistency."""
        return (
            len(self.matches0) == len(self.matches1) == len(self.scores) and
            self.matches0.dtype in [np.int32, np.int64] and
            self.matches1.dtype in [np.int32, np.int64]
        )

    def to_keypoints(
        self,
        kpts0: np.ndarray,
        kpts1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert match indices to matched keypoint coordinates."""
        return kpts0[self.matches0], kpts1[self.matches1]


class BaseMatcher(ABC):
    """
    Abstract base class for feature matchers.

    All matchers must implement:
    - _match_impl(): Core matching logic
    - supports_batching: Whether batch matching is supported
    """

    def __init__(self, config: MatcherConfig, extractor_config: Dict):
        """
        Initialize matcher.

        Args:
            config: Matcher configuration
            extractor_config: Configuration from feature extractor (descriptor dim, type, etc.)
        """
        self.config = config
        self.extractor_config = extractor_config

        # Extract key information from extractor
        self.descriptor_dim = extractor_config["descriptor_dim"]
        self.descriptor_type = extractor_config["descriptor_type"]

        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize matcher (load model, set parameters, etc.)."""
        pass

    @abstractmethod
    def _match_impl(
        self,
        features0: FeatureData,
        features1: FeatureData
    ) -> MatchData:
        """
        Core matching implementation.

        Args:
            features0: Features from first image
            features1: Features from second image

        Returns:
            MatchData object with match indices and scores
        """
        pass

    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """Whether this matcher supports batch processing."""
        pass

    @property
    def name(self) -> str:
        """Matcher name."""
        return self.__class__.__name__.replace("Matcher", "").lower()

    def match(
        self,
        features0: FeatureData,
        features1: FeatureData
    ) -> MatchData:
        """
        Match features between two images.

        Args:
            features0: Features from first image
            features1: Features from second image

        Returns:
            MatchData object
        """
        # Validate inputs
        if not features0.validate() or not features1.validate():
            raise ValueError("Invalid feature data")

        # Handle empty features
        if len(features0) == 0 or len(features1) == 0:
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

        # Check descriptor compatibility
        if features0.descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"Descriptor dimension mismatch: expected {self.descriptor_dim}, "
                f"got {features0.descriptors.shape[1]}"
            )

        if features1.descriptors.shape[1] != self.descriptor_dim:
            raise ValueError(
                f"Descriptor dimension mismatch: expected {self.descriptor_dim}, "
                f"got {features1.descriptors.shape[1]}"
            )

        # Match
        matches = self._match_impl(features0, features1)

        # Validate
        if not matches.validate():
            raise RuntimeError(f"Invalid match data from {self.name}")

        # Limit matches if needed
        if self.config.max_matches and len(matches) > self.config.max_matches:
            matches = self._select_top_k(matches, self.config.max_matches)

        return matches

    def match_pairs(
        self,
        features_dict: Dict[Path, FeatureData],
        pairs: List[Tuple[Path, Path]]
    ) -> Dict[Tuple[Path, Path], MatchData]:
        """
        Match features for multiple image pairs.

        Args:
            features_dict: Dictionary mapping paths to features
            pairs: List of image pairs to match

        Returns:
            Dictionary mapping pairs to matches
        """
        from tqdm import tqdm
        import logging
        logger = logging.getLogger(__name__)

        results = {}
        failed_count = 0

        for path0, path1 in tqdm(pairs, desc=f"Matching {self.name}"):
            if path0 not in features_dict or path1 not in features_dict:
                logger.warning(f"Missing features for pair ({path0.name}, {path1.name})")
                failed_count += 1
                continue

            try:
                matches = self.match(features_dict[path0], features_dict[path1])
                results[(path0, path1)] = matches
            except Exception as e:
                logger.warning(f"Failed to match ({path0.name}, {path1.name}): {e}")
                # Debug: Print more details
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
                logger.debug(f"Features0 shape: kpts={features_dict[path0].keypoints.shape}, "
                           f"desc={features_dict[path0].descriptors.shape}")
                logger.debug(f"Features1 shape: kpts={features_dict[path1].keypoints.shape}, "
                           f"desc={features_dict[path1].descriptors.shape}")
                failed_count += 1
                continue

        logger.info(
            f"Matching complete: {len(results)}/{len(pairs)} pairs matched successfully, "
            f"{failed_count} failed"
        )

        if failed_count > len(pairs) * 0.5:
            logger.error(
                f"More than 50% of pairs failed matching. "
                f"Check feature compatibility and matcher configuration."
            )

        return results

    def _select_top_k(self, matches: MatchData, k: int) -> MatchData:
        """Select top-k matches by score."""
        if len(matches) <= k:
            return matches

        indices = np.argsort(matches.scores)[::-1][:k]

        return MatchData(
            matches0=matches.matches0[indices],
            matches1=matches.matches1[indices],
            scores=matches.scores[indices]
        )

    def geometric_verification(
        self,
        features0: FeatureData,
        features1: FeatureData,
        matches: MatchData,
        method: str = "ransac",
        threshold: float = 1.0
    ) -> Tuple[MatchData, np.ndarray]:
        """
        Perform geometric verification on matches.

        Args:
            features0: Features from first image
            features1: Features from second image
            matches: Initial matches
            method: "ransac" or "magsac"
            threshold: Reprojection error threshold in pixels

        Returns:
            Tuple of (verified_matches, fundamental_matrix)
        """
        import cv2

        kpts0, kpts1 = matches.to_keypoints(features0.keypoints, features1.keypoints)

        if len(kpts0) < 8:
            # Not enough points for fundamental matrix
            return matches, np.eye(3)

        # Estimate fundamental matrix
        method_flag = cv2.FM_RANSAC if method == "ransac" else cv2.USAC_MAGSAC
        F, inlier_mask = cv2.findFundamentalMat(
            kpts0,
            kpts1,
            method=method_flag,
            ransacReprojThreshold=threshold,
            confidence=0.9999,
            maxIters=10000
        )

        if F is None or inlier_mask is None:
            return matches, np.eye(3)

        inlier_mask = inlier_mask.ravel().astype(bool)

        # Filter matches
        verified_matches = MatchData(
            matches0=matches.matches0[inlier_mask],
            matches1=matches.matches1[inlier_mask],
            scores=matches.scores[inlier_mask]
        )

        return verified_matches, F
