"""
Nearest Neighbor matcher with ratio test.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from matchers.base import BaseMatcher, MatcherConfig, MatchData
from extractors.base import FeatureData


class NNMatcher(BaseMatcher):
    """
    Nearest Neighbor matcher with Lowe's ratio test.

    Supports both float (L2/cosine) and binary (Hamming) descriptors.
    """

    def _setup(self):
        """Initialize matcher."""
        # Determine distance metric based on descriptor type
        if self.descriptor_type == "binary":
            import cv2
            self.metric = "hamming"
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            import cv2
            # For learned descriptors, use L2 distance
            self.metric = "l2"
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def _match_impl(
        self,
        features0: FeatureData,
        features1: FeatureData
    ) -> MatchData:
        """Match using nearest neighbor + ratio test."""
        desc0 = features0.descriptors
        desc1 = features1.descriptors

        if len(desc0) == 0 or len(desc1) == 0:
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

        # KNN matching (k=2 for ratio test)
        matches = self.matcher.knnMatch(desc0, desc1, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < self.config.distance_threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) == 0:
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

        # Mutual check if enabled
        if self.config.mutual_check:
            matches_reverse = self.matcher.knnMatch(desc1, desc0, k=1)
            reverse_map = {m[0].trainIdx: m[0].queryIdx for m in matches_reverse if len(m) > 0}

            filtered_matches = []
            for m in good_matches:
                if m.trainIdx in reverse_map and reverse_map[m.trainIdx] == m.queryIdx:
                    filtered_matches.append(m)
            good_matches = filtered_matches

        if len(good_matches) == 0:
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

        # Convert to arrays
        matches0 = np.array([m.queryIdx for m in good_matches], dtype=np.int32)
        matches1 = np.array([m.trainIdx for m in good_matches], dtype=np.int32)

        # Convert distances to scores (lower distance = higher score)
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        # Normalize to [0, 1] where 1 is best match
        max_dist = distances.max() if len(distances) > 0 else 1.0
        scores = 1.0 - (distances / (max_dist + 1e-8))

        return MatchData(
            matches0=matches0,
            matches1=matches1,
            scores=scores
        )

    @property
    def supports_batching(self) -> bool:
        return False
