"""
ORB feature extractor using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import BaseFeatureExtractor, ExtractorConfig, FeatureData


class ORBExtractor(BaseFeatureExtractor):
    """
    ORB (Oriented FAST and Rotated BRIEF) extractor.

    Fast binary features with rotation invariance.
    Uses Hamming distance for matching.
    """

    def _setup(self):
        """Initialize ORB detector."""
        # Get ORB-specific parameters
        n_features = self.config.extra.get("nfeatures", self.config.max_keypoints)
        scale_factor = self.config.extra.get("scaleFactor", 1.2)
        n_levels = self.config.extra.get("nlevels", 8)
        edge_threshold = self.config.extra.get("edgeThreshold", 31)
        first_level = self.config.extra.get("firstLevel", 0)
        WTA_K = self.config.extra.get("WTA_K", 2)
        score_type = self.config.extra.get("scoreType", cv2.ORB_HARRIS_SCORE)
        patch_size = self.config.extra.get("patchSize", 31)
        fast_threshold = self.config.extra.get("fastThreshold", 20)

        self.detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=WTA_K,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold
        )

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """Extract ORB features from image."""
        # Ensure grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Detect and compute
        keypoints, descriptors = self.detector.detectAndCompute(image_gray, None)

        if keypoints is None or len(keypoints) == 0:
            # No keypoints detected
            h, w = image_gray.shape
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 32), dtype=np.uint8),  # ORB: 256 bits = 32 bytes
                scores=np.empty((0,), dtype=np.float32),
                image_shape=(h, w)
            )

        # Convert to expected format
        kpts_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # ORB descriptors are already uint8
        # Keep as binary for Hamming distance matching

        h, w = image_gray.shape

        return FeatureData(
            keypoints=kpts_xy,
            descriptors=descriptors,
            scores=scores,
            image_shape=(h, w)
        )

    @property
    def descriptor_dim(self) -> int:
        return 32  # 256 bits = 32 bytes

    @property
    def descriptor_type(self) -> str:
        return "binary"

    @property
    def supports_batching(self) -> bool:
        return False  # OpenCV ORB doesn't support GPU batching
