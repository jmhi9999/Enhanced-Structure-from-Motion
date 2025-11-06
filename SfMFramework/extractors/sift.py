"""
SIFT feature extractor using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import BaseFeatureExtractor, ExtractorConfig, FeatureData


class SIFTExtractor(BaseFeatureExtractor):
    """
    SIFT (Scale-Invariant Feature Transform) extractor.

    Classic hand-crafted features with excellent robustness.
    Uses L2 distance for matching.
    """

    def _setup(self):
        """Initialize SIFT detector."""
        # Get SIFT-specific parameters
        n_features = self.config.extra.get("nfeatures", self.config.max_keypoints)
        n_octave_layers = self.config.extra.get("nOctaveLayers", 3)
        contrast_threshold = self.config.extra.get(
            "contrastThreshold", self.config.detection_threshold
        )
        edge_threshold = self.config.extra.get(
            "edgeThreshold", self.config.edge_threshold
        )
        sigma = self.config.extra.get("sigma", 1.6)

        self.detector = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """Extract SIFT features from image."""
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
                descriptors=np.empty((0, 128), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=(h, w)
            )

        # Convert to expected format
        kpts_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)

        # SIFT descriptors are uint8 [0, 255], convert to float32 [0, 1]
        # This normalization helps with matching
        descriptors = descriptors.astype(np.float32) / 255.0

        h, w = image_gray.shape

        return FeatureData(
            keypoints=kpts_xy,
            descriptors=descriptors,
            scores=scores,
            image_shape=(h, w)
        )

    @property
    def descriptor_dim(self) -> int:
        return 128

    @property
    def descriptor_type(self) -> str:
        return "float"

    @property
    def supports_batching(self) -> bool:
        return False  # OpenCV SIFT doesn't support GPU batching
