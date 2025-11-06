"""
Base class for feature extractors.
Inspired by hloc architecture with improvements for flexibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class ExtractorConfig:
    """Configuration for feature extractors."""

    # Common parameters
    max_keypoints: int = 4096
    detection_threshold: float = 0.005
    edge_threshold: float = 10.0

    # Device
    device: str = "cuda"

    # Performance
    batch_size: int = 1
    num_workers: int = 4

    # Output
    resize_max: Optional[int] = None  # Resize image before extraction
    grayscale: bool = True

    # Extractor-specific
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


@dataclass
class FeatureData:
    """Container for extracted features."""

    keypoints: np.ndarray      # [N, 2] (x, y) coordinates
    descriptors: np.ndarray    # [N, D] descriptors
    scores: np.ndarray         # [N] detection scores
    image_shape: Tuple[int, int]  # (height, width)

    def __len__(self) -> int:
        return len(self.keypoints)

    def validate(self) -> bool:
        """Validate feature data consistency."""
        n = len(self.keypoints)

        # Allow empty features (no keypoints found)
        if n == 0:
            return (
                len(self.descriptors) == 0 and
                len(self.scores) == 0
            )

        # Check consistency for non-empty features
        return (
            len(self.descriptors) == n and
            len(self.scores) == n and
            self.keypoints.shape[1] == 2 and
            self.descriptors.ndim == 2
        )


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All extractors must implement:
    - _extract_impl(): Core extraction logic
    - descriptor_dim: Descriptor dimensionality
    - descriptor_type: 'float' or 'binary'
    - supports_batching: Whether batch extraction is supported
    """

    def __init__(self, config: ExtractorConfig):
        self.config = config
        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize extractor (load model, set parameters, etc.)."""
        pass

    @abstractmethod
    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """
        Core extraction implementation.

        Args:
            image: Input image [H, W] or [H, W, 3] in uint8 [0, 255]

        Returns:
            FeatureData object with keypoints, descriptors, scores
        """
        pass

    @property
    @abstractmethod
    def descriptor_dim(self) -> int:
        """Descriptor dimensionality (e.g., 128 for SIFT, 256 for SuperPoint)."""
        pass

    @property
    @abstractmethod
    def descriptor_type(self) -> str:
        """Descriptor type: 'float' or 'binary'."""
        pass

    @property
    @abstractmethod
    def supports_batching(self) -> bool:
        """Whether this extractor supports batch processing."""
        pass

    @property
    def name(self) -> str:
        """Extractor name (class name by default)."""
        return self.__class__.__name__.replace("Extractor", "").lower()

    def extract(self, image_path: Path) -> FeatureData:
        """
        Extract features from a single image.

        Args:
            image_path: Path to input image

        Returns:
            FeatureData object
        """
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to grayscale if needed
        if self.config.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize if needed
        if self.config.resize_max is not None:
            h, w = image.shape[:2]
            max_dim = max(h, w)
            if max_dim > self.config.resize_max:
                scale = self.config.resize_max / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

        # Extract features
        features = self._extract_impl(image)

        # Validate
        if not features.validate():
            raise RuntimeError(f"Invalid feature data from {self.name}: inconsistent shapes")

        # Warn if no features found (but don't fail)
        if len(features) == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"No features detected in {image_path.name} by {self.name}")

        # Limit number of keypoints
        if len(features) > self.config.max_keypoints:
            features = self._select_top_k(features, self.config.max_keypoints)

        return features

    def extract_batch(self, image_paths: List[Path]) -> Dict[Path, FeatureData]:
        """
        Extract features from multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            Dictionary mapping paths to FeatureData
        """
        from tqdm import tqdm
        import logging
        logger = logging.getLogger(__name__)

        results = {}
        failed_count = 0
        empty_count = 0

        for path in tqdm(image_paths, desc=f"Extracting {self.name}"):
            try:
                features = self.extract(path)

                # Track empty features
                if len(features) == 0:
                    empty_count += 1

                results[path] = features

            except Exception as e:
                logger.warning(f"Failed to extract features from {path.name}: {e}")
                failed_count += 1
                continue

        # Summary
        total = len(image_paths)
        success = len(results)
        logger.info(
            f"Feature extraction complete: {success}/{total} images successful, "
            f"{empty_count} with no features, {failed_count} failed"
        )

        if failed_count > total * 0.5:
            logger.error(
                f"More than 50% of images failed feature extraction. "
                f"Check image quality, format, or extractor configuration."
            )

        return results

    def _select_top_k(self, features: FeatureData, k: int) -> FeatureData:
        """Select top-k keypoints by score."""
        if len(features) <= k:
            return features

        indices = np.argsort(features.scores)[::-1][:k]

        return FeatureData(
            keypoints=features.keypoints[indices],
            descriptors=features.descriptors[indices],
            scores=features.scores[indices],
            image_shape=features.image_shape
        )

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration for matcher initialization."""
        return {
            "name": self.name,
            "descriptor_dim": self.descriptor_dim,
            "descriptor_type": self.descriptor_type,
            "supports_batching": self.supports_batching,
        }
