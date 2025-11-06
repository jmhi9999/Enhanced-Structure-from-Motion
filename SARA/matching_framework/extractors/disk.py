"""
DISK feature extractor.
"""

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import BaseFeatureExtractor, FeatureData  # noqa: E402


class DISKExtractor(BaseFeatureExtractor):
    """
    DISK: Learning local features with policy gradient.

    Produces 128-D float descriptors well-suited for LightGlue.
    """

    def _setup(self) -> None:
        """Load DISK model with optional configuration overrides."""
        try:
            from lightglue import DISK as LG_DISK
        except ImportError as exc:
            raise ImportError(
                "Please install lightglue with DISK support: pip install lightglue"
            ) from exc

        extra = self.config.extra
        weights = extra.get("weights", "depth")
        max_keypoints = extra.get("max_keypoints", self.config.max_keypoints)
        nms_window_size = extra.get("nms_window_size", 5)
        detection_threshold = extra.get("detection_threshold", 0.0)
        pad_if_not_divisible = extra.get("pad_if_not_divisible", True)

        self.model = LG_DISK(
            weights=weights,
            max_keypoints=max_keypoints,
            nms_window_size=nms_window_size,
            detection_threshold=detection_threshold,
            pad_if_not_divisible=pad_if_not_divisible,
        ).eval()
        self.model = self.model.to(self.config.device)

    def _prepare_tensor(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert numpy image to DISK input tensor."""
        if image.ndim == 2:
            tensor = torch.from_numpy(image).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            tensor = torch.from_numpy(image[:, :, ::-1]).float() / 255.0  # BGR -> RGB
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        tensor = tensor.to(self.config.device)
        return tensor, image.shape[:2]

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """Extract DISK keypoints and descriptors."""
        import logging

        logger = logging.getLogger(__name__)

        image_tensor, image_shape = self._prepare_tensor(image)

        try:
            with torch.no_grad():
                prediction = self.model({"image": image_tensor})
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"DISK forward pass failed: {exc}")
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, self.descriptor_dim), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=image_shape,
            )

        keypoints = prediction["keypoints"][0].detach().cpu().numpy()
        descriptors = prediction["descriptors"][0].detach().cpu().numpy()
        scores = prediction["keypoint_scores"][0].detach().cpu().numpy()

        if descriptors.ndim == 2 and descriptors.shape[0] != len(keypoints):
            descriptors = descriptors.T

        num_kpts = len(keypoints)
        if num_kpts == 0:
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, self.descriptor_dim), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=image_shape,
            )

        if descriptors.shape[0] != num_kpts or scores.shape[0] != num_kpts:
            logger.warning(
                "DISK output mismatch: keypoints=%s, descriptors=%s, scores=%s",
                keypoints.shape,
                descriptors.shape,
                scores.shape,
            )
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, self.descriptor_dim), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=image_shape,
            )

        return FeatureData(
            keypoints=keypoints.astype(np.float32),
            descriptors=descriptors.astype(np.float32),
            scores=scores.astype(np.float32),
            image_shape=image_shape,
        )

    @property
    def descriptor_dim(self) -> int:
        return 128

    @property
    def descriptor_type(self) -> str:
        return "float"

    @property
    def supports_batching(self) -> bool:
        return True
