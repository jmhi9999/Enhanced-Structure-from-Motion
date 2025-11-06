"""
ALIKED feature extractor.
"""

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import BaseFeatureExtractor, FeatureData  # noqa: E402


class ALIKEDExtractor(BaseFeatureExtractor):
    """
    ALIKED: Adaptive and Learned Interest Point Detector and Descriptor.

    Provides 128-D float descriptors optimised for LightGlue.
    """

    def _setup(self) -> None:
        """Load ALIKED model with configurable parameters."""
        try:
            from lightglue import ALIKED as LG_ALIKED
        except ImportError as exc:
            raise ImportError(
                "Please install lightglue with ALIKED support: pip install lightglue"
            ) from exc

        extra = self.config.extra
        model_name = extra.get("model_name", "aliked-n16")
        max_num_keypoints = extra.get("max_num_keypoints", self.config.max_keypoints)
        detection_threshold = extra.get("detection_threshold", 0.1)
        nms_radius = extra.get("nms_radius", 2)

        self.model = LG_ALIKED(
            model_name=model_name,
            max_num_keypoints=max_num_keypoints,
            detection_threshold=detection_threshold,
            nms_radius=nms_radius,
        ).eval()
        self.model = self.model.to(self.config.device)

    def _to_tensor(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert numpy image to model-ready tensor."""
        if image.ndim == 3:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape[:2]
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        image_tensor = image_tensor.to(self.config.device)
        return image_tensor, (h, w)

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """Extract ALIKED keypoints and descriptors."""
        import logging

        logger = logging.getLogger(__name__)

        image_tensor, image_shape = self._to_tensor(image)

        try:
            with torch.no_grad():
                prediction = self.model({"image": image_tensor})
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"ALIKED forward pass failed: {exc}")
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, self.descriptor_dim), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=image_shape,
            )

        keypoints = prediction["keypoints"][0].detach().cpu().numpy()
        descriptors = prediction["descriptors"][0].detach().cpu().numpy()
        scores = prediction["keypoint_scores"][0].detach().cpu().numpy()

        # Ensure descriptors are [N, D]
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
                "ALIKED output mismatch: keypoints=%s, descriptors=%s, scores=%s",
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
