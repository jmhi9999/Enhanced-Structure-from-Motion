"""
SuperPoint feature extractor.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from extractors.base import BaseFeatureExtractor, ExtractorConfig, FeatureData


class SuperPointExtractor(BaseFeatureExtractor):
    """
    SuperPoint: Self-Supervised Interest Point Detection and Description.

    Learning-based detector with 256-dim descriptors optimized for cosine similarity.
    """

    def _setup(self):
        """Initialize SuperPoint model."""
        try:
            from lightglue import SuperPoint as SP
            self.model = SP(max_num_keypoints=self.config.max_keypoints).eval()
            self.model = self.model.to(self.config.device)
        except ImportError:
            raise ImportError("Please install lightglue: pip install lightglue")

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        """Extract SuperPoint features."""
        import torch
        import logging
        logger = logging.getLogger(__name__)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = image.shape

        try:
            # Normalize to [0, 1] and add batch/channel dims
            image_tensor = torch.from_numpy(image).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            image_tensor = image_tensor.to(self.config.device)

            # Extract
            with torch.no_grad():
                pred = self.model({"image": image_tensor})
        except Exception as e:
            logger.error(f"SuperPoint model forward failed: {e}")
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 256), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=(h, w)
            )

        try:
            # Convert to numpy
            keypoints = pred["keypoints"][0].cpu().numpy()  # Should be [N, 2]
            descriptors = pred["descriptors"][0].cpu().numpy()  # Should be [256, N]
            scores = pred["keypoint_scores"][0].cpu().numpy()  # Should be [N]

            # Ensure descriptors are transposed correctly
            # SuperPoint returns descriptors as [D, N], we need [N, D]
            if descriptors.ndim == 2:
                if descriptors.shape[0] == 256:
                    # Descriptors are [256, N], transpose to [N, 256]
                    descriptors = descriptors.T
                elif descriptors.shape[1] != 256:
                    logger.warning(f"Unexpected descriptor shape: {descriptors.shape}, expected [N, 256] or [256, N]")
                    # Try to fix if possible
                    if descriptors.shape[0] != 256 and descriptors.shape[1] != 256:
                        # Neither dimension is 256, can't fix
                        return FeatureData(
                            keypoints=np.empty((0, 2), dtype=np.float32),
                            descriptors=np.empty((0, 256), dtype=np.float32),
                            scores=np.empty((0,), dtype=np.float32),
                            image_shape=(h, w)
                        )
            elif descriptors.ndim == 1:
                # Single descriptor, reshape
                if len(descriptors) == 256:
                    descriptors = descriptors.reshape(1, -1)
                else:
                    logger.error(f"Invalid single descriptor size: {len(descriptors)}, expected 256")
                    return FeatureData(
                        keypoints=np.empty((0, 2), dtype=np.float32),
                        descriptors=np.empty((0, 256), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        image_shape=(h, w)
                    )

            # Validate shapes before creating FeatureData
            n_kpts = len(keypoints)
            n_desc = len(descriptors)
            n_scores = len(scores)

            if n_kpts != n_desc or n_kpts != n_scores:
                logger.warning(
                    f"Shape mismatch: keypoints={keypoints.shape} ({n_kpts}), "
                    f"descriptors={descriptors.shape} ({n_desc}), "
                    f"scores={scores.shape} ({n_scores})"
                )
                # Return empty features instead of crashing
                return FeatureData(
                    keypoints=np.empty((0, 2), dtype=np.float32),
                    descriptors=np.empty((0, 256), dtype=np.float32),
                    scores=np.empty((0,), dtype=np.float32),
                    image_shape=(h, w)
                )

            # Handle empty features (no keypoints detected)
            if len(keypoints) == 0:
                keypoints = np.empty((0, 2), dtype=np.float32)
                descriptors = np.empty((0, 256), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
            else:
                # Ensure correct shapes
                if keypoints.ndim == 1:
                    if len(keypoints) == 2:
                        keypoints = keypoints.reshape(1, -1)
                    else:
                        logger.error(f"Invalid single keypoint size: {len(keypoints)}, expected 2")
                        return FeatureData(
                            keypoints=np.empty((0, 2), dtype=np.float32),
                            descriptors=np.empty((0, 256), dtype=np.float32),
                            scores=np.empty((0,), dtype=np.float32),
                            image_shape=(h, w)
                        )

                if keypoints.shape[1] != 2:
                    logger.error(f"Invalid keypoint shape: {keypoints.shape}, expected [N, 2]")
                    return FeatureData(
                        keypoints=np.empty((0, 2), dtype=np.float32),
                        descriptors=np.empty((0, 256), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        image_shape=(h, w)
                    )

            return FeatureData(
                keypoints=keypoints.astype(np.float32),
                descriptors=descriptors.astype(np.float32),
                scores=scores.astype(np.float32),
                image_shape=(h, w)
            )

        except Exception as e:
            logger.error(f"Error processing SuperPoint output: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return FeatureData(
                keypoints=np.empty((0, 2), dtype=np.float32),
                descriptors=np.empty((0, 256), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                image_shape=(h, w)
            )

    @property
    def descriptor_dim(self) -> int:
        return 256

    @property
    def descriptor_type(self) -> str:
        return "float"

    @property
    def supports_batching(self) -> bool:
        return True  # SuperPoint supports batch processing
