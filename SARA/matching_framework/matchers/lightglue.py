"""
LightGlue matcher.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from matchers.base import BaseMatcher, MatcherConfig, MatchData
from extractors.base import FeatureData


class LightGlueMatcher(BaseMatcher):
    """
    LightGlue: Local Feature Matching at Light Speed.

    Attention-based matcher for learned descriptors (SuperPoint, ALIKED, DISK).
    """

    def _setup(self):
        """Initialize LightGlue model."""
        try:
            from lightglue import LightGlue
        except ImportError:
            raise ImportError("Please install lightglue: pip install lightglue")

        # Determine feature type from extractor name
        extractor_name = self.extractor_config.get("name", "superpoint")

        # Map extractor names to LightGlue features
        feature_map = {
            "superpoint": "superpoint",
            "aliked": "aliked",
            "disk": "disk",
            "sift": "sift",  # LightGlue doesn't support SIFT well, but we can try
        }

        features = feature_map.get(extractor_name, "superpoint")

        self.model = LightGlue(
            features=features,
            depth_confidence=self.config.extra.get("depth_confidence", 0.95),
            width_confidence=self.config.extra.get("width_confidence", 0.99),
        ).eval()

        self.model = self.model.to(self.config.device)

    def _match_impl(
        self,
        features0: FeatureData,
        features1: FeatureData
    ) -> MatchData:
        """Match using LightGlue."""
        if len(features0) == 0 or len(features1) == 0:
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

        # Ensure numpy arrays (convert from list if needed)
        kpts0_np = np.asarray(features0.keypoints, dtype=np.float32)
        kpts1_np = np.asarray(features1.keypoints, dtype=np.float32)
        desc0_np = np.asarray(features0.descriptors, dtype=np.float32)
        desc1_np = np.asarray(features1.descriptors, dtype=np.float32)

        # Prepare input for LightGlue
        kpts0 = torch.from_numpy(kpts0_np).float().unsqueeze(0)  # [1, N, 2]
        kpts1 = torch.from_numpy(kpts1_np).float().unsqueeze(0)

        desc0 = torch.from_numpy(desc0_np).float().unsqueeze(0)  # [1, N, D]
        desc1 = torch.from_numpy(desc1_np).float().unsqueeze(0)

        # Move to device
        data = {
            "image0": {
                "keypoints": kpts0.to(self.config.device),
                "descriptors": desc0.to(self.config.device),
            },
            "image1": {
                "keypoints": kpts1.to(self.config.device),
                "descriptors": desc1.to(self.config.device),
            }
        }

        try:
            # Match
            with torch.no_grad():
                pred = self.model(data)

            # Extract matches
            matches_tensor = pred["matches"]  # [S, 2] or [B, S, 2] or list
            scores_tensor = pred.get("scores", None)  # [S] or [B, S] or list

            # Convert to tensor if list
            if isinstance(matches_tensor, list):
                if len(matches_tensor) == 0:
                    return MatchData(
                        matches0=np.array([], dtype=np.int32),
                        matches1=np.array([], dtype=np.int32),
                        scores=np.array([], dtype=np.float32)
                    )
                matches_tensor = torch.stack(matches_tensor) if isinstance(matches_tensor[0], torch.Tensor) else torch.tensor(matches_tensor)

            if scores_tensor is not None and isinstance(scores_tensor, list):
                scores_tensor = torch.stack(scores_tensor) if (scores_tensor and isinstance(scores_tensor[0], torch.Tensor)) else torch.tensor(scores_tensor)

            # Handle batch dimension if present
            if hasattr(matches_tensor, 'ndim') and matches_tensor.ndim == 3:
                matches_tensor = matches_tensor[0]  # Take first batch
            if scores_tensor is not None and hasattr(scores_tensor, 'ndim') and scores_tensor.ndim == 2:
                scores_tensor = scores_tensor[0]

            if len(matches_tensor) == 0:
                return MatchData(
                    matches0=np.array([], dtype=np.int32),
                    matches1=np.array([], dtype=np.int32),
                    scores=np.array([], dtype=np.float32)
                )

            # Convert to numpy
            matches0 = matches_tensor[:, 0].cpu().numpy().astype(np.int32)
            matches1 = matches_tensor[:, 1].cpu().numpy().astype(np.int32)

            if scores_tensor is not None:
                scores = scores_tensor.cpu().numpy().astype(np.float32)
            else:
                scores = np.ones(len(matches0), dtype=np.float32)

            return MatchData(
                matches0=matches0,
                matches1=matches1,
                scores=scores
            )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"LightGlue matching failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            # Return empty matches instead of crashing
            return MatchData(
                matches0=np.array([], dtype=np.int32),
                matches1=np.array([], dtype=np.int32),
                scores=np.array([], dtype=np.float32)
            )

    @property
    def supports_batching(self) -> bool:
        return True
