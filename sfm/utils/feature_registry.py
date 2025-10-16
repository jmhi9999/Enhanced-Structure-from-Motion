"""Utility helpers to manage dynamic keypoints/descriptors per image."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class FeatureRegistry:
    """Maintain a mutable registry of keypoints/descriptors for an image."""

    def __init__(self, feature_entry: Dict[str, np.ndarray]):
        self.entry = feature_entry

        keypoints = feature_entry.get("keypoints", np.zeros((0, 2), dtype=np.float32))
        descriptors = feature_entry.get("descriptors", np.zeros((0, 128), dtype=np.float32))
        scores = feature_entry.get("scores", np.zeros((len(keypoints),), dtype=np.float32))

        self.descriptor_dim = descriptors.shape[1] if descriptors.ndim == 2 and descriptors.size > 0 else 128

        self.keypoints = [tuple(map(float, pt)) for pt in keypoints]
        self.descriptors = [desc.astype(np.float32) for desc in descriptors]
        self.scores = [float(s) for s in scores]

        self.lookup = {self._quantize(pt): idx for idx, pt in enumerate(self.keypoints)}

    def register(
        self,
        coord: Tuple[float, float],
        descriptor: Optional[np.ndarray] = None,
        score: float = 0.5,
    ) -> int:
        """Register a keypoint if absent, returning its index."""
        key = self._quantize(coord)
        if key in self.lookup:
            return self.lookup[key]

        idx = len(self.keypoints)
        self.lookup[key] = idx
        self.keypoints.append((float(coord[0]), float(coord[1])))

        if descriptor is None:
            descriptor = np.zeros((self.descriptor_dim,), dtype=np.float32)
        else:
            descriptor = descriptor.astype(np.float32)

        if descriptor.shape[0] != self.descriptor_dim:
            descriptor = np.resize(descriptor, (self.descriptor_dim,))

        self.descriptors.append(descriptor)
        self.scores.append(float(score))

        return idx

    def finalize(self) -> None:
        """Write back arrays to the underlying feature entry."""
        self.entry["keypoints"] = np.asarray(self.keypoints, dtype=np.float32)
        self.entry["descriptors"] = np.asarray(self.descriptors, dtype=np.float32)
        self.entry["scores"] = np.asarray(self.scores, dtype=np.float32)

    @staticmethod
    def _quantize(coord: Tuple[float, float], precision: int = 1) -> Tuple[int, int]:
        """Quantize coordinate for robust hashing."""
        scale = 10 ** precision
        return (int(round(coord[0] * scale)), int(round(coord[1] * scale)))
