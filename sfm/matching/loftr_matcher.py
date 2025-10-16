"""LoFTR matcher wrapper with optional attention guidance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

try:
    from kornia.feature import LoFTR  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency enforced in requirements
    raise ImportError(
        "LoFTR not available. Install kornia>=0.6.12 to use the LoFTR matcher."
    ) from exc


@dataclass
class LoFTROptions:
    """Configuration for LoFTRMatcher."""

    pretrained: str = "outdoor"
    max_long_edge: int = 832
    lowres_long_edge: int = 448
    use_attention_guidance: bool = True
    attention_boost: float = 0.5  # Blend factor for attention-guided intensity


class LoFTRMatcher:
    """Detector-free matcher built on top of kornia LoFTR."""

    def __init__(self, device: torch.device, options: Optional[LoFTROptions] = None):
        self.device = device
        self.opt = options or LoFTROptions()
        self.model = LoFTR(pretrained=self.opt.pretrained).to(self.device).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    def match(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        attention0: Optional[np.ndarray] = None,
        attention1: Optional[np.ndarray] = None,
        low_resolution: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Compute matches between two images."""
        max_edge = self.opt.lowres_long_edge if low_resolution else self.opt.max_long_edge

        tensor0, scale0 = self._prepare_image(image0, attention0, max_edge)
        tensor1, scale1 = self._prepare_image(image1, attention1, max_edge)

        with torch.no_grad():
            out = self.model(
                {
                    "image0": tensor0.to(self.device),
                    "image1": tensor1.to(self.device),
                }
            )

        kpts0 = out["keypoints0"].cpu().numpy()
        kpts1 = out["keypoints1"].cpu().numpy()
        confidence = out["confidence"].cpu().numpy()

        # Rescale to original resolution
        kpts0 /= scale0
        kpts1 /= scale1

        return {
            "keypoints0": kpts0.astype(np.float32),
            "keypoints1": kpts1.astype(np.float32),
            "confidence": confidence.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _prepare_image(
        self,
        image: np.ndarray,
        attention_map: Optional[np.ndarray],
        max_long_edge: int,
    ) -> Tuple[torch.Tensor, float]:
        """Convert image to normalized grayscale tensor and resize."""
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = gray.astype(np.float32) / 255.0

        h, w = gray.shape[:2]
        scale = 1.0

        if max(h, w) > max_long_edge:
            scale = max_long_edge / max(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w

        if attention_map is not None and self.opt.use_attention_guidance:
            attn_resized = cv2.resize(
                attention_map,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )
            attn_norm = attn_resized - attn_resized.min()
            if attn_norm.max() > 0:
                attn_norm /= attn_norm.max()
            gray = gray * (1.0 - self.opt.attention_boost) + gray * attn_norm * self.opt.attention_boost

        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()
        return tensor, scale
