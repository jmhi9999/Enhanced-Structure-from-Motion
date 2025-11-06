from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "timm is required for DINO embeddings. Install with `pip install timm`."
    ) from exc

from .io_utils import ensure_dir, list_image_paths, stem_from_path

logger = logging.getLogger(__name__)


def build_dino_model(device: str = "cuda") -> torch.nn.Module:
    """Instantiate a pretrained DINOv2 model."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU for DINO.")
        device = "cpu"

    device_obj = torch.device(device)
    model_name = "vit_large_patch14_dinov2.lvd142m"
    logger.info(f"Loading DINO model {model_name} on {device_obj}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device_obj)
    return model


def _extract_cls_embedding(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Extract CLS embedding from DINO model output."""
    with torch.no_grad():
        feats = model.forward_features(batch)

    if isinstance(feats, dict):
        for key in ("x_norm_clstoken", "cls_token", "last_cls_token", "pooled"):
            if key in feats:
                return feats[key]
        if "x_norm_patchtokens" in feats:
            # Average patch tokens if CLS not present
            return feats["x_norm_patchtokens"].mean(dim=1, keepdim=True)

    if isinstance(feats, torch.Tensor):
        return feats[:, 0:1, ...] if feats.ndim == 3 else feats

    raise RuntimeError("Unexpected feature output from DINO model.")


def compute_or_load_embeddings(
    img_dir: str,
    out_dir: str,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Return dict mapping image stem to cached CLS embedding."""
    embeddings_dir = ensure_dir(Path(out_dir) / "embeddings")
    model = build_dino_model(device=device)

    data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**data_cfg)
    device_tensor = next(model.parameters()).device  # type: ignore[call-overload]
    embeddings: Dict[str, np.ndarray] = {}

    for img_path in list_image_paths(img_dir):
        stem = stem_from_path(img_path)
        cache_path = embeddings_dir / f"{stem}.npy"
        if cache_path.exists():
            embeddings[stem] = np.load(cache_path)
            continue

        with Image.open(img_path) as im:
            rgb = im.convert("RGB")

        tensor = transform(rgb).unsqueeze(0).to(device_tensor)
        cls_token = _extract_cls_embedding(model, tensor)
        cls_np = cls_token.squeeze().float().cpu().numpy().astype(np.float32)
        np.save(cache_path, cls_np)
        embeddings[stem] = cls_np

    return embeddings
