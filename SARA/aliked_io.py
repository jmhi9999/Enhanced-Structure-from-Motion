from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from .io_utils import ensure_dir, iter_feature_paths, list_image_paths, stem_from_path


def load_aliked_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load ALIKED feature file with standardised keys."""
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"ALIKED npz not found: {npz_path}")

    with np.load(path, allow_pickle=False) as data:
        kpt = data["keypoints"].astype(np.float32)
        desc = data["descriptors"].astype(np.float32)
        score = data["scores"].astype(np.float32)
        shape = tuple(data["image_shape"]) if "image_shape" in data else tuple(data.get("shape", (0, 0)))

    return {"kpt": kpt, "desc": desc, "score": score, "shape": shape}


def _normalise_stem(value: str) -> str:
    """Return a safe stem string from an arbitrary identifier."""
    path = Path(value)
    return path.stem if path.suffix else str(value)


def ensure_aliked_features(
    img_dir: str,
    out_dir: str,
    allowed_stems: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Ensure ALIKED features are available for all images.

    Currently expects existing npz caches in ``out_dir/features``.
    """
    features_dir = ensure_dir(Path(out_dir) / "features")
    if allowed_stems is not None:
        image_stems = {_normalise_stem(stem) for stem in allowed_stems}
    else:
        image_stems = {stem_from_path(p) for p in list_image_paths(img_dir)}

    features: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_path in iter_feature_paths(features_dir):
        stem = stem_from_path(npz_path)
        if stem not in image_stems:
            continue
        features[stem] = load_aliked_npz(str(npz_path))

    missing = image_stems - set(features.keys())
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Missing ALIKED features for {len(missing)} images: {missing_list}"
        )

    return features
