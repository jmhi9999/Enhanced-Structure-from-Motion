from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd


def list_image_paths(img_dir: str) -> List[Path]:
    """Return sorted list of image paths."""
    img_path = Path(img_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    images: List[Path] = []
    for pattern in patterns:
        images.extend(img_path.glob(pattern))
        images.extend(img_path.glob(pattern.upper()))
    return sorted(images)


def stem_from_path(path: Path) -> str:
    """Return filename stem without extension."""
    return path.stem


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pairs_csv(
    pairs: Sequence[Tuple[str, str, float, float, float]],
    output_path: Path,
) -> None:
    """Persist pair list to CSV with standard columns."""
    ensure_dir(output_path.parent)
    df = pd.DataFrame(
        pairs, columns=["i", "j", "score", "overlap", "parallax"]
    )
    df.to_csv(output_path, index=False)


def save_pairs_for_matcher(
    pairs: Iterable[Tuple[str, str, float]],
    output_path: Path,
    suffix: str = ".jpg",
) -> None:
    """Save lightweight JSONL pairing file for downstream matchers."""
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fh:
        for i, j, _ in pairs:
            fh.write(json.dumps({"image_i": f"{i}{suffix}", "image_j": f"{j}{suffix}"}) + "\n")


def normalise_feature_dict(
    raw_feats: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Ensure feature dictionary uses image stem keys and required attributes."""
    normalised: Dict[str, Dict[str, np.ndarray]] = {}
    for key, payload in raw_feats.items():
        path = Path(key)
        stem = stem_from_path(path) if path.suffix else key
        required = ("keypoints", "descriptors", "scores")
        if not all(k in payload for k in required):
            missing = [k for k in required if k not in payload]
            raise KeyError(f"Feature dict for {key} missing fields: {missing}")
        normalised[stem] = {
            "kpt": np.asarray(payload["keypoints"], dtype=np.float32),
            "desc": np.asarray(payload["descriptors"], dtype=np.float32),
            "score": np.asarray(payload["scores"], dtype=np.float32),
            "shape": tuple(payload.get("image_shape", payload.get("shape", (0, 0)))),
        }
    return normalised


def iter_feature_paths(features_dir: Path) -> Iterator[Path]:
    """Yield NPZ feature files from a directory."""
    for path in sorted(features_dir.glob("*.npz")):
        yield path
