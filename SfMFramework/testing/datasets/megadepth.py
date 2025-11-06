"""MegaDepth dataset loader."""

import json
from pathlib import Path
from typing import List
import numpy as np

from .base import BaseDataset, DatasetSample


class MegaDepthDataset(BaseDataset):
    """MegaDepth benchmark dataset.

    Expected structure:
        megadepth/
        ├── scene_info/
        │   ├── scene0000.npz
        │   └── ...
        └── pairs.json
    """

    def _load_samples(self) -> List[DatasetSample]:
        """Load MegaDepth samples."""
        samples = []

        # Load pairs file
        pairs_file = self.root_path / "pairs.json"
        if not pairs_file.exists():
            raise ValueError(f"MegaDepth pairs.json not found at {pairs_file}")

        with open(pairs_file, 'r') as f:
            pairs_data = json.load(f)

        scene_info_dir = self.root_path / "scene_info"

        for pair in pairs_data:
            scene_id = pair["scene_id"]
            img0_name = pair["image0"]
            img1_name = pair["image1"]

            # Load scene info
            scene_file = scene_info_dir / f"{scene_id}.npz"
            if not scene_file.exists():
                continue

            scene_data = np.load(scene_file, allow_pickle=True)

            # Get image paths
            images_dir = self.root_path / scene_id / "images"
            img0_path = images_dir / img0_name
            img1_path = images_dir / img1_name

            if not img0_path.exists() or not img1_path.exists():
                continue

            # Get calibration and poses
            K0 = scene_data[f"K_{img0_name}"]
            K1 = scene_data[f"K_{img1_name}"]
            T0 = scene_data[f"T_{img0_name}"]
            T1 = scene_data[f"T_{img1_name}"]

            # Compute relative pose
            T_0to1 = np.linalg.inv(T1) @ T0

            # Optional depth maps
            depth_dir = self.root_path / scene_id / "depth"
            depth0 = depth_dir / img0_name.replace(".jpg", ".png") if depth_dir.exists() else None
            depth1 = depth_dir / img1_name.replace(".jpg", ".png") if depth_dir.exists() else None

            samples.append(DatasetSample(
                image0=img0_path,
                image1=img1_path,
                K0=K0.astype(np.float32),
                K1=K1.astype(np.float32),
                T_0to1=T_0to1.astype(np.float32),
                depth0=depth0 if depth0 and depth0.exists() else None,
                depth1=depth1 if depth1 and depth1.exists() else None,
                metadata={"scene": scene_id, "overlap": pair.get("overlap", 0)}
            ))

        return samples
