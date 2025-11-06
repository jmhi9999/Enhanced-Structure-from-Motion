"""ETH3D dataset loader."""

import json
from pathlib import Path
from typing import List
import numpy as np

from .base import BaseDataset, DatasetSample


class ETH3DDataset(BaseDataset):
    """ETH3D benchmark dataset.

    Expected structure:
        eth3d/
        ├── scene1/
        │   ├── images/
        │   │   ├── image0000.jpg
        │   │   └── ...
        │   ├── calibration.txt
        │   └── poses.txt
        └── scene2/
            └── ...
    """

    def _load_samples(self) -> List[DatasetSample]:
        """Load ETH3D samples."""
        samples = []

        # Find all scenes
        scenes = [d for d in self.root_path.iterdir() if d.is_dir()]

        for scene_dir in scenes:
            images_dir = scene_dir / "images"
            calib_file = scene_dir / "calibration.txt"
            poses_file = scene_dir / "poses.txt"

            if not images_dir.exists() or not calib_file.exists() or not poses_file.exists():
                continue

            # Load calibration
            K = self._load_calibration(calib_file)

            # Load poses
            poses = self._load_poses(poses_file)

            # Get image list
            image_list = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

            # Create pairs (consecutive frames)
            for i in range(len(image_list) - 1):
                img0 = image_list[i]
                img1 = image_list[i + 1]

                # Get poses
                T0 = poses.get(img0.stem)
                T1 = poses.get(img1.stem)

                if T0 is None or T1 is None:
                    continue

                # Compute relative pose
                T_0to1 = np.linalg.inv(T1) @ T0

                samples.append(DatasetSample(
                    image0=img0,
                    image1=img1,
                    K0=K.copy(),
                    K1=K.copy(),
                    T_0to1=T_0to1,
                    metadata={"scene": scene_dir.name}
                ))

        return samples

    def _load_calibration(self, calib_file: Path) -> np.ndarray:
        """Load camera intrinsics from calibration file."""
        with open(calib_file, 'r') as f:
            lines = f.readlines()

        # Parse intrinsics (fx, fy, cx, cy)
        params = [float(x) for x in lines[0].strip().split()]
        fx, fy, cx, cy = params[:4]

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def _load_poses(self, poses_file: Path) -> dict:
        """Load camera poses from poses file."""
        poses = {}

        with open(poses_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # Image name
            img_name = lines[i].strip()
            i += 1

            # 4x4 transformation matrix
            T = []
            for _ in range(4):
                row = [float(x) for x in lines[i].strip().split()]
                T.append(row)
                i += 1

            poses[img_name] = np.array(T, dtype=np.float32)

        return poses
