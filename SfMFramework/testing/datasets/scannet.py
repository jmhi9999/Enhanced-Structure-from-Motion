"""ScanNet dataset loader."""

from pathlib import Path
from typing import List
import numpy as np

from .base import BaseDataset, DatasetSample


class ScanNetDataset(BaseDataset):
    """ScanNet benchmark dataset.

    Expected structure:
        scannet/
        ├── scene0000_00/
        │   ├── color/
        │   │   ├── 0.jpg
        │   │   └── ...
        │   ├── depth/
        │   │   ├── 0.png
        │   │   └── ...
        │   ├── pose/
        │   │   ├── 0.txt
        │   │   └── ...
        │   └── intrinsic.txt
        └── scene0001_00/
            └── ...
    """

    def _load_samples(self) -> List[DatasetSample]:
        """Load ScanNet samples."""
        samples = []

        # Find all scenes
        scenes = [d for d in self.root_path.iterdir() if d.is_dir()]

        for scene_dir in scenes:
            color_dir = scene_dir / "color"
            depth_dir = scene_dir / "depth"
            pose_dir = scene_dir / "pose"
            intrinsic_file = scene_dir / "intrinsic.txt"

            if not color_dir.exists() or not pose_dir.exists() or not intrinsic_file.exists():
                continue

            # Load intrinsics
            K = self._load_intrinsics(intrinsic_file)

            # Get image list
            image_list = sorted(color_dir.glob("*.jpg"))

            # Create pairs (every 10 frames for efficiency)
            for i in range(0, len(image_list) - 1, 10):
                img0 = image_list[i]
                img1 = image_list[i + 1]

                # Get pose files
                pose0_file = pose_dir / f"{img0.stem}.txt"
                pose1_file = pose_dir / f"{img1.stem}.txt"

                if not pose0_file.exists() or not pose1_file.exists():
                    continue

                # Load poses
                T0 = self._load_pose(pose0_file)
                T1 = self._load_pose(pose1_file)

                # Skip invalid poses
                if not self._is_valid_pose(T0) or not self._is_valid_pose(T1):
                    continue

                # Compute relative pose
                T_0to1 = np.linalg.inv(T1) @ T0

                # Depth maps
                depth0 = depth_dir / f"{img0.stem}.png" if depth_dir.exists() else None
                depth1 = depth_dir / f"{img1.stem}.png" if depth_dir.exists() else None

                samples.append(DatasetSample(
                    image0=img0,
                    image1=img1,
                    K0=K.copy(),
                    K1=K.copy(),
                    T_0to1=T_0to1,
                    depth0=depth0 if depth0 and depth0.exists() else None,
                    depth1=depth1 if depth1 and depth1.exists() else None,
                    metadata={"scene": scene_dir.name}
                ))

        return samples

    def _load_intrinsics(self, intrinsic_file: Path) -> np.ndarray:
        """Load camera intrinsics."""
        K = np.loadtxt(intrinsic_file)[:3, :3].astype(np.float32)
        return K

    def _load_pose(self, pose_file: Path) -> np.ndarray:
        """Load camera pose."""
        T = np.loadtxt(pose_file).astype(np.float32)
        return T

    def _is_valid_pose(self, T: np.ndarray) -> bool:
        """Check if pose is valid (not NaN or Inf)."""
        return not (np.isnan(T).any() or np.isinf(T).any())
