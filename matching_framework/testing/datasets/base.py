"""Base dataset class for benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class DatasetSample:
    """Single image pair sample with ground truth."""

    image0: Path                        # Path to first image
    image1: Path                        # Path to second image
    K0: np.ndarray                      # Camera intrinsics [3, 3] for image0
    K1: np.ndarray                      # Camera intrinsics [3, 3] for image1
    T_0to1: np.ndarray                  # Relative pose [4, 4] from image0 to image1
    depth0: Optional[Path] = None       # Optional depth map for image0
    depth1: Optional[Path] = None       # Optional depth map for image1
    metadata: Optional[Dict] = None     # Additional metadata


class BaseDataset(ABC):
    """Base class for benchmark datasets."""

    def __init__(self, root_path: str, split: str = "test"):
        """
        Args:
            root_path: Root directory of the dataset
            split: Dataset split ("train", "val", "test")
        """
        self.root_path = Path(root_path)
        self.split = split

        if not self.root_path.exists():
            raise ValueError(f"Dataset path does not exist: {root_path}")

        self.samples = self._load_samples()

    @abstractmethod
    def _load_samples(self) -> List[DatasetSample]:
        """Load dataset samples with ground truth.

        Returns:
            List of DatasetSample objects
        """
        pass

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        """Get a sample by index."""
        return self.samples[idx]

    @staticmethod
    def relative_pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
        """Compute rotation and translation error.

        Args:
            T_est: Estimated pose [4, 4]
            T_gt: Ground truth pose [4, 4]

        Returns:
            (rotation_error_deg, translation_error_deg)
        """
        # Rotation error
        R_est = T_est[:3, :3]
        R_gt = T_gt[:3, :3]
        R_err = R_est.T @ R_gt
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_err = np.degrees(np.arccos(trace))

        # Translation error
        t_est = T_est[:3, 3]
        t_gt = T_gt[:3, 3]
        t_est = t_est / (np.linalg.norm(t_est) + 1e-8)
        t_gt = t_gt / (np.linalg.norm(t_gt) + 1e-8)
        trans_err = np.degrees(np.arccos(np.clip(np.dot(t_est, t_gt), -1, 1)))

        return rot_err, trans_err
