"""Dataset loaders for benchmarking."""

from .base import BaseDataset, DatasetSample
from .eth3d import ETH3DDataset
from .scannet import ScanNetDataset
from .megadepth import MegaDepthDataset
from .colmap_scene import ColmapSceneDataset

__all__ = [
    "BaseDataset",
    "DatasetSample",
    "ETH3DDataset",
    "ScanNetDataset",
    "MegaDepthDataset",
    "ColmapSceneDataset",
]


class DatasetFactory:
    """Factory for creating datasets."""

    _datasets = {
        "eth3d": ETH3DDataset,
        "scannet": ScanNetDataset,
        "megadepth": MegaDepthDataset,
        "colmap_scene": ColmapSceneDataset,
    }

    @classmethod
    def create(cls, name: str, root_path: str, **kwargs):
        """Create dataset by name."""
        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
        return cls._datasets[name](root_path, **kwargs)

    @classmethod
    def list_datasets(cls):
        """List available datasets."""
        return list(cls._datasets.keys())
