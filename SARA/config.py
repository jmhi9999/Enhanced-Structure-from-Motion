from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class MPAConfig:
    """Configuration container for the MPA pair selection pipeline."""

    img_dir: str
    out_dir: str

    knn_k: int = 30

    top_t_mutual: int = 128
    min_nn_for_ransac: int = 32
    ransac_iters: int = 15
    ransac_conf: float = 0.999

    descriptor_metric: str = "cosine"  # "cosine" for learned descriptors, "l2" for SIFT/SURF

    tau_overlap: float = 0.10
    tau_parallax: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0

    loop_budget_per_node: float = 0.5
    deg_cap: Optional[int] = 6

    # Advanced augmentation strategies
    enable_multi_scale_loops: bool = True
    small_loop_ratio: float = 0.5  # 50% of loop budget
    medium_loop_ratio: float = 0.3  # 30% of loop budget
    large_loop_ratio: float = 0.2  # 20% of loop budget

    enable_long_baseline_anchors: bool = True
    anchor_count: int = 10
    anchor_percentile: float = 0.95  # Top 5%

    enable_weak_view_reinforcement: bool = True
    weak_view_percentile: float = 0.20  # Bottom 20%
    weak_view_extra_edges: int = 2

    use_intrinsics: bool = True
    fx: Optional[float] = None
    fy: Optional[float] = None

    num_workers: int = 8
    cache_dir: Optional[str] = None
    device: str = "cuda"

    # GPU optimization settings
    use_gpu_batch: bool = True  # Use GPU batch processing for mutual NN and parallax
    gpu_batch_size_mutual_nn: int = 128  # Batch size for mutual NN on GPU
    gpu_batch_size_parallax: int = 256  # Batch size for parallax on GPU

    embeddings_dir: Optional[str] = None
    features_dir: Optional[str] = None
    pairs_filename: str = "pairs.csv"
    matcher_pairs_filename: str = "pairs_for_matcher.jsonl"

    diagnostic_plots: bool = False

    extra: dict = field(default_factory=dict)

    def resolve_cache_dir(self) -> Path:
        """Resolve and create the cache directory used for intermediate products."""
        if self.cache_dir:
            cache_root = Path(self.cache_dir)
        else:
            cache_root = Path(self.out_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def resolve_embeddings_dir(self) -> Path:
        """Directory where DINO embeddings are cached."""
        if self.embeddings_dir:
            return Path(self.embeddings_dir)
        return self.resolve_cache_dir() / "embeddings"

    def resolve_features_dir(self) -> Path:
        """Directory where ALIKED features are expected to live."""
        if self.features_dir:
            return Path(self.features_dir)
        return self.resolve_cache_dir() / "features"

    def as_dict(self) -> dict:
        """Return a shallow dict representation that downstream utilities can mutate."""
        data = self.__dict__.copy()
        data["cache_dir"] = str(self.resolve_cache_dir())
        data["embeddings_dir"] = str(self.resolve_embeddings_dir())
        data["features_dir"] = str(self.resolve_features_dir())
        return data
