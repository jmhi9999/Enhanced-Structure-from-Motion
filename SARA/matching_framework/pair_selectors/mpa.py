"""
MPA (Multi-scale Parallax-Aware) pair selection.
"""

from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pair_selectors.base import BasePairSelector, PairSelectorConfig


class MPAPairSelector(BasePairSelector):
    """
    MPA pair selection using DINO embeddings + geometric scoring.

    Excellent for large datasets with intelligent pair selection.
    O(n log n) complexity.
    """

    def _setup(self):
        """Setup MPA dependencies."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from mpa.config import MPAConfig
            from mpa.cli import run_mpa
            self.MPAConfig = MPAConfig
            self.run_mpa = run_mpa
        except ImportError as e:
            raise ImportError(f"MPA module not found: {e}")

    def _select_impl(
        self,
        image_list: List[Path],
        features: Dict[Path, any] = None
    ) -> List[Tuple[Path, Path]]:
        """Select pairs using MPA."""
        if features is None:
            raise ValueError("MPA requires features for pair selection")

        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get image directory (assume all images in same dir)
            img_dir = str(image_list[0].parent)
            out_dir = tmp_dir

            # Configure MPA
            cfg = self.MPAConfig(
                img_dir=img_dir,
                out_dir=out_dir,
                knn_k=self.config.extra.get("knn_k", 30),
                top_t_mutual=self.config.extra.get("top_t_mutual", 128),
                min_nn_for_ransac=self.config.extra.get("min_nn_for_ransac", 32),
                tau_overlap=self.config.extra.get("tau_overlap", 0.10),
                tau_parallax=self.config.extra.get("tau_parallax", 0.05),
                loop_budget_per_node=self.config.extra.get("loop_budget_per_node", 0.5),
                descriptor_metric=self.config.extra.get("descriptor_metric", "cosine"),
                device=self.config.device,
            )

            # Export features to MPA format
            self._export_features_for_mpa(features, Path(out_dir))

            # Run MPA
            result = self.run_mpa(cfg)

            # Parse pairs
            pairs = []
            stem_to_path = {p.stem: p for p in image_list}

            for stem_i, stem_j, *_ in result["pairs"]:
                if stem_i in stem_to_path and stem_j in stem_to_path:
                    pairs.append((stem_to_path[stem_i], stem_to_path[stem_j]))

            return pairs

    def _export_features_for_mpa(self, features: Dict, mpa_root: Path):
        """Export features to MPA-compatible format."""
        import numpy as np

        features_dir = mpa_root / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        for path, feat_data in features.items():
            stem = path.stem

            # Convert to MPA format
            kpts = feat_data.keypoints
            desc = feat_data.descriptors
            scores = feat_data.scores
            shape = feat_data.image_shape

            # Save as npz
            np.savez(
                features_dir / f"{stem}.npz",
                keypoints=kpts.astype(np.float32),
                descriptors=desc.astype(np.float32),
                scores=scores.astype(np.float32),
                image_shape=np.array(shape, dtype=np.int32),
            )
