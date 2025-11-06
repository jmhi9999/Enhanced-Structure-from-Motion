"""
SARA (Multi-scale Parallax-Aware) pair selection.
"""

from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pair_selectors.base import BasePairSelector, PairSelectorConfig


class SARAPairSelector(BasePairSelector):
    """
    SARA pair selection using DINO embeddings + geometric scoring.

    Excellent for large datasets with intelligent pair selection.
    O(n log n) complexity.
    """

    def _setup(self):
        """Setup SARA dependencies."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from SARA.config import SARAConfig
            from SARA.cli import run_SARA
            self.SARAConfig = SARAConfig
            self.run_SARA = run_SARA
        except ImportError as e:
            raise ImportError(f"SARA module not found: {e}")

    def _select_impl(
        self,
        image_list: List[Path],
        features: Dict[Path, any] = None
    ) -> List[Tuple[Path, Path]]:
        """Select pairs using SARA."""
        if features is None:
            raise ValueError("SARA requires features for pair selection")

        # Create temporary output directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get image directory (assume all images in same dir)
            img_dir = str(image_list[0].parent)
            out_dir = tmp_dir

            # Configure SARA
            cfg = self.SARAConfig(
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

            # Export features to SARA format
            self._export_features_for_SARA(features, Path(out_dir))

            # Run SARA
            result = self.run_SARA(cfg)

            # Parse pairs
            pairs = []
            stem_to_path = {p.stem: p for p in image_list}

            for stem_i, stem_j, *_ in result["pairs"]:
                if stem_i in stem_to_path and stem_j in stem_to_path:
                    pairs.append((stem_to_path[stem_i], stem_to_path[stem_j]))

            return pairs

    def _export_features_for_SARA(self, features: Dict, SARA_root: Path):
        """Export features to SARA-coSARAtible format."""
        import numpy as np

        features_dir = SARA_root / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        for path, feat_data in features.items():
            stem = path.stem

            # Convert to SARA format
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
