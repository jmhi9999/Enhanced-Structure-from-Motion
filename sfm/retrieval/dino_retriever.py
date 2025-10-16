"""DINO CLS-based image retrieval with FAISS support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional GPU acceleration
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None
    _FAISS_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Result of DINO retrieval."""

    pairs: List[Tuple[str, str]]
    metadata: Dict[Tuple[str, str], Dict[str, float]]


class DINOCLSRetriever:
    """Nearest neighbour retrieval on CLS embeddings with re-ranking."""

    def __init__(self, use_gpu: Optional[bool] = None):
        self.use_gpu = use_gpu
        if self.use_gpu is None:
            self.use_gpu = _FAISS_AVAILABLE and faiss.get_num_gpus() > 0  # type: ignore[attr-defined]

        self.index = None
        self.cls_matrix: Optional[np.ndarray] = None
        self.paths: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, features: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Build FAISS index from CLS embeddings."""
        self.paths = list(features.keys())
        cls_vectors = [features[path]["dino_cls"] for path in self.paths]
        self.cls_matrix = np.stack(cls_vectors).astype(np.float32)

        if _FAISS_AVAILABLE:
            dim = self.cls_matrix.shape[1]
            if self.use_gpu:
                res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                index_flat = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
                self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # type: ignore[attr-defined]
            else:
                self.index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
            self.index.add(self.cls_matrix)  # type: ignore[operator]
        else:
            self.index = None

    def retrieve(
        self,
        features: Dict[str, Dict[str, np.ndarray]],
        topk_primary: int = 60,
        topk_final: int = 20,
    ) -> RetrievalResult:
        """Retrieve candidate image pairs."""
        if self.cls_matrix is None or len(self.paths) == 0:
            self.build(features)

        reciprocal_map = self._search_neighbors(topk_primary)
        pairs, metadata = self._rerank_pairs(features, reciprocal_map, topk_final)

        return RetrievalResult(pairs=pairs, metadata=metadata)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _search_neighbors(self, topk: int) -> Dict[int, List[Tuple[int, float]]]:
        """Return reciprocal neighbours for each image index."""
        neighbour_map: Dict[int, List[Tuple[int, float]]] = {}

        for idx, vec in enumerate(self.cls_matrix):
            vec = vec.reshape(1, -1)
            if self.index is not None:
                sims, neigh = self.index.search(vec, topk + 1)  # type: ignore[operator]
                sims = sims[0]
                neigh = neigh[0]
            else:
                sims = self.cls_matrix @ vec.T
                sims = sims.flatten()
                neigh = np.argsort(sims)[::-1]

            candidates: List[Tuple[int, float]] = []
            for nbr, sim in zip(neigh, sims):
                if nbr == idx:
                    continue
                candidates.append((int(nbr), float(sim)))
                if len(candidates) >= topk:
                    break

            neighbour_map[idx] = candidates

        # Reciprocal filtering
        reciprocal_map: Dict[int, List[Tuple[int, float]]] = {
            idx: [] for idx in range(len(self.paths))
        }

        for idx, candidates in neighbour_map.items():
            for nbr, sim in candidates:
                nbr_candidates = neighbour_map.get(nbr, [])
                nbr_ids = [c for c, _ in nbr_candidates]
                if idx in nbr_ids:
                    reciprocal_map[idx].append((nbr, sim))

        return reciprocal_map

    def _rerank_pairs(
        self,
        features: Dict[str, Dict[str, np.ndarray]],
        reciprocal_map: Dict[int, List[Tuple[int, float]]],
        topk_final: int,
    ) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], Dict[str, float]]]:
        """Re-rank candidate pairs using patch similarity."""
        pair_scores: Dict[Tuple[int, int], Dict[str, float]] = {}

        for idx, candidates in reciprocal_map.items():
            anchor_path = self.paths[idx]
            anchor_feat = features[anchor_path]

            scored_candidates: List[Tuple[int, float, float]] = []

            for nbr, cls_sim in candidates:
                nbr_path = self.paths[nbr]
                nbr_feat = features[nbr_path]
                patch_sim = self._compute_patch_similarity(anchor_feat, nbr_feat)
                score = 0.7 * cls_sim + 0.3 * patch_sim
                scored_candidates.append((nbr, score, patch_sim))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            for nbr, score, patch_sim in scored_candidates[:topk_final]:
                pair = tuple(sorted((idx, nbr)))
                cls_sim = self._cls_similarity(idx, nbr)

                if pair not in pair_scores or score > pair_scores[pair]["combined_score"]:
                    pair_scores[pair] = {
                        "combined_score": float(score),
                        "cls_similarity": float(cls_sim),
                        "patch_similarity": float(patch_sim),
                    }

        # Convert to path pairs
        pairs: List[Tuple[str, str]] = []
        metadata: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (idx_a, idx_b), stats in pair_scores.items():
            path_a = self.paths[idx_a]
            path_b = self.paths[idx_b]
            pair = (path_a, path_b)
            pairs.append(pair)
            metadata[pair] = stats

        return pairs, metadata

    def _cls_similarity(self, idx_a: int, idx_b: int) -> float:
        """Compute cosine similarity between two CLS vectors."""
        vec_a = self.cls_matrix[idx_a]
        vec_b = self.cls_matrix[idx_b]
        sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))
        return sim

    @staticmethod
    def _compute_patch_similarity(
        feat_a: Dict[str, np.ndarray],
        feat_b: Dict[str, np.ndarray],
        top_patches: int = 400,
        top_values: int = 50,
    ) -> float:
        """Compute patch-level similarity using top attention patches."""
        tokens_a = feat_a["dino_patch_tokens"]
        tokens_b = feat_b["dino_patch_tokens"]
        attn_a = feat_a["dino_patch_attention"]
        attn_b = feat_b["dino_patch_attention"]

        idx_a = np.argsort(attn_a)[::-1][:top_patches]
        idx_b = np.argsort(attn_b)[::-1][:top_patches]

        emb_a = tokens_a[idx_a]
        emb_b = tokens_b[idx_b]

        emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
        emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

        sim_matrix = emb_a @ emb_b.T
        flat = sim_matrix.flatten()
        if flat.size == 0:
            return 0.0
        if top_values >= flat.size:
            top_vals = flat
        else:
            split_idx = flat.size - top_values
            top_vals = np.partition(flat, split_idx)[split_idx:]
        return float(np.mean(top_vals))
