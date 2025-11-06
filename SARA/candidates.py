from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def knn_candidates(embeds: Dict[str, np.ndarray], k: int) -> List[Tuple[str, str]]:
    """Return symmetric cosine kNN candidate edge list."""
    if not embeds:
        return []

    stems = sorted(embeds.keys())
    matrix = np.stack([embeds[s] for s in stems], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    k = max(1, min(k, len(stems) - 1))

    candidate_pairs = set()
    for idx, stem in enumerate(stems):
        top_idx = np.argpartition(-scores[idx], k)[:k]
        for j in top_idx:
            if j == idx:
                continue
            a, b = stems[idx], stems[j]
            edge = tuple(sorted((a, b)))
            candidate_pairs.add(edge)

    return sorted(candidate_pairs)
