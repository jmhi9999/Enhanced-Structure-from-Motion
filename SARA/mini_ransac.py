from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "OpenCV is required for mini RANSAC. Install with `pip install opencv-python`."
    ) from exc


def _normalise_descriptors(desc: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return desc / norms


def mutual_nn(
    descA: np.ndarray,
    descB: np.ndarray,
    t: int,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mutual nearest neighbours with configurable distance metric.

    Args:
        descA: Descriptors from image A (N x D)
        descB: Descriptors from image B (M x D)
        t: Number of top mutual matches to return
        metric: Distance metric - "cosine" (for learned descriptors) or "l2" (for SIFT)

    Returns:
        Tuple of (indices_A, indices_B) for mutual nearest neighbors
    """
    if descA.size == 0 or descB.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    descA = descA.astype(np.float32)
    descB = descB.astype(np.float32)

    if metric == "cosine":
        # Cosine similarity (for SuperPoint, ALIKED, DISK)
        descA = _normalise_descriptors(descA)
        descB = _normalise_descriptors(descB)
        sim = descA @ descB.T  # Higher is better
        best_j = np.argmax(sim, axis=1)
        best_i = np.argmax(sim, axis=0)

        mutual_pairs = []
        for i, j in enumerate(best_j):
            if best_i[j] == i:
                mutual_pairs.append((i, j, sim[i, j]))

        if not mutual_pairs:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Sort by similarity (higher is better)
        mutual_pairs.sort(key=lambda x: x[2], reverse=True)

    elif metric == "l2":
        # L2 distance (for SIFT, SURF)
        # Compute pairwise L2 distances
        # dist[i,j] = ||descA[i] - descB[j]||^2
        normA = np.sum(descA**2, axis=1, keepdims=True)  # N x 1
        normB = np.sum(descB**2, axis=1, keepdims=True)  # M x 1
        dist = normA + normB.T - 2 * (descA @ descB.T)  # N x M
        dist = np.maximum(dist, 0)  # Numerical stability

        best_j = np.argmin(dist, axis=1)  # Lower is better
        best_i = np.argmin(dist, axis=0)

        mutual_pairs = []
        for i, j in enumerate(best_j):
            if best_i[j] == i:
                mutual_pairs.append((i, j, -dist[i, j]))  # Negate for sorting

        if not mutual_pairs:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        # Sort by distance (lower is better, so negate)
        mutual_pairs.sort(key=lambda x: x[2], reverse=True)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'l2'")

    mutual_pairs = mutual_pairs[: min(t, len(mutual_pairs))]

    A_idx = np.array([i for i, _, _ in mutual_pairs], dtype=np.int32)
    B_idx = np.array([j for _, j, _ in mutual_pairs], dtype=np.int32)
    return A_idx, B_idx


def estimate_F_mini_ransac(
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    max_iters: int,
    confidence: float,
    reproj_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a lightweight RANSAC to estimate the fundamental matrix."""
    if len(ptsA) < 8 or len(ptsB) < 8:
        return np.eye(3, dtype=np.float32), np.zeros(len(ptsA), dtype=bool)

    ptsA = np.asarray(ptsA, dtype=np.float32)
    ptsB = np.asarray(ptsB, dtype=np.float32)

    F, mask = cv2.findFundamentalMat(
        ptsA,
        ptsB,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=reproj_threshold,
        confidence=float(confidence),
        maxIters=int(max(1, max_iters)),
    )

    if F is None or mask is None:
        return np.eye(3, dtype=np.float32), np.zeros(len(ptsA), dtype=bool)

    mask = mask.astype(bool).reshape(-1)
    return F.astype(np.float32), mask
