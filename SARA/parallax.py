from __future__ import annotations

import numpy as np


def _homogeneous(pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    return np.concatenate([pts, ones], axis=1)


def _normalise_with_intrinsics(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    K_inv = np.linalg.inv(K)
    rays = (K_inv @ _homogeneous(pts).T).T
    return rays


def _normalise_without_intrinsics(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    scale = float(max(H, W))
    cx = W * 0.5
    cy = H * 0.5
    x = (pts[:, 0] - cx) / scale
    y = (pts[:, 1] - cy) / scale
    z = np.ones_like(x)
    return np.stack([x, y, z], axis=1)


def parallax_proxy(
    F: np.ndarray,
    ptsA: np.ndarray,
    ptsB: np.ndarray,
    inliers: np.ndarray,
    image_shape: tuple[int, int],
    K: np.ndarray | None = None,
) -> float:
    """
    Return a scalar parallax proxy based on median sine of bearing angles.
    """
    if F is None or ptsA.shape[0] == 0 or not np.any(inliers):
        return 0.0

    ptsA = np.asarray(ptsA, dtype=np.float32)[inliers]
    ptsB = np.asarray(ptsB, dtype=np.float32)[inliers]

    if ptsA.shape[0] < 4:
        return 0.0

    if K is not None:
        rays_a = _normalise_with_intrinsics(ptsA, K)
        rays_b = _normalise_with_intrinsics(ptsB, K)
    else:
        rays_a = _normalise_without_intrinsics(ptsA, image_shape)
        rays_b = _normalise_without_intrinsics(ptsB, image_shape)

    # Normalise rays
    rays_a /= np.linalg.norm(rays_a, axis=1, keepdims=True)
    rays_b /= np.linalg.norm(rays_b, axis=1, keepdims=True)

    cos_theta = np.sum(rays_a * rays_b, axis=1).clip(-1.0, 1.0)
    theta = np.arccos(cos_theta)
    parallax = np.sin(theta)

    if parallax.size == 0 or not np.isfinite(parallax).any():
        return 0.0

    return float(np.median(parallax))
