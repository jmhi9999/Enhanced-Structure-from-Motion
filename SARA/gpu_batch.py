"""GPU-accelerated batch processing for MPA pipeline."""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def batch_mutual_nn_gpu(
    desc_pairs: List[Tuple[np.ndarray, np.ndarray]],
    t: int,
    device: str = "cuda",
    batch_size: int = 128,
    metric: str = "cosine",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    GPU-accelerated batch mutual nearest neighbor matching.

    Args:
        desc_pairs: List of (descA, descB) descriptor pairs
        t: Top-t mutual matches to keep
        device: Device for computation
        batch_size: Number of pairs to process at once
        metric: Distance metric - "cosine" for learned descriptors, "l2" for SIFT

    Returns:
        List of (idxA, idxB) index pairs
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU for mutual NN")

    device_obj = torch.device(device)
    results = []

    for i in range(0, len(desc_pairs), batch_size):
        batch = desc_pairs[i : i + batch_size]
        batch_results = []

        for descA, descB in batch:
            if descA.size == 0 or descB.size == 0:
                batch_results.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)))
                continue

            # Convert to torch tensors
            descA_t = torch.from_numpy(descA.astype(np.float32)).to(device_obj)
            descB_t = torch.from_numpy(descB.astype(np.float32)).to(device_obj)

            if metric == "cosine":
                # L2 normalize for cosine similarity
                descA_t = torch.nn.functional.normalize(descA_t, p=2, dim=1)
                descB_t = torch.nn.functional.normalize(descB_t, p=2, dim=1)

                # Cosine similarity (higher is better)
                sim = descA_t @ descB_t.T  # [M, N]

                # Mutual nearest neighbors
                best_j = torch.argmax(sim, dim=1)  # A→B
                best_i = torch.argmax(sim, dim=0)  # B→A

                # Find mutual matches
                mutual_mask = best_i[best_j] == torch.arange(len(descA_t), device=device_obj)
                mutual_idx_a = torch.where(mutual_mask)[0]
                mutual_idx_b = best_j[mutual_idx_a]

                if len(mutual_idx_a) == 0:
                    batch_results.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)))
                    continue

                # Get similarity scores for mutual matches (higher is better)
                mutual_scores = sim[mutual_idx_a, mutual_idx_b]

                # Top-t by score
                if len(mutual_scores) > t:
                    top_t_idx = torch.argsort(mutual_scores, descending=True)[:t]
                    mutual_idx_a = mutual_idx_a[top_t_idx]
                    mutual_idx_b = mutual_idx_b[top_t_idx]

            elif metric == "l2":
                # L2 distance (lower is better)
                # dist[i,j] = ||descA[i] - descB[j]||^2
                normA = torch.sum(descA_t**2, dim=1, keepdim=True)  # [M, 1]
                normB = torch.sum(descB_t**2, dim=1, keepdim=True)  # [N, 1]
                dist = normA + normB.T - 2 * (descA_t @ descB_t.T)  # [M, N]
                dist = torch.clamp(dist, min=0)  # Numerical stability

                # Mutual nearest neighbors (argmin for distance)
                best_j = torch.argmin(dist, dim=1)  # A→B
                best_i = torch.argmin(dist, dim=0)  # B→A

                # Find mutual matches
                mutual_mask = best_i[best_j] == torch.arange(len(descA_t), device=device_obj)
                mutual_idx_a = torch.where(mutual_mask)[0]
                mutual_idx_b = best_j[mutual_idx_a]

                if len(mutual_idx_a) == 0:
                    batch_results.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)))
                    continue

                # Get distances for mutual matches (lower is better, so negate for sorting)
                mutual_distances = dist[mutual_idx_a, mutual_idx_b]

                # Top-t by distance (negate for descending sort)
                if len(mutual_distances) > t:
                    top_t_idx = torch.argsort(mutual_distances, descending=False)[:t]
                    mutual_idx_a = mutual_idx_a[top_t_idx]
                    mutual_idx_b = mutual_idx_b[top_t_idx]

            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'l2'")

            # Convert back to numpy
            idx_a = mutual_idx_a.cpu().numpy().astype(np.int32)
            idx_b = mutual_idx_b.cpu().numpy().astype(np.int32)

            batch_results.append((idx_a, idx_b))

        results.extend(batch_results)

    return results


def batch_parallax_gpu(
    pts_pairs: List[Tuple[np.ndarray, np.ndarray]],
    inliers_list: List[np.ndarray],
    image_shapes: List[Tuple[int, int]],
    intrinsics_list: List[np.ndarray | None],
    device: str = "cuda",
    batch_size: int = 256,
) -> List[float]:
    """
    GPU-accelerated batch parallax computation.

    Args:
        pts_pairs: List of (ptsA, ptsB) point pairs
        inliers_list: List of inlier masks
        image_shapes: List of (H, W) tuples
        intrinsics_list: List of K matrices (or None)
        device: Device for computation
        batch_size: Number of pairs to process at once

    Returns:
        List of parallax scores
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU for parallax")

    device_obj = torch.device(device)
    results = []

    for i in range(0, len(pts_pairs), batch_size):
        batch_pts = pts_pairs[i : i + batch_size]
        batch_inliers = inliers_list[i : i + batch_size]
        batch_shapes = image_shapes[i : i + batch_size]
        batch_K = intrinsics_list[i : i + batch_size]

        for (ptsA, ptsB), inliers, shape, K in zip(
            batch_pts, batch_inliers, batch_shapes, batch_K
        ):
            if ptsA.shape[0] == 0 or not np.any(inliers):
                results.append(0.0)
                continue

            # Filter by inliers
            ptsA = ptsA[inliers].astype(np.float32)
            ptsB = ptsB[inliers].astype(np.float32)

            if ptsA.shape[0] < 4:
                results.append(0.0)
                continue

            # Convert to torch
            ptsA_t = torch.from_numpy(ptsA).to(device_obj)
            ptsB_t = torch.from_numpy(ptsB).to(device_obj)

            # Normalize to ray directions
            if K is not None:
                K_t = torch.from_numpy(K.astype(np.float32)).to(device_obj)
                K_inv = torch.inverse(K_t)

                # Homogeneous coordinates
                ones = torch.ones((ptsA_t.shape[0], 1), device=device_obj)
                ptsA_h = torch.cat([ptsA_t, ones], dim=1)
                ptsB_h = torch.cat([ptsB_t, ones], dim=1)

                # Normalized rays
                rays_a = (K_inv @ ptsA_h.T).T
                rays_b = (K_inv @ ptsB_h.T).T
            else:
                # Normalized image coordinates
                H, W = shape
                scale = float(max(H, W))
                cx = W * 0.5
                cy = H * 0.5

                x_a = (ptsA_t[:, 0] - cx) / scale
                y_a = (ptsA_t[:, 1] - cy) / scale
                z_a = torch.ones_like(x_a)
                rays_a = torch.stack([x_a, y_a, z_a], dim=1)

                x_b = (ptsB_t[:, 0] - cx) / scale
                y_b = (ptsB_t[:, 1] - cy) / scale
                z_b = torch.ones_like(x_b)
                rays_b = torch.stack([x_b, y_b, z_b], dim=1)

            # Normalize rays
            rays_a = torch.nn.functional.normalize(rays_a, p=2, dim=1)
            rays_b = torch.nn.functional.normalize(rays_b, p=2, dim=1)

            # Parallax angle
            cos_theta = torch.sum(rays_a * rays_b, dim=1).clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)
            parallax = torch.sin(theta)

            # Median
            if parallax.numel() == 0 or not torch.isfinite(parallax).any():
                results.append(0.0)
            else:
                valid_parallax = parallax[torch.isfinite(parallax)]
                if valid_parallax.numel() > 0:
                    results.append(float(torch.median(valid_parallax).cpu()))
                else:
                    results.append(0.0)

    return results


def preload_features_to_gpu(
    features: Dict[str, Dict[str, np.ndarray]],
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Preload all features to GPU for faster access.

    Args:
        features: Dictionary of features (numpy arrays)
        device: Target device

    Returns:
        Dictionary of features (torch tensors on GPU)
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, keeping features on CPU")

    device_obj = torch.device(device)
    gpu_features = {}

    logger.info(f"Preloading {len(features)} feature sets to {device}")

    for key, feat in features.items():
        gpu_feat = {}
        for field in ["kpt", "desc", "score", "keypoints", "descriptors", "scores"]:
            if field in feat:
                value = feat[field]
                if isinstance(value, np.ndarray):
                    gpu_feat[field] = torch.from_numpy(value.astype(np.float32)).to(device_obj)
                else:
                    gpu_feat[field] = value

        # Preserve metadata
        if "shape" in feat:
            gpu_feat["shape"] = feat["shape"]
        if "image_shape" in feat:
            gpu_feat["image_shape"] = feat["image_shape"]

        gpu_features[key] = gpu_feat

    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        logger.info(f"GPU memory after feature preload: {mem_allocated:.1f} MB")

    return gpu_features


def clear_gpu_cache(device: str = "cuda"):
    """Clear GPU cache and run garbage collection."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_cached = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU cache cleared: {mem_allocated:.1f} MB allocated, {mem_cached:.1f} MB cached")
