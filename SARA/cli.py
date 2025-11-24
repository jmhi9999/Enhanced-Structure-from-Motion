from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import SARAConfig
from .dino_embed import compute_or_load_embeddings
from .aliked_io import ensure_aliked_features
from .candidates import knn_candidates
from .mini_ransac import mutual_nn, estimate_F_mini_ransac
from .parallax import parallax_proxy
from .scoring import overlap_score, edge_score
from .gpu_batch import (
    batch_mutual_nn_gpu,
    batch_parallax_gpu,
    clear_gpu_cache,
)
from .mst import maximum_spanning_tree
from .augment import (
    leaf_augment,
    triangle_gain_augment,
    multi_scale_loop_augmentation,
    add_long_baseline_anchors,
    reinforce_weak_views,
)
from .io_utils import save_pairs_csv, save_pairs_for_matcher

logger = logging.getLogger(__name__)


def _check_gpu_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _intrinsics_from_config(
    cfg: SARAConfig,
    shape: Tuple[int, int],
) -> np.ndarray | None:
    if not cfg.use_intrinsics:
        return None
    if cfg.fx is None:
        return None
    fx = float(cfg.fx)
    fy = float(cfg.fy) if cfg.fy is not None else fx
    H, W = shape
    cx = W * 0.5
    cy = H * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _feature_shape(feat: Dict[str, np.ndarray]) -> Tuple[int, int]:
    shape = feat.get("shape") or feat.get("image_shape")
    if shape and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 1080, 1920


def _extract_allowed_stems(cfg: SARAConfig) -> Optional[Set[str]]:
    """Return the subset of image stems the caller wants to process, if any."""
    extras = getattr(cfg, "extra", None) or {}
    allowed = extras.get("allowed_image_stems")
    if not allowed:
        return None
    if isinstance(allowed, (str, Path)):
        return {_stem_from_any(allowed)}
    stems: Set[str] = set()
    for item in allowed:
        stems.add(_stem_from_any(item))
    return stems


def _stem_from_any(value) -> str:
    """Normalise any incoming identifier to an image stem."""
    if isinstance(value, Path):
        return value.stem
    return Path(str(value)).stem


def run_SARA(cfg: SARAConfig) -> Dict[str, List[Tuple[str, str, float, float, float]]]:
    """Execute the SARA pipeline and persist outputs."""
    allowed_stems = _extract_allowed_stems(cfg)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    embeddings = compute_or_load_embeddings(cfg.img_dir, cfg.out_dir, device=cfg.device)
    if allowed_stems is not None and embeddings:
        embeddings = {stem: emb for stem, emb in embeddings.items() if stem in allowed_stems}
    features = ensure_aliked_features(cfg.img_dir, cfg.out_dir, allowed_stems=allowed_stems)

    nodes = sorted(set(embeddings.keys()) & set(features.keys()))
    if len(nodes) < 2:
        raise RuntimeError("Need at least two images with embeddings and features.")

    logger.info("SARA: computing candidate graph")
    embeds_subset = {k: embeddings[k] for k in nodes}
    candidate_edges = knn_candidates(embeds_subset, cfg.knn_k)

    logger.info(
        "SARA: evaluating %d candidate edges (knn_k=%d) with GPU acceleration",
        len(candidate_edges),
        cfg.knn_k,
    )

    # Check if GPU is available for batch processing
    use_gpu = cfg.use_gpu_batch and cfg.device == "cuda" and _check_gpu_available()
    if use_gpu:
        logger.info("SARA: using GPU batch processing (memory-efficient mode)")
    else:
        if not cfg.use_gpu_batch:
            logger.info("SARA: GPU batch processing disabled by config")
        else:
            logger.info("SARA: using CPU processing (GPU not available)")

    edge_stats: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    all_weighted_edges: List[Tuple[str, str, float]] = []

    # Prepare batches for GPU processing
    desc_pairs = []
    pair_keys = []
    feature_data = []

    for si, sj in candidate_edges:
        fa = features[si]
        fb = features[sj]

        # Get descriptors (always numpy for memory efficiency)
        desc_a = fa.get("desc", fa.get("descriptors"))
        desc_b = fb.get("desc", fb.get("descriptors"))

        desc_pairs.append((desc_a, desc_b))
        pair_keys.append((si, sj))
        feature_data.append((fa, fb))

    # Batch GPU mutual NN
    logger.info(f"SARA: computing mutual nearest neighbors (GPU batch, metric={cfg.descriptor_metric})")
    if use_gpu:
        mutual_results = batch_mutual_nn_gpu(
            desc_pairs,
            cfg.top_t_mutual,
            device=cfg.device,
            batch_size=cfg.gpu_batch_size_mutual_nn,
            metric=cfg.descriptor_metric,
        )
    else:
        # Fallback to CPU
        mutual_results = [
            mutual_nn(desc_a, desc_b, cfg.top_t_mutual, metric=cfg.descriptor_metric)
            for desc_a, desc_b in desc_pairs
        ]

    # Filter and prepare for RANSAC
    ransac_data = []
    ransac_indices = []
    # Store matches for optional export
    sara_matches: Dict[Tuple[str, str], Dict] = {}

    for idx, ((si, sj), (idx_a, idx_b), (fa, fb)) in enumerate(
        zip(pair_keys, mutual_results, feature_data)
    ):
        if len(idx_a) < cfg.min_nn_for_ransac:
            continue

        # Get keypoints (always numpy for memory efficiency)
        kpts_a = fa.get("kpt", fa.get("keypoints"))
        kpts_b = fb.get("kpt", fb.get("keypoints"))

        ptsA = kpts_a[idx_a]
        ptsB = kpts_b[idx_b]

        ransac_data.append((ptsA, ptsB, fa, fb, kpts_a, kpts_b, idx_a, idx_b))
        ransac_indices.append((si, sj))

    # RANSAC (still CPU, but could be optimized with kornia)
    logger.info(f"SARA: running RANSAC on {len(ransac_data)} valid pairs")

    pts_pairs = []
    inliers_list = []
    image_shapes = []
    intrinsics_list = []
    valid_pair_indices = []  # Track which pairs passed RANSAC

    for (ptsA, ptsB, fa, fb, kpts_a, kpts_b, idx_a, idx_b), (si, sj) in zip(ransac_data, ransac_indices):
        F, inliers = estimate_F_mini_ransac(
            ptsA,
            ptsB,
            cfg.ransac_iters,
            cfg.ransac_conf,
        )

        if inliers.sum() < cfg.min_nn_for_ransac:
            continue

        H, W = _feature_shape(fa)
        intrinsics = _intrinsics_from_config(cfg, (H, W))

        pts_pairs.append((ptsA, ptsB))
        inliers_list.append(inliers)
        image_shapes.append((H, W))
        intrinsics_list.append(intrinsics)
        valid_pair_indices.append((si, sj))  # Track this pair

        # Compute overlap (cheap)
        overlap = overlap_score(int(inliers.sum()), cfg.top_t_mutual)

        # Store for later
        key = tuple(sorted((si, sj)))
        edge_stats[key] = (None, overlap, None)  # Placeholder for score and parallax

        # Store matches (RANSAC inliers only)
        inlier_idx_a = idx_a[inliers]
        inlier_idx_b = idx_b[inliers]

        # Create scores based on descriptor similarity (use uniform scores for now)
        scores = np.ones(len(inlier_idx_a), dtype=np.float32)

        sara_matches[(si, sj)] = {
            'keypoints0': kpts_a.astype(np.float32),
            'keypoints1': kpts_b.astype(np.float32),
            'matches0': inlier_idx_a.astype(np.int32),
            'matches1': inlier_idx_b.astype(np.int32),
            'mscores0': scores,
            'mscores1': scores,
            'image_shape0': tuple(_feature_shape(fa)),
            'image_shape1': tuple(_feature_shape(fb)),
        }

    # Batch GPU parallax computation
    logger.info(f"SARA: computing parallax (GPU batch) for {len(pts_pairs)} pairs")
    if use_gpu:
        parallax_results = batch_parallax_gpu(
            pts_pairs,
            inliers_list,
            image_shapes,
            intrinsics_list,
            device=cfg.device,
            batch_size=cfg.gpu_batch_size_parallax,
        )
    else:
        # Fallback to CPU
        parallax_results = [
            parallax_proxy(None, ptsA, ptsB, inliers, shape, K)
            for (ptsA, ptsB), inliers, shape, K in zip(
                pts_pairs, inliers_list, image_shapes, intrinsics_list
            )
        ]

    # Compute final scores - now indices match!
    logger.info(f"SARA: computing final scores for {len(valid_pair_indices)} pairs")

    filtered_by_score = 0
    for (si, sj), parallax in zip(valid_pair_indices, parallax_results):
        key = tuple(sorted((si, sj)))
        if key not in edge_stats:
            logger.warning(f"SARA: key {key} not in edge_stats (should not happen)")
            continue

        _, overlap, _ = edge_stats[key]

        score = edge_score(
            overlap,
            parallax,
            cfg.alpha,
            cfg.beta,
            cfg.tau_overlap,
            cfg.tau_parallax,
            cfg.scoring_mode,
        )

        if score <= 0.0 or not math.isfinite(score):
            filtered_by_score += 1
            del edge_stats[key]
            continue

        edge_stats[key] = (score, overlap, parallax)
        all_weighted_edges.append((si, sj, score))

    logger.info(f"SARA: {len(all_weighted_edges)} edges passed all filters")
    logger.info(f"SARA: {filtered_by_score} edges filtered by score thresholds")

    # Clear GPU cache
    if use_gpu:
        clear_gpu_cache(cfg.device)

    if len(all_weighted_edges) < len(nodes) - 1:
        logger.error(f"SARA: Not enough edges! Need {len(nodes) - 1}, got {len(all_weighted_edges)}")
        logger.error(f"SARA: Consider lowering thresholds: tau_overlap={cfg.tau_overlap}, tau_parallax={cfg.tau_parallax}")
        raise RuntimeError(
            f"Not enough valid edges to build MST. "
            f"Need {len(nodes) - 1} edges but only have {len(all_weighted_edges)}. "
            f"Try lowering --sara_tau_overlap (current: {cfg.tau_overlap}) or "
            f"--sara_tau_parallax (current: {cfg.tau_parallax})"
        )

    # Stage 1: MST (base connectivity)
    logger.info("SARA: building maximum spanning tree")
    mst_edges = maximum_spanning_tree(nodes, all_weighted_edges)
    logger.info(f"SARA: MST has {len(mst_edges)} edges")

    # Apply graph construction mode
    if cfg.graph_construction_mode == "mst_only":
        logger.info("SARA: using MST only (ablation mode)")
        current_edges = mst_edges
    elif cfg.graph_construction_mode == "mst_leaf":
        logger.info("SARA: using MST + Leaf augmentation (ablation mode)")
        current_edges = leaf_augment(nodes, all_weighted_edges, mst_edges, cfg.deg_cap)
        logger.info(f"SARA: after leaf augmentation: {len(current_edges)} edges")
    else:  # "full" mode
        # Stage 2: Leaf augmentation (standard)
        logger.info("SARA: augmenting leaf nodes")
        current_edges = leaf_augment(nodes, all_weighted_edges, mst_edges, cfg.deg_cap)
        logger.info(f"SARA: after leaf augmentation: {len(current_edges)} edges")

        # Stage 3: Weak-view reinforcement (if enabled)
        if cfg.enable_weak_view_reinforcement:
            logger.info("SARA: reinforcing weak views")
            current_edges = reinforce_weak_views(
                nodes,
                all_weighted_edges,
                current_edges,
                features,
                cfg.weak_view_percentile,
                cfg.weak_view_extra_edges,
            )
            logger.info(f"SARA: after weak-view reinforcement: {len(current_edges)} edges")

        # Stage 4: Multi-scale loop augmentation (if enabled)
        budget = int(math.ceil(cfg.loop_budget_per_node * len(nodes)))
        if cfg.enable_multi_scale_loops:
            logger.info(f"SARA: adding multi-scale loops (budget={budget})")
            current_edges = multi_scale_loop_augmentation(
                nodes,
                all_weighted_edges,
                current_edges,
                budget,
                cfg.small_loop_ratio,
                cfg.medium_loop_ratio,
                cfg.large_loop_ratio,
            )
            logger.info(f"SARA: after multi-scale loops: {len(current_edges)} edges")
        else:
            # Fallback to original triangle gain augmentation
            logger.info(f"SARA: adding loops via triangle gain (budget={budget})")
            current_edges = triangle_gain_augment(nodes, all_weighted_edges, current_edges, budget)
            logger.info(f"SARA: after loop augmentation: {len(current_edges)} edges")

        # Stage 5: Long-baseline anchors (if enabled)
        if cfg.enable_long_baseline_anchors:
            logger.info("SARA: adding long-baseline anchors")
            current_edges = add_long_baseline_anchors(
                nodes,
                all_weighted_edges,
                current_edges,
                embeddings,
                cfg.anchor_count,
                cfg.anchor_percentile,
            )
            logger.info(f"SARA: after anchors: {len(current_edges)} edges")

    # Build final pairs with metadata
    final_pairs = []
    for u, v, w in current_edges:
        key = tuple(sorted((u, v)))
        score, overlap, parallax = edge_stats[key]
        final_pairs.append((u, v, score, overlap, parallax))

    save_pairs_csv(final_pairs, Path(cfg.out_dir) / cfg.pairs_filename)
    save_pairs_for_matcher(
        [(u, v, s) for u, v, s, _, _ in final_pairs],
        Path(cfg.out_dir) / cfg.matcher_pairs_filename,
    )

    # Save SARA matches if any were computed
    if sara_matches:
        logger.info(f"SARA: saving {len(sara_matches)} matches to sara_matches.h5")
        try:
            # Import save function from sfm.utils.io_utils
            import sys
            from pathlib import Path as P
            # Add parent directory to path to import sfm module
            parent_dir = P(__file__).resolve().parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from sfm.utils.io_utils import save_matches
            save_matches(sara_matches, Path(cfg.out_dir) / "sara_matches.h5")
            logger.info("SARA: matches saved successfully")
        except ImportError as e:
            logger.warning(f"SARA: Could not save matches (import error): {e}")
        except Exception as e:
            logger.warning(f"SARA: Could not save matches: {e}")

    return {"pairs": final_pairs, "matches": sara_matches}
