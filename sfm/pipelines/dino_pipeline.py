"""DINO CLS-based retrieval and matching pipeline integrated with LoFTR."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sfm.matching import LoFTRMatcher, LoFTROptions, match_dino_patches
from sfm.retrieval import DINOCLSRetriever
from sfm.utils.feature_registry import FeatureRegistry
from sfm.core.geometric_verification import GeometricVerification

LOGGER = logging.getLogger(__name__)


def _sample_attention(attention_map: np.ndarray, image_shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Bilinear sample of attention map at pixel coordinates."""
    if points.size == 0:
        return np.zeros((0,), dtype=np.float32)

    img_h, img_w = image_shape
    grid_h, grid_w = attention_map.shape

    xs = np.clip((points[:, 0] / (img_w + 1e-8)) * grid_w - 0.5, 0.0, grid_w - 1.0)
    ys = np.clip((points[:, 1] / (img_h + 1e-8)) * grid_h - 0.5, 0.0, grid_h - 1.0)

    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, grid_w - 1)
    y0 = np.floor(ys).astype(int)
    y1 = np.clip(y0 + 1, 0, grid_h - 1)

    wx = xs - x0
    wy = ys - y0

    top_left = attention_map[y0, x0]
    top_right = attention_map[y0, x1]
    bottom_left = attention_map[y1, x0]
    bottom_right = attention_map[y1, x1]

    top = top_left * (1 - wx) + top_right * wx
    bottom = bottom_left * (1 - wx) + bottom_right * wx
    values = top * (1 - wy) + bottom * wy

    if values.size and values.max() > 0:
        values = values / (values.max() + 1e-8)

    return values.astype(np.float32)


def run_dino_pipeline(
    processed_images: Dict[str, np.ndarray],
    features: Dict[str, Dict[str, Any]],
    device: torch.device,
    output_path: Path,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Any], Dict[Tuple[str, str], Dict[str, torch.Tensor]], Dict[str, float]]:
    """Execute DINO retrieval and LoFTR matching."""
    stage_times: Dict[str, float] = {}

    # Retrieval -----------------------------------------------------------------
    start = time.time()
    retriever = DINOCLSRetriever()
    retrieval = retriever.retrieve(
        features,
        topk_primary=kwargs.get("dino_topk_primary", 60),
        topk_final=kwargs.get("dino_topk_final", 20),
    )
    stage_times["pair_selection"] = time.time() - start
    LOGGER.info(
        "DINO retrieval identified %d candidate pairs", len(retrieval.pairs)
    )

    # Matching ------------------------------------------------------------------
    start = time.time()
    loftr_options = LoFTROptions(
        pretrained=kwargs.get("loftr_pretrained", "outdoor"),
        max_long_edge=kwargs.get("loftr_max_long_edge", 832),
        lowres_long_edge=kwargs.get("loftr_lowres_long_edge", 448),
        use_attention_guidance=kwargs.get("dino_use_attention_guidance", True),
        attention_boost=kwargs.get("dino_attention_boost", 0.5),
    )
    loftr_matcher = LoFTRMatcher(device=device, options=loftr_options)

    magsac_config = {
        "geometric_method": "opencv_magsac",
        "confidence": kwargs.get("magsac_confidence", 0.999),
        "max_iterations": kwargs.get("magsac_max_iterations", 2000),
        "threshold": kwargs.get("magsac_threshold", 2.0),
        "min_matches": kwargs.get("min_inliers", 15),
    }
    geo_verifier = GeometricVerification(magsac_config)

    min_inliers = kwargs.get("min_inliers", 15)
    dino_cos_thresh = kwargs.get("dino_cosine_threshold", 0.8)
    dino_topk_match = kwargs.get("dino_topk_match", 800)

    registries = {path: FeatureRegistry(entry) for path, entry in features.items()}
    matches: Dict[Tuple[str, str], Any] = {}
    matches_tensors: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

    for img_a, img_b in retrieval.pairs:
        image_a = processed_images[img_a]
        image_b = processed_images[img_b]

        feat_a = features[img_a]
        feat_b = features[img_b]

        attn_a = feat_a.get("dino_attention_map")
        attn_b = feat_b.get("dino_attention_map")

        loftr_result = loftr_matcher.match(
            image_a, image_b, attention0=attn_a, attention1=attn_b, low_resolution=False
        )

        kpts0 = loftr_result["keypoints0"]
        kpts1 = loftr_result["keypoints1"]
        confidences = loftr_result["confidence"]

        inlier_mask = None
        if len(kpts0) >= min_inliers:
            _, inlier_mask = geo_verifier.find_fundamental_matrix(kpts0, kpts1)

        match_source = "loftr"
        if inlier_mask is None or inlier_mask.sum() < min_inliers:
            dino_match = match_dino_patches(
                feat_a,
                feat_b,
                cos_thresh=dino_cos_thresh,
                topk_i=dino_topk_match,
                topk_j=dino_topk_match,
            )
            if len(dino_match.matches0) < min_inliers:
                continue

            kpts0 = feat_a["dino_patch_coords"][dino_match.matches0]
            kpts1 = feat_b["dino_patch_coords"][dino_match.matches1]
            confidences = dino_match.cosine_similarity
            _, inlier_mask = geo_verifier.find_fundamental_matrix(kpts0, kpts1)
            if inlier_mask is None or inlier_mask.sum() < min_inliers:
                continue
            kpts0 = kpts0[inlier_mask]
            kpts1 = kpts1[inlier_mask]
            confidences = confidences[inlier_mask]
            match_source = "dino_fallback"
        else:
            kpts0 = kpts0[inlier_mask]
            kpts1 = kpts1[inlier_mask]
            confidences = confidences[inlier_mask]

        registry_a = registries[img_a]
        registry_b = registries[img_b]

        attention_a = _sample_attention(attn_a, feat_a["image_shape"], kpts0)
        attention_b = _sample_attention(attn_b, feat_b["image_shape"], kpts1)

        indices_a = []
        indices_b = []
        for pt_a, pt_b, att_a_val, att_b_val in zip(kpts0, kpts1, attention_a, attention_b):
            desc_a = None
            desc_b = None
            if match_source == "dino_fallback":
                idx_patch_a = np.argmin(np.linalg.norm(feat_a["dino_patch_coords"] - pt_a, axis=1))
                idx_patch_b = np.argmin(np.linalg.norm(feat_b["dino_patch_coords"] - pt_b, axis=1))
                desc_a = feat_a["dino_patch_tokens"][idx_patch_a]
                desc_b = feat_b["dino_patch_tokens"][idx_patch_b]

            idx_a = registry_a.register(tuple(pt_a), descriptor=desc_a, score=float(att_a_val))
            idx_b = registry_b.register(tuple(pt_b), descriptor=desc_b, score=float(att_b_val))
            indices_a.append(idx_a)
            indices_b.append(idx_b)

        if len(indices_a) < min_inliers:
            continue

        indices_a_np = np.asarray(indices_a, dtype=np.int32)
        indices_b_np = np.asarray(indices_b, dtype=np.int32)
        kpts0_np = kpts0.astype(np.float32)
        kpts1_np = kpts1.astype(np.float32)
        conf_norm = confidences.astype(np.float32)
        if conf_norm.size and conf_norm.max() > 0:
            conf_norm = conf_norm / conf_norm.max()

        pair_metadata = retrieval.metadata.get((img_a, img_b), {})
        pair_weight_terms = {
            "cls_similarity": float(pair_metadata.get("cls_similarity", 0.0)),
            "patch_similarity": float(pair_metadata.get("patch_similarity", 0.0)),
            "local_score": float(conf_norm.mean() if conf_norm.size else 0.0),
            "attention_score": float(((attention_a + attention_b) * 0.5).mean() if attention_a.size else 0.0),
            "match_source": match_source,
            "num_inliers": int(len(indices_a)),
        }

        matches[(img_a, img_b)] = {
            "matches0": indices_a_np,
            "matches1": indices_b_np,
            "mscores0": conf_norm,
            "mscores1": conf_norm,
            "points0": kpts0_np,
            "points1": kpts1_np,
            "attention0": attention_a,
            "attention1": attention_b,
            "pair_weight_terms": pair_weight_terms,
        }

        matches_tensors[(img_a, img_b)] = {
            "matches0": torch.from_numpy(indices_a_np).to(device),
            "matches1": torch.from_numpy(indices_b_np).to(device),
            "mscores0": torch.from_numpy(conf_norm).to(device),
            "mscores1": torch.from_numpy(conf_norm).to(device),
        }

    for registry in registries.values():
        registry.finalize()

    stage_times["feature_matching"] = time.time() - start
    LOGGER.info(
        "DINO pipeline produced %d verified pairs", len(matches)
    )

    updated_features = {path: registry.entry for path, registry in registries.items()}
    return updated_features, matches, matches_tensors, stage_times
