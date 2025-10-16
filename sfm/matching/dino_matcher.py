"""Mutual nearest neighbour matching for DINO patch embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class DinoMatchResult:
    """Container for DINO patch matching results."""

    matches0: np.ndarray
    matches1: np.ndarray
    scores: np.ndarray
    attention0: np.ndarray
    attention1: np.ndarray
    cosine_similarity: np.ndarray


def _select_top_indices(attention: np.ndarray, topk: int) -> np.ndarray:
    """Return indices of top-k attention scores."""
    topk = min(topk, attention.shape[0])
    if topk <= 0:
        raise ValueError("topk must be positive.")
    return np.argsort(attention)[::-1][:topk]


def match_dino_patches(
    feat_i: Dict[str, np.ndarray],
    feat_j: Dict[str, np.ndarray],
    cos_thresh: float = 0.8,
    topk_i: int = 800,
    topk_j: int = 800,
) -> DinoMatchResult:
    """Match DINO patch embeddings with mutual NN filtering."""
    tokens_i = feat_i["dino_patch_tokens"]
    tokens_j = feat_j["dino_patch_tokens"]
    attn_i = feat_i["dino_patch_attention"]
    attn_j = feat_j["dino_patch_attention"]

    idx_i = _select_top_indices(attn_i, topk_i)
    idx_j = _select_top_indices(attn_j, topk_j)

    emb_i = torch.from_numpy(tokens_i[idx_i]).float()
    emb_j = torch.from_numpy(tokens_j[idx_j]).float()

    emb_i = torch.nn.functional.normalize(emb_i, dim=-1)
    emb_j = torch.nn.functional.normalize(emb_j, dim=-1)

    with torch.no_grad():
        sim = emb_i @ emb_j.t()

    sim_np = sim.cpu().numpy()

    nn_i = np.argmax(sim_np, axis=1)
    nn_j = np.argmax(sim_np, axis=0)

    matches0 = []
    matches1 = []
    scores = []

    for local_idx_i, local_idx_j in enumerate(nn_i):
        if nn_j[local_idx_j] != local_idx_i:
            continue
        score = sim_np[local_idx_i, local_idx_j]
        if score < cos_thresh:
            continue
        matches0.append(idx_i[local_idx_i])
        matches1.append(idx_j[local_idx_j])
        scores.append(score)

    if len(matches0) == 0:
        return DinoMatchResult(
            matches0=np.array([], dtype=np.int32),
            matches1=np.array([], dtype=np.int32),
            scores=np.array([], dtype=np.float32),
            attention0=np.array([], dtype=np.float32),
            attention1=np.array([], dtype=np.float32),
            cosine_similarity=np.array([], dtype=np.float32),
        )

    matches0_np = np.asarray(matches0, dtype=np.int32)
    matches1_np = np.asarray(matches1, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)

    attention0 = attn_i[matches0_np]
    attention1 = attn_j[matches1_np]

    return DinoMatchResult(
        matches0=matches0_np,
        matches1=matches1_np,
        scores=scores_np,
        attention0=attention0.astype(np.float32),
        attention1=attention1.astype(np.float32),
        cosine_similarity=scores_np.astype(np.float32),
    )
