from __future__ import annotations

import math


def overlap_score(inliers: int, total: int) -> float:
    """Return overlap proxy based on inlier ratio."""
    total = max(1, total)
    frac = max(0.0, min(1.0, inliers / total))
    return float(frac)


def edge_score(
    overlap: float,
    parallax: float,
    alpha: float,
    beta: float,
    tau_overlap: float,
    tau_parallax: float,
    scoring_mode: str = "combined",
) -> float:
    """Combine overlap and parallax with thresholds and exponents.

    Args:
        overlap: Overlap score (0-1)
        parallax: Parallax score (0-1)
        alpha: Exponent for overlap
        beta: Exponent for parallax
        tau_overlap: Minimum overlap threshold
        tau_parallax: Minimum parallax threshold
        scoring_mode: "overlap_only", "parallax_only", or "combined"
    """
    if overlap < tau_overlap or parallax < tau_parallax:
        return 0.0

    if scoring_mode == "overlap_only":
        score = overlap ** alpha
    elif scoring_mode == "parallax_only":
        score = parallax ** beta
    else:  # "combined"
        score = (overlap ** alpha) * (parallax ** beta)

    return float(score)
