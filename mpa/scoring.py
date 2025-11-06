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
) -> float:
    """Combine overlap and parallax with thresholds and exponents."""
    if overlap < tau_overlap or parallax < tau_parallax:
        return 0.0
    score = (overlap ** alpha) * (parallax ** beta)
    return float(score)
