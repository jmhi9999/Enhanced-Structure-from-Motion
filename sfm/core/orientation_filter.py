"""
Orientation-Based Pre-filtering for Geometric Verification

This module implements Theorem 3: Orientation constraints define a convex
polytope in transformation space. It provides both stochastic (legacy) and
deterministic utilities so that the algebraic consensus pipeline can remain
mathematically transparent.

Mathematical Framework:
    - Orientation consistency: |θ_dst - θ_src - θ(A)| ≤ τ
    - Interval representation in rotation space
    - Deterministic candidate enumeration via interval intersections

Author: Claude Code
Date: 2025-10-30
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import numpy as np
from scipy.stats import circmean, circstd

from .polynomial_system import Correspondence


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class OrientationStatistics:
    """
    Circular statistics for orientation analysis.

    Attributes:
        mean_angle: Circular mean of orientation differences (radians)
        concentration: Concentration parameter (higher = more concentrated)
        std: Circular standard deviation (radians)
        n_samples: Number of samples
    """
    mean_angle: float
    concentration: float
    std: float
    n_samples: int

    @property
    def is_concentrated(self) -> bool:
        """Check if orientations are highly concentrated."""
        return self.std < np.radians(30)


@dataclass(frozen=True)
class OrientationConstraint:
    """
    Individual orientation constraint represented as angular interval.

    Attributes:
        index: Index of correspondence providing the constraint
        delta: Observed orientation difference θ_dst - θ_src
        lower: Lower bound (wrapped to [-π, π])
        upper: Upper bound (wrapped to [-π, π])
    """
    index: int
    delta: float
    lower: float
    upper: float

    def contains(self, angle: float) -> bool:
        """Check if given angle lies inside the interval."""
        value = wrap_angle(angle)
        if self.lower <= self.upper:
            return self.lower <= value <= self.upper
        # Interval wraps around π
        return value >= self.lower or value <= self.upper


class OrientationFilter:
    """
    Orientation-based pre-filtering using circular statistics.

    Supports both stochastic (RANSAC-like) and deterministic estimation
    of the dominant rotation. Deterministic utilities are used by the
    algebraic "smart mode".
    """

    def __init__(
        self,
        tau: float = 0.3,  # ~17 degrees
        min_consensus: int = 3,
        use_robust_estimation: bool = True
    ):
        """
        Initialize orientation filter.

        Args:
            tau: Orientation threshold (radians)
            min_consensus: Minimum number of orientation-consistent matches
            use_robust_estimation: Use RANSAC-like robust estimation for orientation
        """
        self.tau = float(tau)
        self.min_consensus = int(min_consensus)
        self.use_robust_estimation = bool(use_robust_estimation)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_correspondences(
        self,
        correspondences: List[Correspondence]
    ) -> Tuple[List[Correspondence], OrientationStatistics]:
        """
        Filter correspondences by orientation consistency.

        Args:
            correspondences: List of matches with orientations

        Returns:
            Tuple of (filtered_correspondences, statistics)
        """
        constraints, valid_indices = self.compute_constraints_with_indices(correspondences)

        if len(constraints) < self.min_consensus:
            stats = OrientationStatistics(
                mean_angle=0.0,
                concentration=0.0,
                std=np.pi,
                n_samples=0
            )
            return correspondences, stats

        theta_diffs = np.array([c.delta for c in constraints], dtype=float)

        if self.use_robust_estimation:
            dominant_angle, inlier_mask = self._estimate_dominant_orientation_robust(theta_diffs)
        else:
            dominant_angle, inlier_mask = self._estimate_dominant_orientation_deterministic(constraints)

        consistent_mask = np.abs(
            self._angular_difference(theta_diffs, dominant_angle)
        ) < self.tau

        final_mask = consistent_mask & inlier_mask

        keep_map = {
            idx: bool(final_mask[pos]) for pos, idx in enumerate(valid_indices)
        }

        filtered: List[Correspondence] = []
        for idx, corr in enumerate(correspondences):
            if corr.orientation_src is None or corr.orientation_dst is None:
                filtered.append(corr)
                continue
            if keep_map.get(idx, False):
                filtered.append(corr)

        consistent_diffs = theta_diffs[final_mask]
        stats = self._compute_statistics(consistent_diffs)

        return filtered, stats

    def compute_constraints(
        self,
        correspondences: Iterable[Correspondence]
    ) -> List[OrientationConstraint]:
        """Convenience wrapper returning only constraints."""
        constraints, _ = self.compute_constraints_with_indices(correspondences)
        return constraints

    def compute_constraints_with_indices(
        self,
        correspondences: Iterable[Correspondence]
    ) -> Tuple[List[OrientationConstraint], List[int]]:
        """
        Convert correspondences to angular constraints.

        Returns:
            constraints: List of OrientationConstraint
            indices: Positions of correspondences contributing constraints
        """
        constraints: List[OrientationConstraint] = []
        indices: List[int] = []

        for idx, corr in enumerate(correspondences):
            if corr.orientation_src is None or corr.orientation_dst is None:
                continue

            delta = self._angular_difference(corr.orientation_dst, corr.orientation_src)
            lower = wrap_angle(delta - self.tau)
            upper = wrap_angle(delta + self.tau)

            constraints.append(OrientationConstraint(
                index=idx,
                delta=float(delta),
                lower=float(lower),
                upper=float(upper)
            ))
            indices.append(idx)

        return constraints, indices

    def enumerate_candidate_angles(
        self,
        correspondences: Iterable[Correspondence],
        include_midpoint: bool = True
    ) -> List[Tuple[float, List[int]]]:
        """
        Enumerate deterministic candidate rotation angles from constraints.

        Returns:
            List of (angle, active_constraint_indices) candidates
        """
        constraints = self.compute_constraints(correspondences)

        if len(constraints) < 2:
            if len(constraints) == 0:
                return [(0.0, [])]
            single = constraints[0]
            center = wrap_angle((single.lower + single.upper) / 2.0)
            return [(center, [single.index])]

        candidates: List[Tuple[float, List[int]]] = []

        # Boundary angles (activate source constraint)
        for c in constraints:
            candidates.append((c.lower, [c.index]))
            candidates.append((c.upper, [c.index]))

        # Pairwise overlaps
        n = len(constraints)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = self._interval_intersection(constraints[i], constraints[j])
                if overlap is None:
                    continue
                lower, upper = overlap
                center = lower if lower == upper else wrap_angle((lower + upper) / 2.0)
                candidates.append((center, [constraints[i].index, constraints[j].index]))

        if include_midpoint:
            aggregate = self._aggregate_interval(constraints)
            if aggregate is not None:
                lower, upper = aggregate
                center = lower if lower == upper else wrap_angle((lower + upper) / 2.0)
                active = [c.index for c in constraints if c.contains(center)]
                candidates.append((center, active))

        # Deduplicate angles (tolerance 1e-6)
        unique_angles: List[Tuple[float, List[int]]] = []
        seen: List[float] = []
        for angle, active in candidates:
            normalized = wrap_angle(angle)
            if any(abs(wrap_angle(normalized - prev)) < 1e-6 for prev in seen):
                continue
            seen.append(normalized)
            unique_angles.append((normalized, active))

        return unique_angles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _angular_difference(
        self,
        angle1: np.ndarray,
        angle2: float
    ) -> np.ndarray:
        """Compute angular difference in [-π, π]."""
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def _estimate_dominant_orientation_robust(
        self,
        theta_diffs: np.ndarray,
        n_trials: int = 50
    ) -> Tuple[float, np.ndarray]:
        """Robust estimation using stochastic sampling (legacy path)."""
        if theta_diffs.size == 0:
            return 0.0, np.zeros(0, dtype=bool)

        best_inliers = -1
        best_angle = float(circmean(theta_diffs))
        best_mask = np.ones(theta_diffs.size, dtype=bool)

        for _ in range(n_trials):
            idx = np.random.randint(theta_diffs.size)
            candidate = float(theta_diffs[idx])

            diffs = np.abs(self._angular_difference(theta_diffs, candidate))
            inlier_mask = diffs < self.tau
            n_inliers = int(np.sum(inlier_mask))

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_angle = float(circmean(theta_diffs[inlier_mask]))
                best_mask = inlier_mask

        return best_angle, best_mask

    def _estimate_dominant_orientation_deterministic(
        self,
        constraints: List[OrientationConstraint]
    ) -> Tuple[float, np.ndarray]:
        """Deterministic orientation estimate via interval aggregation."""
        if not constraints:
            return 0.0, np.zeros(0, dtype=bool)

        aggregate = self._aggregate_interval(constraints)
        if aggregate is not None:
            lower, upper = aggregate
            dominant = lower if lower == upper else wrap_angle((lower + upper) / 2.0)
        else:
            deltas = np.array([c.delta for c in constraints])
            dominant = float(np.median(deltas))

        mask = np.array([c.contains(dominant) for c in constraints], dtype=bool)
        return dominant, mask

    def _interval_intersection(
        self,
        c1: OrientationConstraint,
        c2: OrientationConstraint
    ) -> Optional[Tuple[float, float]]:
        """Intersection of two wrapped intervals. Returns None if empty."""
        samples = np.linspace(-np.pi, np.pi, num=720, endpoint=False)
        mask = np.array([c1.contains(a) and c2.contains(a) for a in samples], dtype=bool)
        if not mask.any():
            return None
        indices = np.where(mask)[0]
        return float(samples[indices[0]]), float(samples[indices[-1]])

    def _aggregate_interval(
        self,
        constraints: List[OrientationConstraint]
    ) -> Optional[Tuple[float, float]]:
        """Intersection interval across all constraints (if non-empty)."""
        samples = np.linspace(-np.pi, np.pi, num=720, endpoint=False)
        mask = np.ones(samples.shape, dtype=bool)
        for constraint in constraints:
            mask &= np.array([constraint.contains(a) for a in samples], dtype=bool)
            if not mask.any():
                return None
        indices = np.where(mask)[0]
        return float(samples[indices[0]]), float(samples[indices[-1]])

    def _compute_statistics(
        self,
        theta_diffs: np.ndarray
    ) -> OrientationStatistics:
        """Compute circular statistics for orientation differences."""
        if theta_diffs.size == 0:
            return OrientationStatistics(
                mean_angle=0.0,
                concentration=0.0,
                std=np.pi,
                n_samples=0
            )

        mean_angle = float(circmean(theta_diffs))
        std = float(circstd(theta_diffs))

        if std < 0.1:
            concentration = float('inf') if std == 0.0 else 1.0 / (std ** 2)
        else:
            R = float(np.mean(np.cos(theta_diffs - mean_angle)))
            concentration = max(0.0, R)

        return OrientationStatistics(
            mean_angle=mean_angle,
            concentration=float(concentration),
            std=std,
            n_samples=int(theta_diffs.size)
        )

    # ------------------------------------------------------------------
    # Visualization helper (unchanged legacy utility)
    # ------------------------------------------------------------------

    def visualize_orientation_distribution(
        self,
        correspondences: List[Correspondence],
        output_path: Optional[str] = None
    ):
        """Visualize orientation distribution (for debugging)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return

        constraints = self.compute_constraints(correspondences)
        if not constraints:
            print("No orientation information available")
            return

        theta_diffs = np.array([c.delta for c in constraints])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(np.degrees(theta_diffs), bins=36, edgecolor='black', alpha=0.7)
        ax1.axvline(
            np.degrees(circmean(theta_diffs)),
            color='red',
            linestyle='--',
            label='Circular mean'
        )
        ax1.axvline(
            np.degrees(circmean(theta_diffs) - self.tau),
            color='green',
            linestyle=':',
            label=f'±τ ({np.degrees(self.tau):.1f}°)'
        )
        ax1.axvline(
            np.degrees(circmean(theta_diffs) + self.tau),
            color='green',
            linestyle=':'
        )
        ax1.set_xlabel('Orientation Difference (degrees)')
        ax1.set_ylabel('Count')
        ax1.set_title('Orientation Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(122, projection='polar')
        ax2.hist(theta_diffs, bins=36, edgecolor='black', alpha=0.7)
        ax2.set_title('Circular Distribution')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()


def compute_orientation_polytope_vertices(
    correspondences: List[Correspondence],
    tau: float = 0.3
) -> List[Tuple[float, List[int]]]:
    """
    Convenience wrapper for deterministic candidate enumeration.

    Returns:
        List of (rotation_angle, active_constraint_indices)
    """
    filter_obj = OrientationFilter(tau=tau, use_robust_estimation=False)
    return filter_obj.enumerate_candidate_angles(correspondences)


if __name__ == "__main__":
    print("=== Orientation Filter Example ===\n")

    np.random.seed(42)
    n_inliers = 20
    n_outliers = 5

    correspondences = []

    true_rotation = np.radians(45)
    for _ in range(n_inliers):
        corr = Correspondence(
            p_src=np.random.rand(2) * 100,
            p_dst=np.random.rand(2) * 100,
            orientation_src=np.random.rand() * 2 * np.pi,
            orientation_dst=None
        )
        corr.orientation_dst = corr.orientation_src + true_rotation + np.random.randn() * 0.1
        correspondences.append(corr)

    for _ in range(n_outliers):
        correspondences.append(Correspondence(
            p_src=np.random.rand(2) * 100,
            p_dst=np.random.rand(2) * 100,
            orientation_src=np.random.rand() * 2 * np.pi,
            orientation_dst=np.random.rand() * 2 * np.pi
        ))

    print(f"Total correspondences: {len(correspondences)}")
    print(f"  Inliers: {n_inliers}")
    print(f"  Outliers: {n_outliers}")
    print(f"True rotation: {np.degrees(true_rotation):.2f}°\n")

    deterministic_filter = OrientationFilter(
        tau=np.radians(17),
        use_robust_estimation=False
    )
    filtered, stats = deterministic_filter.filter_correspondences(correspondences)

    print(f"After deterministic filtering: {len(filtered)} correspondences")
    print(f"Mean angle: {np.degrees(stats.mean_angle):.2f}°")
    print(f"Std (deg): {np.degrees(stats.std):.2f}")

    candidates = deterministic_filter.enumerate_candidate_angles(correspondences)
    print("\nCandidate angles (degrees):")
    for angle, active in candidates[:10]:
        print(f"  {np.degrees(angle):.2f}° from constraints {active}")

    print("\nDone.")
