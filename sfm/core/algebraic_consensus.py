"""
Algebraic Consensus for Geometric Verification

This is the main module implementing the algebraic consensus pipeline
for geometric verification in Structure-from-Motion.

Pipeline:
    1. Orientation pre-filtering (Theorem 3: Convex polytope)
    2. Closed-form affine solver on minimal sets (Theorem 1: Algebraic certificates)

Author: Claude Code
Date: 2025-10-30
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import time
import logging

from .polynomial_system import (
    Correspondence,
    AffineTransformation,
    PolynomialSystemBuilder
)
from .groebner_solver import (
    GroebnerSolver,
    GroebnerResult,
    ConsistencyStatus
)
from .orientation_filter import (
    OrientationFilter,
    OrientationStatistics,
    wrap_angle
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Complete result from algebraic consensus verification.

    Attributes:
        inlier_ratio: Ratio of inliers [0, 1]
        transformation: Best affine transformation
        n_inliers: Number of inliers
        n_total: Total number of correspondences
        certificate: Algebraic certificate (if available)
        runtime: Total runtime (seconds)
        orientation_stats: Orientation filtering statistics
        method: Method used ('closed_form' or 'groebner')
    """
    inlier_ratio: float
    transformation: Optional[AffineTransformation]
    n_inliers: int
    n_total: int
    certificate: Optional[str] = None
    runtime: float = 0.0
    orientation_stats: Optional[OrientationStatistics] = None
    method: str = 'unknown'

    def __repr__(self) -> str:
        return (
            f"VerificationResult(\n"
            f"  inlier_ratio={self.inlier_ratio:.3f},\n"
            f"  n_inliers={self.n_inliers}/{self.n_total},\n"
            f"  method='{self.method}',\n"
            f"  runtime={self.runtime*1000:.2f}ms\n"
            f")"
        )


class AlgebraicConsensus:
    """
    Main class for algebraic consensus-based geometric verification.

    This implements a clean, compact pipeline:
        - Training-free (no deep learning)
        - Deterministic smart mode (exhaustive algebraic search)
        - Optional stochastic fallback (RANSAC with closed-form solver)
    """

    def __init__(
        self,
        # Orientation filtering
        orientation_tau: float = 0.3,  # ~17 degrees
        use_orientation_filter: bool = True,
        mode: str = 'hybrid',

        # RANSAC parameters
        n_trials: int = 100,
        inlier_threshold: float = 5.0,  # pixels
        confidence: float = 0.99,

        # Solver options
        use_closed_form: bool = True,  # Use fast closed-form solver (recommended)

        # Advanced options
        adaptive_threshold: bool = True,
        min_inliers: int = 3,
        max_deterministic_combinations: Optional[int] = 500,
        hybrid_min_ratio: float = 0.1
    ):
        """
        Initialize algebraic consensus verifier.

        Args:
            orientation_tau: Orientation threshold (radians)
            use_orientation_filter: Enable orientation pre-filtering
            mode: 'deterministic' for exhaustive algebraic search,
                  'ransac' for stochastic sampling,
                  'hybrid' (default) for deterministic first with RANSAC fallback
            n_trials: Maximum RANSAC trials
            inlier_threshold: Inlier threshold (pixels)
            confidence: RANSAC confidence level
            use_closed_form: Use closed-form solver (faster than Gröbner)
            adaptive_threshold: Adapt threshold to image resolution
            min_inliers: Minimum inliers for valid result
            max_deterministic_combinations: Cap on minimal-set combinations (None for unlimited)
            hybrid_min_ratio: Minimum inlier ratio required to keep deterministic result before fallback
        """
        valid_modes = {'deterministic', 'ransac', 'hybrid'}
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        # Orientation filter
        self.use_orientation_filter = use_orientation_filter
        orientation_robust = mode != 'deterministic'
        self.orientation_filter = OrientationFilter(
            tau=orientation_tau,
            min_consensus=min_inliers,
            use_robust_estimation=orientation_robust
        )

        # Gröbner solver
        self.groebner_solver = GroebnerSolver(
            monomial_order='lex',
            numerical_threshold=1e-6,
            use_template=True
        )

        # RANSAC parameters
        self.n_trials = n_trials
        self.inlier_threshold = inlier_threshold
        self.confidence = confidence
        self.use_closed_form = use_closed_form
        self.adaptive_threshold = adaptive_threshold
        self.min_inliers = min_inliers
        self.mode = mode
        self.max_deterministic_combinations = max_deterministic_combinations
        self.hybrid_min_ratio = hybrid_min_ratio

        # Statistics
        self.stats = {
            'n_calls': 0,
            'total_runtime': 0.0,
            'n_orientation_filtered': 0,
            'n_groebner_calls': 0,
            'n_sos_calls': 0,
            'n_deterministic_combos': 0,
            'n_hybrid_fallbacks': 0,
            'n_ransac_calls': 0
        }

    def verify_pair(
        self,
        correspondences: List[Correspondence],
        image_shape: Optional[Tuple[int, int]] = None
    ) -> VerificationResult:
        """
        Main verification function.

        Args:
            correspondences: List of keypoint correspondences
            image_shape: (height, width) for adaptive threshold

        Returns:
            VerificationResult with inlier ratio and transformation
        """
        start_time = time.time()
        self.stats['n_calls'] += 1

        # Early exit for insufficient matches
        if len(correspondences) < self.min_inliers:
            return VerificationResult(
                inlier_ratio=0.0,
                transformation=None,
                n_inliers=0,
                n_total=len(correspondences),
                certificate="Insufficient correspondences",
                runtime=time.time() - start_time,
                method='none'
            )

        # Phase 1: Orientation pre-filtering
        if self.use_orientation_filter:
            filtered_corrs, orientation_stats = self.orientation_filter.filter_correspondences(
                correspondences
            )
            self.stats['n_orientation_filtered'] += (len(correspondences) - len(filtered_corrs))
        else:
            filtered_corrs = correspondences
            orientation_stats = None

        if len(filtered_corrs) < self.min_inliers:
            return VerificationResult(
                inlier_ratio=0.0,
                transformation=None,
                n_inliers=0,
                n_total=len(correspondences),
                certificate="No orientation-consistent correspondences",
                runtime=time.time() - start_time,
                orientation_stats=orientation_stats,
                method='orientation_filter'
            )

        # Adaptive threshold
        threshold = self._compute_adaptive_threshold(image_shape)

        # Phase 2: Deterministic algebraic search and optional fallback
        certificate_parts = []
        best_transformation: Optional[AffineTransformation] = None
        best_inliers = 0
        method = 'none'
        deterministic_ratio = 0.0
        limit_hit = False

        if self.mode in {'deterministic', 'hybrid'}:
            det_transformation, det_inliers, det_certificate, combos_tried, limit_hit = self._verify_deterministic(
                filtered_corrs,
                correspondences,
                threshold,
                orientation_stats
            )
            self.stats['n_deterministic_combos'] += combos_tried

            if det_transformation is not None:
                best_transformation = det_transformation
                best_inliers = det_inliers
                method = 'deterministic'
                deterministic_ratio = det_inliers / len(correspondences)

            if det_certificate:
                certificate_parts.append(det_certificate)

        fallback_used = False
        if self.mode == 'hybrid':
            need_fallback = (
                best_transformation is None
                or deterministic_ratio < self.hybrid_min_ratio
                or (limit_hit and deterministic_ratio < (self.hybrid_min_ratio * 1.5))
            )
            if need_fallback:
                fallback_used = True
                self.stats['n_hybrid_fallbacks'] += 1
                fallback_transformation, fallback_inliers, fallback_method, fallback_certificate = self._verify_ransac(
                    filtered_corrs,
                    correspondences,
                    threshold
                )

                if fallback_certificate:
                    certificate_parts.append(f"fallback={fallback_method}; {fallback_certificate}")
                elif fallback_method != 'none':
                    certificate_parts.append(f"fallback={fallback_method}")

                if fallback_transformation is not None and fallback_inliers > best_inliers:
                    best_transformation = fallback_transformation
                    best_inliers = fallback_inliers
                    method = f"hybrid_{fallback_method}"

        elif self.mode == 'ransac':
            best_transformation, best_inliers, method, method_certificate = self._verify_ransac(
                filtered_corrs,
                correspondences,
                threshold
            )
            if method_certificate:
                certificate_parts.append(method_certificate)

        if fallback_used and method == 'deterministic':
            method = 'hybrid_det'

        if method == 'none' and best_transformation is None and fallback_used:
            method = 'hybrid_ransac'

        certificate = "; ".join([part for part in certificate_parts if part]) if certificate_parts else None

        runtime = time.time() - start_time
        self.stats['total_runtime'] += runtime

        return VerificationResult(
            inlier_ratio=(best_inliers / len(correspondences)) if correspondences else 0.0,
            transformation=best_transformation,
            n_inliers=best_inliers,
            n_total=len(correspondences),
            certificate=certificate,
            runtime=runtime,
            orientation_stats=orientation_stats,
            method=method
        )

    def _compute_adaptive_threshold(
        self,
        image_shape: Optional[Tuple[int, int]]
    ) -> float:
        """
        Compute adaptive inlier threshold based on image resolution.

        Args:
            image_shape: (height, width)

        Returns:
            Threshold in pixels
        """
        if not self.adaptive_threshold or image_shape is None:
            return self.inlier_threshold

        # Threshold = 0.7% of image diagonal
        h, w = image_shape
        diagonal = np.sqrt(h**2 + w**2)
        adaptive = 0.007 * diagonal

        # Clamp to [5px, 20px]
        return np.clip(adaptive, 5.0, 20.0)

    def _compute_num_trials(
        self,
        n_matches: int,
        epsilon: float = 0.2
    ) -> int:
        """
        Compute number of RANSAC trials needed.

        After orientation filtering, effective inlier ratio is higher,
        so fewer trials are needed.

        Args:
            n_matches: Number of matches
            epsilon: Expected outlier ratio

        Returns:
            Number of trials
        """
        import math

        # Effective inlier ratio after filtering
        epsilon_effective = max(epsilon, 0.4)

        # Standard RANSAC formula
        trials = math.log(1 - self.confidence) / math.log(1 - epsilon_effective**3)

        # Clamp to reasonable range
        return max(10, min(int(trials), self.n_trials))

    def _sample_minimal_set(
        self,
        correspondences: List[Correspondence],
        size: int = 3
    ) -> List[Correspondence]:
        """
        Sample minimal set for affine transformation.

        Args:
            correspondences: List of matches
            size: Size of minimal set (3 for affine)

        Returns:
            Sampled minimal set
        """
        import random
        return random.sample(correspondences, size)

    def _verify_ransac(
        self,
        filtered_corrs: List[Correspondence],
        all_corrs: List[Correspondence],
        threshold: float
    ) -> Tuple[Optional[AffineTransformation], int, str, Optional[str]]:
        """Stochastic verification using random minimal set sampling."""
        self.stats['n_ransac_calls'] += 1
        best_transformation = None
        best_inliers = 0
        best_method = 'none'
        best_certificate = None

        n_trials = self._compute_num_trials(len(filtered_corrs))

        for _ in range(n_trials):
            minimal_set = self._sample_minimal_set(filtered_corrs, size=3)

            if self.use_closed_form:
                result = self.groebner_solver.solve_closed_form(minimal_set)
                method = 'closed_form'
            else:
                result = self.groebner_solver.solve_minimal_set(minimal_set)
                method = 'groebner'
                self.stats['n_groebner_calls'] += 1

            if result.status != ConsistencyStatus.CONSISTENT:
                continue

            n_inliers, _ = self.groebner_solver.count_inliers(
                result.transformation,
                filtered_corrs,
                threshold=threshold
            )

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_transformation = result.transformation
                best_method = method
                best_certificate = result.certificate

        if best_transformation is not None:
            final_inliers, _ = self.groebner_solver.count_inliers(
                best_transformation,
                all_corrs,
                threshold=threshold
            )
        else:
            final_inliers = 0

        return best_transformation, final_inliers, best_method, best_certificate

    def _verify_deterministic(
        self,
        filtered_corrs: List[Correspondence],
        all_corrs: List[Correspondence],
        threshold: float,
        orientation_stats: Optional[OrientationStatistics]
    ) -> Tuple[Optional[AffineTransformation], int, str, int, bool]:
        """Deterministic verification via exhaustive minimal-set enumeration."""
        from itertools import combinations

        if len(filtered_corrs) < 3:
            return None, 0, "deterministic combos=0 (insufficient data)", 0, False

        candidate_angles = self.orientation_filter.enumerate_candidate_angles(filtered_corrs)
        ordered_indices = self._deterministic_order(filtered_corrs, orientation_stats)

        max_combos = self.max_deterministic_combinations
        combos_tried = 0
        best_transformation = None
        best_inliers = 0
        best_certificate = None
        limit_hit = False

        for combo in combinations(ordered_indices, 3):
            if max_combos is not None and combos_tried >= max_combos:
                limit_hit = True
                break

            combos_tried += 1
            minimal_set = [filtered_corrs[idx] for idx in combo]
            result = self.groebner_solver.solve_closed_form(minimal_set)

            if result.status != ConsistencyStatus.CONSISTENT:
                continue

            n_inliers, _ = self.groebner_solver.count_inliers(
                result.transformation,
                all_corrs,
                threshold=threshold
            )

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_transformation = result.transformation
                best_certificate = result.certificate

        if best_transformation is None:
            certificate = (
                f"deterministic combos={combos_tried} "
                f"(orientation_candidates={len(candidate_angles)}) – no consistent solution"
            )
            return None, 0, certificate, combos_tried, limit_hit

        certificate_parts = [
            f"deterministic combos={combos_tried}",
            f"orientation_candidates={len(candidate_angles)}"
        ]
        if best_certificate:
            certificate_parts.append(best_certificate)

        certificate = "; ".join(certificate_parts)
        if max_combos is not None and combos_tried >= max_combos:
            limit_hit = True
        return best_transformation, best_inliers, certificate, combos_tried, limit_hit

    def _deterministic_order(
        self,
        correspondences: List[Correspondence],
        orientation_stats: Optional[OrientationStatistics]
    ) -> List[int]:
        """
        Sort correspondences deterministically using orientation proximity and scores.
        """
        target_angle = 0.0
        if orientation_stats and orientation_stats.n_samples > 0 and orientation_stats.is_concentrated:
            target_angle = orientation_stats.mean_angle

        def score_tuple(item: Tuple[int, Correspondence]) -> Tuple[int, float, float, int]:
            idx, corr = item

            # Orientation proximity
            if corr.orientation_src is None or corr.orientation_dst is None:
                orientation_rank = 1
                orientation_gap = float(idx)
            else:
                delta = wrap_angle((corr.orientation_dst - corr.orientation_src) - target_angle)
                orientation_rank = 0
                orientation_gap = abs(delta)

            # Prefer higher detector scores if available
            score_src = -(corr.score_src or 0.0)
            score_dst = -(corr.score_dst or 0.0)

            return (orientation_rank, orientation_gap, score_src + score_dst, idx)

        ordered = sorted(enumerate(correspondences), key=score_tuple)
        return [idx for idx, _ in ordered]

    def get_statistics(self) -> Dict[str, float]:
        """Get usage statistics."""
        if self.stats['n_calls'] == 0:
            return self.stats

        avg_runtime = self.stats['total_runtime'] / self.stats['n_calls'] * 1000  # ms

        return {
            **self.stats,
            'avg_runtime_ms': avg_runtime
        }

    def reset_statistics(self):
        """Reset usage statistics."""
        self.stats = {
            'n_calls': 0,
            'total_runtime': 0.0,
            'n_orientation_filtered': 0,
            'n_groebner_calls': 0,
            'n_sos_calls': 0,
            'n_deterministic_combos': 0,
            'n_hybrid_fallbacks': 0,
            'n_ransac_calls': 0
        }


def convert_matches_to_correspondences(
    hits: List[Tuple[int, int]],
    kpts_src: Dict,
    kpts_dst: Dict
) -> List[Correspondence]:
    """
    Convert vocabulary tree hits to Correspondence objects.

    Args:
        hits: List of (src_idx, dst_idx) match pairs
        kpts_src: Source keypoints dict with keys:
            - 'keypoints': Nx2 array
            - 'scores': N array (optional)
            - 'orientations': N array (optional)
        kpts_dst: Destination keypoints dict

    Returns:
        List of Correspondence objects
    """
    correspondences = []

    keypoints_src = kpts_src['keypoints']
    keypoints_dst = kpts_dst['keypoints']

    n_src = keypoints_src.shape[0]
    n_dst = keypoints_dst.shape[0]

    skipped = 0

    for src_idx, dst_idx in hits:
        if not (0 <= src_idx < n_src) or not (0 <= dst_idx < n_dst):
            skipped += 1
            continue

        # Extract keypoint coordinates
        p_src = keypoints_src[src_idx]
        p_dst = keypoints_dst[dst_idx]

        # Extract orientations (if available)
        orientation_src = None
        orientation_dst = None
        if 'orientations' in kpts_src and 'orientations' in kpts_dst:
            orientation_src = kpts_src['orientations'][src_idx]
            orientation_dst = kpts_dst['orientations'][dst_idx]

        # Extract scores (if available)
        score_src = None
        score_dst = None
        if 'scores' in kpts_src and 'scores' in kpts_dst:
            score_src = kpts_src['scores'][src_idx]
            score_dst = kpts_dst['scores'][dst_idx]

        correspondences.append(Correspondence(
            p_src=p_src,
            p_dst=p_dst,
            orientation_src=orientation_src,
            orientation_dst=orientation_dst,
            score_src=score_src,
            score_dst=score_dst
        ))

    if skipped:
        logger.warning(
            "Dropped %d correspondences with out-of-bounds indices (src<%d, dst<%d)",
            skipped,
            n_src,
            n_dst
        )

    return correspondences


if __name__ == "__main__":
    print("=== Algebraic Consensus Example ===\n")

    # Create synthetic test data
    np.random.seed(42)

    # True transformation
    true_rotation = np.radians(30)
    true_A = np.array([
        [np.cos(true_rotation), -np.sin(true_rotation)],
        [np.sin(true_rotation), np.cos(true_rotation)]
    ])
    true_t = np.array([10.0, 5.0])
    true_transform = AffineTransformation(A=true_A, t=true_t)

    print(f"True transformation:")
    print(f"  Rotation: {np.degrees(true_rotation):.2f}°")
    print(f"  Translation: {true_t}\n")

    # Generate inliers
    n_inliers = 50
    n_outliers = 20

    correspondences = []

    for _ in range(n_inliers):
        # Random source point
        p_src = np.random.rand(2) * 100

        # Transform to destination
        p_dst = true_A @ p_src + true_t + np.random.randn(2) * 0.5  # Small noise

        # Orientation (consistent with rotation)
        ori_src = np.random.rand() * 2 * np.pi
        ori_dst = ori_src + true_rotation + np.random.randn() * 0.05

        correspondences.append(Correspondence(
            p_src=p_src,
            p_dst=p_dst,
            orientation_src=ori_src,
            orientation_dst=ori_dst,
            score_src=np.random.rand(),
            score_dst=np.random.rand()
        ))

    # Generate outliers
    for _ in range(n_outliers):
        correspondences.append(Correspondence(
            p_src=np.random.rand(2) * 100,
            p_dst=np.random.rand(2) * 100,  # Random, not consistent
            orientation_src=np.random.rand() * 2 * np.pi,
            orientation_dst=np.random.rand() * 2 * np.pi
        ))

    print(f"Test data:")
    print(f"  Inliers: {n_inliers}")
    print(f"  Outliers: {n_outliers}")
    print(f"  Total: {len(correspondences)}\n")

    # Run verification
    verifier = AlgebraicConsensus(
        orientation_tau=np.radians(17),
        use_orientation_filter=True,
        mode='deterministic',  # Deterministic mode (no RANSAC!)
        inlier_threshold=5.0,
        use_closed_form=True,
        max_deterministic_combinations=500
    )

    result = verifier.verify_pair(
        correspondences,
        image_shape=(480, 640)
    )

    print(f"Verification result:")
    print(result)

    if result.transformation:
        print(f"\nRecovered transformation:")
        print(f"  A =\n{result.transformation.A}")
        print(f"  t = {result.transformation.t}")
        print(f"  Rotation: {np.degrees(result.transformation.rotation_angle):.2f}°")

        print(f"\nError vs. ground truth:")
        A_error = np.linalg.norm(result.transformation.A - true_A, 'fro')
        t_error = np.linalg.norm(result.transformation.t - true_t)
        print(f"  ||A - A_true||_F = {A_error:.4f}")
        print(f"  ||t - t_true||_2 = {t_error:.4f}")

    if result.orientation_stats:
        print(f"\nOrientation statistics:")
        print(f"  Mean angle: {np.degrees(result.orientation_stats.mean_angle):.2f}°")
        print(f"  Std: {np.degrees(result.orientation_stats.std):.2f}°")
        print(f"  Concentration: {result.orientation_stats.concentration:.3f}")

    # Show statistics
    print(f"\nVerifier statistics:")
    for key, value in verifier.get_statistics().items():
        print(f"  {key}: {value}")
