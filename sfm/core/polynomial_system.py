"""
Polynomial System Formulation for Geometric Verification

This module converts geometric constraints (affine transformations, orientations)
into polynomial systems suitable for Gröbner basis computation.

Mathematical Framework:
    Affine transformation: p' = A·p + t
    Polynomial constraint: g_i(A,t) = p_i' - A·p_i - t = 0

Author: Claude Code
Date: 2025-10-30
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from sympy import symbols, Matrix, Expr, expand
from sympy.core.symbol import Symbol


@dataclass
class AffineTransformation:
    """
    Represents an affine transformation in 2D.

    Attributes:
        A: 2x2 rotation/scale matrix
        t: 2D translation vector
    """
    A: np.ndarray  # Shape: (2, 2)
    t: np.ndarray  # Shape: (2,)

    def __post_init__(self):
        """Validate dimensions."""
        assert self.A.shape == (2, 2), f"A must be 2x2, got {self.A.shape}"
        assert self.t.shape == (2,), f"t must be 2D, got {self.t.shape}"

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to points.

        Args:
            points: Nx2 array of points

        Returns:
            Transformed points (Nx2)
        """
        return (self.A @ points.T).T + self.t

    @property
    def rotation_angle(self) -> float:
        """Extract rotation angle from transformation matrix."""
        return np.arctan2(self.A[1, 0], self.A[0, 0])


@dataclass
class Correspondence:
    """
    A single keypoint correspondence between two images.

    Attributes:
        p_src: Source point (x, y)
        p_dst: Destination point (x, y)
        orientation_src: Keypoint orientation in source image (radians)
        orientation_dst: Keypoint orientation in destination image (radians)
        score_src: Detection confidence in source image [0, 1]
        score_dst: Detection confidence in destination image [0, 1]
    """
    p_src: np.ndarray  # Shape: (2,)
    p_dst: np.ndarray  # Shape: (2,)
    orientation_src: Optional[float] = None
    orientation_dst: Optional[float] = None
    score_src: Optional[float] = None
    score_dst: Optional[float] = None

    def __post_init__(self):
        """Validate dimensions."""
        self.p_src = np.asarray(self.p_src, dtype=np.float64)
        self.p_dst = np.asarray(self.p_dst, dtype=np.float64)
        assert self.p_src.shape == (2,), f"p_src must be 2D, got {self.p_src.shape}"
        assert self.p_dst.shape == (2,), f"p_dst must be 2D, got {self.p_dst.shape}"


class PolynomialSystemBuilder:
    """
    Builds polynomial systems from geometric constraints.

    This class converts affine transformation constraints into polynomial
    equations suitable for Gröbner basis computation.
    """

    def __init__(self):
        """Initialize symbolic variables for affine transformation."""
        # Symbolic variables: a11, a12, a21, a22, tx, ty
        self.a11, self.a12 = symbols('a11 a12', real=True)
        self.a21, self.a22 = symbols('a21 a22', real=True)
        self.tx, self.ty = symbols('tx ty', real=True)

        # Symbolic transformation matrix
        self.A_symbolic = Matrix([
            [self.a11, self.a12],
            [self.a21, self.a22]
        ])
        self.t_symbolic = Matrix([self.tx, self.ty])

    @property
    def variables(self) -> List[Symbol]:
        """Return list of symbolic variables in canonical order."""
        return [self.a11, self.a12, self.a21, self.a22, self.tx, self.ty]

    def build_affine_constraints(
        self,
        correspondences: List[Correspondence]
    ) -> List[Expr]:
        """
        Build polynomial constraints from affine transformation equation.

        For each correspondence (p_src, p_dst), generates:
            g_i(A,t) = p_dst - A·p_src - t = 0

        Args:
            correspondences: List of keypoint matches

        Returns:
            List of polynomial expressions (2 per correspondence: x and y components)
        """
        polynomials = []

        for corr in correspondences:
            # Convert numpy arrays to SymPy Matrix
            p_src = Matrix(corr.p_src.tolist())
            p_dst = Matrix(corr.p_dst.tolist())

            # Constraint: p_dst = A * p_src + t
            # Rearranged: p_dst - A * p_src - t = 0
            constraint = p_dst - (self.A_symbolic * p_src + self.t_symbolic)

            # Extract x and y components
            gx = expand(constraint[0])
            gy = expand(constraint[1])

            polynomials.append(gx)
            polynomials.append(gy)

        return polynomials

    def build_orientation_constraints(
        self,
        correspondences: List[Correspondence],
        tau: float = 0.3
    ) -> List[Expr]:
        """
        Build polynomial constraints from orientation consistency.

        For each correspondence with orientation, generates:
            |θ_dst - θ_src - θ(A)| ≤ τ
        where θ(A) = arctan(a21/a11) is the rotation angle of A.

        Note: This returns trigonometric polynomials which increase
        the effective degree. For large systems, orientation filtering
        should be done numerically as a pre-processing step.

        Args:
            correspondences: List of keypoint matches with orientations
            tau: Orientation threshold (radians)

        Returns:
            List of orientation constraint polynomials
        """
        from sympy import atan2, Abs, cos, sin

        polynomials = []

        # Rotation angle of transformation
        theta_A = atan2(self.a21, self.a11)

        for corr in correspondences:
            if corr.orientation_src is None or corr.orientation_dst is None:
                continue

            # Expected orientation change
            theta_expected = corr.orientation_dst - corr.orientation_src

            # Constraint: |theta_expected - theta(A)| ≤ tau
            # This is trigonometric, so we use cos/sin linearization
            # cos(theta_expected - theta_A) ≥ cos(tau)

            diff = theta_expected - theta_A
            constraint = cos(diff) - cos(tau)

            polynomials.append(expand(constraint))

        return polynomials

    def numerical_to_symbolic(
        self,
        A: np.ndarray,
        t: np.ndarray
    ) -> dict:
        """
        Convert numerical affine transformation to symbolic substitution dict.

        Args:
            A: 2x2 transformation matrix
            t: 2D translation vector

        Returns:
            Dictionary mapping symbolic variables to numerical values
        """
        return {
            self.a11: float(A[0, 0]),
            self.a12: float(A[0, 1]),
            self.a21: float(A[1, 0]),
            self.a22: float(A[1, 1]),
            self.tx: float(t[0]),
            self.ty: float(t[1])
        }

    def symbolic_to_numerical(
        self,
        solution: dict
    ) -> AffineTransformation:
        """
        Convert symbolic solution to numerical affine transformation.

        Args:
            solution: Dictionary from SymPy solve()

        Returns:
            AffineTransformation object
        """
        A = np.array([
            [float(solution[self.a11]), float(solution[self.a12])],
            [float(solution[self.a21]), float(solution[self.a22])]
        ], dtype=np.float64)

        t = np.array([
            float(solution[self.tx]),
            float(solution[self.ty])
        ], dtype=np.float64)

        return AffineTransformation(A=A, t=t)

    def evaluate_residuals(
        self,
        transformation: AffineTransformation,
        correspondences: List[Correspondence]
    ) -> np.ndarray:
        """
        Evaluate residuals for a given transformation.

        Args:
            transformation: Affine transformation to test
            correspondences: List of keypoint matches

        Returns:
            Array of residuals (Euclidean distances)
        """
        residuals = []

        for corr in correspondences:
            # Transform source point
            p_predicted = transformation.A @ corr.p_src + transformation.t

            # Compute residual
            residual = np.linalg.norm(p_predicted - corr.p_dst)
            residuals.append(residual)

        return np.array(residuals)


def validate_minimal_set(correspondences: List[Correspondence]) -> bool:
    """
    Validate that a minimal set of correspondences is non-degenerate.

    For affine transformations, we need at least 3 non-collinear points.

    Args:
        correspondences: List of at least 3 correspondences

    Returns:
        True if the set is valid (non-degenerate)
    """
    if len(correspondences) < 3:
        return False

    # Extract source points
    points = np.array([corr.p_src for corr in correspondences[:3]])

    # Check collinearity using cross product
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    # Non-collinear if cross product is non-zero
    return abs(cross) > 1e-6


if __name__ == "__main__":
    # Example usage
    print("=== Polynomial System Builder Example ===\n")

    # Create correspondences
    correspondences = [
        Correspondence(
            p_src=np.array([0.0, 0.0]),
            p_dst=np.array([1.0, 2.0]),
            orientation_src=0.0,
            orientation_dst=0.5
        ),
        Correspondence(
            p_src=np.array([1.0, 0.0]),
            p_dst=np.array([2.0, 3.0])
        ),
        Correspondence(
            p_src=np.array([0.0, 1.0]),
            p_dst=np.array([0.5, 3.0])
        )
    ]

    # Build polynomial system
    builder = PolynomialSystemBuilder()
    polynomials = builder.build_affine_constraints(correspondences)

    print(f"Number of correspondences: {len(correspondences)}")
    print(f"Number of polynomial constraints: {len(polynomials)}")
    print(f"Variables: {builder.variables}\n")

    print("First polynomial constraint:")
    print(polynomials[0])
    print()

    # Validate minimal set
    is_valid = validate_minimal_set(correspondences)
    print(f"Minimal set valid (non-degenerate): {is_valid}")
