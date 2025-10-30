# Algebraic Consensus – Strategy Notes

These notes answer the three open questions around the Gröbner branch and the overall design philosophy. Everything here stays within the *mathematically simple but beautiful* spirit of the module.

## 1. Making the Gröbner path usable as a default

- **Template precomputation** – We already generate the affine elimination template; extend that idea to the Gröbner branch by caching the polynomial system structure (monomial basis, coefficient matrix) once per correspondence ordering. Reuse it across minimal sets by only updating numeric coefficients before calling `groebner`. This removes repeated SymPy graph construction.
- **Domain-aware solving** – Restrict SymPy to `QQ` when inputs are float32/float64 but quantised (e.g., by pre-scaling coordinates to integers). This avoids expensive algebraic number reconstruction and keeps the basis in rational arithmetic.
- **Hybrid solver schedule** – Run the closed-form template first, and only send a minimal set to Gröbner if the template reports a warning (denominator magnitude < ε, ill-conditioned matrix, or det(A)≈0). With proper health checks the Gröbner branch is exercised on <1% of cases yet remains the mathematically authoritative fallback.
- **Batching & parallelism** – Collect "Gröbner candidates" during deterministic enumeration and evaluate them in a worker pool. Each job is CPU-bound but independent; a simple `ProcessPoolExecutor` gives almost linear speed-up for those rare calls.
- **Early rejection heuristics** – Before invoking `groebner`, compute quick invariants (e.g., signed area ratios, cross-ratios). If they disagree beyond a tolerance, skip symbolic elimination entirely and record the algebraic inconsistency.
- **NumPy-friendly residual checks** – When SymPy returns a transformation, validate using vectorised residual evaluation (already available via `count_inliers`). Keep tolerances slightly looser (e.g., 1.5× pixel threshold) so that numerical noise does not ping-pong the solver back to Gröbner.

Together these steps make the symbolic branch responsive enough to keep `use_closed_form=False` within reach for power users, while still advertising the fast template solver as the first line.

## 2. Deterministic alternatives to full Gröbner elimination

- **Resultant-based certificates** – Replace Buchberger with a Sylvester resultant: form the 3×3 affine system, eliminate variables via determinant computation, and test whether the resultant polynomial collapses to zero. This yields a yes/no consistency certificate without a full Gröbner basis.
- **Rational univariate representation (RUR)** – Derive a single-variable polynomial for the rotation angle (using orientation-constrained cross ratios), solve it analytically, and back-substitute for translation. This keeps everything in closed form and exposes each algebraic step.
- **Dual-space linearisation** – Lift correspondences to homogeneous 3D (dual affine space) and solve with SVD under determinant constraints. Degenerate sets still produce a verifiable algebraic certificate (rank drop), but the computation is purely linear algebra.
- **Interval-certified voting** – Enumerate candidate angles via Theorem 3, propagate each interval through the affine template, and keep only the intervals that satisfy bounded residual inequalities. No randomness, yet the process remains explainable as a sequence of convex feasibility checks.

Each option is deterministic, auditable, and meshes well with the existing orientation-first pipeline. They can either complement or substitute the Gröbner branch depending on how strong a certificate you need.

## 3. “Mathematically simple but beautiful” playbook

- **Order is everything** – Keep the pipeline: orientation polytope → minimal-set algebra → numeric verification. Each stage is a theorem-backed filter; no heuristics are introduced without a proof obligation.
- **Expose the certificates** – Every rejection should come with an explanation (`deterministic combos=42; denominator=1.2e-5; resultant≠0`). Surfacing these in logs (and eventually the UI) keeps the user in the loop.
- **Adaptive, not fragile** – Default thresholds (`tau≈17°`, `max_combos≈500`, `min_ratio≈0.2`) are good starting points, but wire them so that the caller can connect dataset-specific schedulers (e.g., lower τ for narrow-baseline videos, raise `max_combos` when many orientations are missing).
- **Deterministic fallbacks** – When deterministic consensus cannot certify enough pairs, immediately fall back to (a) re-running without the orientation gate, (b) relaxed thresholds, or (c) ordered RANSAC with fixed seeds. Always document which fallback succeeded.
- **Smart pair sourcing** – Ensure the vocabulary-tree + sequential mixture stays well-behaved by clipping out-of-bounds indices and re-weighting pair scores based on consensus statistics. That keeps the later algebraic steps from getting poisoned by upstream bookkeeping errors.
- **Two-stage default** – Ship a quick deterministic pass (small combo budget) followed by a reproducible RANSAC fallback; surface which stage produced the certificate so users know when the bridge was needed.

Following this checklist allows us to ship a solver that reads like a short mathematical argument, yet survives real-world noise and scale.
