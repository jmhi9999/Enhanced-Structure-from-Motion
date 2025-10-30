# Algebraic Correspondence Graph for Robust SfM

**Problem Statement**: 현재 파이프라인은 vocab tree + algebraic consensus로 구성되어 있으나, **low-overlap + low-texture** 환경에서 실패하는 근본적 한계가 있음.

- Vocab tree: 충분한 shared visual words가 없으면 pair 자체를 놓침
- Brute force matching: Exhaustive하지만 N²개 쌍에 LightGlue를 돌리기엔 너무 느림
- Algebraic consensus: Input correspondence가 부족하면 (< 10개) 무용지물

**Goal**: Brute force의 **exhaustiveness** + Algebraic의 **efficiency** + **White-box mathematical framework**

---

## Core Idea: Multi-Stage Algebraic Filtering

```
Stage 1: Algebraic Pre-filtering (Fast)
  → Orientation/spatial compatibility check
  → O(0.1ms) per pair, fully algebraic
  → Reject ~80% of incompatible pairs

Stage 2: Minimal Algebraic Verification (Medium)
  → 3-point polynomial solver
  → O(1ms) per pair, deterministic
  → Further reject ~50% of remaining pairs

Stage 3: Deep Matching (Expensive, but selective)
  → LightGlue on verified pairs only
  → O(50ms) per pair
  → High-quality correspondences

Stage 4: Global Algebraic Optimization (Optional)
  → Multi-view polynomial consistency
  → Cycle constraints as ideal membership
  → Detect and fix incorrect edges
```

---

## What Gets Replaced

### Before (Current Pipeline)
```
[Features]
    ↓
[Vocab Tree] ─────→ Top-200 pairs (BoW similarity)
    ↓
[LightGlue] ──────→ Dense matching (200 × 50ms = 10s)
    ↓
[Algebraic Consensus] → Geometric verification (200 × 5ms = 1s)
    ↓
[COLMAP]

Total: ~11s for 100 images
```

**Limitations**:
- ❌ Vocab tree misses low-overlap pairs (sparse visual words)
- ❌ No control over what pairs are selected (black-box heuristic)
- ❌ Wastes LightGlue on pairs that will fail verification

### After (Algebraic Graph)
```
[Features]
    ↓
[Algebraic Pre-filter] → All 4,950 pairs checked (0.1ms each = 0.5s)
    ↓ (80% rejected)
[Minimal Verification] → 1,000 pairs (1ms each = 1s)
    ↓ (50% rejected)
[LightGlue] ──────────→ 500 pairs (50ms each = 25s)
    ↓
[Global Algebraic Opt] → Consistency check + outlier removal (optional, ~2s)
    ↓
[COLMAP]

Total: ~28.5s for 100 images
```

**Replaced Components**:
1. **Vocab Tree** → Algebraic pre-filtering (orientation polytope)
2. **Fixed top-K** → Adaptive algebraic thresholding
3. **(Optional) Stage 4.5 Consensus** → Integrated into pre-filtering

**Kept Components**:
- Feature extraction (unchanged)
- LightGlue matching (but on fewer, better pairs)
- COLMAP reconstruction (unchanged)

---

## Computation Analysis

### Time Breakdown (100 images, 4,950 pairs)

| Stage | Operation | Per-Pair Time | Total Pairs | Total Time | GPU? |
|-------|-----------|---------------|-------------|------------|------|
| **1. Pre-filter** | Orientation histogram | 0.1ms | 4,950 | **0.5s** | No (CPU) |
| **2. Minimal verify** | 3-pt polynomial solve | 1ms | 1,000 (20%) | **1.0s** | No (CPU) |
| **3. LightGlue** | Deep matching | 50ms | 500 (10%) | **25s** | Yes (GPU) |
| **4. Global opt** | Cycle consistency | - | - | **2s** | No (CPU) |

**Total: ~28.5s** (vs current 11s)

### Scaling (N images)

| N | Pairs | Pre-filter | Verify | LightGlue | Total |
|---|-------|------------|--------|-----------|-------|
| 50 | 1,225 | 0.1s | 0.2s | 6s | **6.3s** |
| 100 | 4,950 | 0.5s | 1.0s | 25s | **26.5s** |
| 200 | 19,900 | 2s | 4s | 100s | **106s** |
| 500 | 124,750 | 12s | 25s | 625s | **662s** (11min) |

### GPU Usage

**Current pipeline**:
- Vocab tree: CPU (NumPy/PyTorch)
- LightGlue: **GPU** (200 pairs × 50ms = 10s GPU time)
- Algebraic consensus: CPU (NumPy/SciPy)

**Algebraic graph**:
- Pre-filtering: CPU (NumPy, vectorized)
- Minimal verification: CPU (NumPy, parallelizable)
- LightGlue: **GPU** (500 pairs × 50ms = 25s GPU time)
- Global opt: CPU (sparse linear algebra)

**GPU Time Increase**: 10s → 25s (2.5×)
**Reason**: More pairs verified = more GPU matching needed

### Optimization Opportunities

1. **Batch GPU Pre-filtering**: Move orientation checks to GPU
   - Current: 0.5s (CPU)
   - Optimized: 0.05s (GPU batched)

2. **Parallel Minimal Verification**: Multiprocessing
   - Current: 1s (single thread)
   - Optimized: 0.2s (5 threads)

3. **Early Stopping**: Stop when enough pairs found
   - Current: Check all 4,950 pairs
   - Optimized: Stop at 500 verified → ~5-10× faster

4. **Sparse Graph Optimization**: Use only high-confidence edges
   - Current: 500 pairs to LightGlue
   - Optimized: 200 pairs (top scores) → 10s GPU time

**Optimized Total**: 0.05s + 0.2s + 10s + 2s = **12.25s** (comparable to current!)

---

## What Problems Does This Solve?

### ✅ Problem 1: Low-Overlap Pairs Missed by Vocab Tree

**Scenario**: Two images with 15% overlap
- Current: Vocab tree gives low BoW score → pair ignored
- Algebraic: Checks all pairs → finds the few shared features → verified

**Why it works**: Exhaustive pair checking (not top-K filtering)

### ✅ Problem 2: Low-Texture Regions

**Scenario**: White wall, sky, uniform surfaces (5-10 features)
- Current: Algebraic consensus fails (too few correspondences)
- Algebraic graph: Uses multi-view constraints

**How**:
```
Image A ← 5 matches → Image B
Image B ← 8 matches → Image C
Image C ← 6 matches → Image A

Individual pairs: FAIL (too few)
Cycle constraint: A→B→C→A polynomial consistency → PASS
```

**Mathematical principle**:
- 3-view geometry has **trilinear constraints**
- Can recover structure with fewer per-pair correspondences
- Polynomial ideal of cycle = stricter verification

### ✅ Problem 3: White-Box Framework

**Current pipeline**:
```
Vocab tree (heuristic TF-IDF)
  ↓
LightGlue (learned transformer)
  ↓
Algebraic consensus (mathematical)
```
→ **Hybrid approach**: Hard to debug when it fails

**Algebraic graph**:
```
Orientation polytope (Theorem 3: convex geometry)
  ↓
Polynomial solving (Theorem 1: algebraic certificates)
  ↓
Global consistency (Ideal membership test)
```
→ **Unified mathematical framework**: Every decision is explainable

**Example failure diagnosis**:
- Current: "Vocab tree didn't return this pair" → Why? Unknown.
- Algebraic: "Orientation polytope empty" → Explicitly: no consistent rotation exists

### ✅ Problem 4: Robustness to Outliers

**Current**: RANSAC removed, deterministic consensus used
- Problem: One bad pair can break COLMAP

**Algebraic graph**: Global consistency check
```python
# Cycle constraint example
# If A→B→C→A doesn't compose to identity, at least one edge is wrong

for cycle in graph.find_cycles():
    T_AB, T_BC, T_CA = get_transformations(cycle)

    # Polynomial constraint: T_AB ∘ T_BC ∘ T_CA = I
    residual = compose(T_AB, T_BC, T_CA) - Identity

    if polynomial_norm(residual) > threshold:
        # At least one edge is inconsistent
        edges_to_verify = rank_by_algebraic_certificate(cycle)
        remove_weakest_edge()
```

This is **provably correct**: Inconsistent cycles are detected algebraically.

### ✅ Problem 5: Adaptive to Dataset

**Current**: Fixed top-K (200 pairs from vocab tree)
- Sparse dataset: Needs more pairs → misses connections
- Dense dataset: Wastes time on redundant pairs

**Algebraic**: Adaptive thresholding
```python
# Keep adding pairs until algebraic score threshold
verified_pairs = []
for pair in all_pairs_sorted_by_score:
    if pair.algebraic_confidence > adaptive_threshold():
        verified_pairs.append(pair)

    if len(verified_pairs) >= min_pairs:
        # Check if graph is well-connected
        if graph_connectivity(verified_pairs) > 0.9:
            break  # Enough pairs
```

---

## Mathematical Foundation

### Theorem 1: Orientation Polytope Compatibility (Fast Pre-filter)

For two images with oriented features, the set of compatible rotations forms a **convex polytope** in rotation space.

**Algorithmic consequence**:
- Check if polytope is non-empty (linear programming)
- O(N) features → O(N) halfspace constraints
- Time: O(N) per pair (vectorized: O(0.1ms))

### Theorem 2: Minimal Polynomial Certificate (Medium Verify)

For 3 correspondences under affine transformation, compatibility is equivalent to **existence of solution** to a degree-2 polynomial system.

**Algorithmic consequence**:
- Precomputed elimination template (offline)
- Runtime: Substitute coefficients + solve 6×6 linear system
- Time: O(1ms) per pair

### Theorem 3: Multi-View Algebraic Consistency (Global Opt)

For a set of pairwise transformations {T_ij}, global consistency is equivalent to **1 ∉ I** where I is the ideal generated by cycle composition polynomials.

**Algorithmic consequence**:
- Build ideal from all cycles in graph
- Gröbner basis test for consistency
- Outlier detection: which edge violates consistency?

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (1-2 days)
- [ ] `AlgebraicPreFilter`: Fast orientation/spatial compatibility
- [ ] `MinimalVerifier`: 3-point polynomial solver
- [ ] `AlgebraicGraph`: Graph construction + management

### Phase 2: Graph Construction (1 day)
- [ ] Exhaustive pair enumeration
- [ ] Adaptive thresholding
- [ ] Early stopping heuristics

### Phase 3: Integration (1 day)
- [ ] Replace vocab tree in `sfm_pipeline.py`
- [ ] Keep LightGlue on verified pairs
- [ ] Output format compatible with COLMAP

### Phase 4: Global Optimization (Optional, 2-3 days)
- [ ] Cycle enumeration
- [ ] Polynomial composition
- [ ] Gröbner-based outlier detection

### Phase 5: Optimization (1-2 days)
- [ ] GPU batching for pre-filter
- [ ] Multiprocessing for verification
- [ ] Early stopping based on connectivity

**Total: 5-10 days** for full implementation

---

## Trade-offs

### Pros
✅ **Exhaustive**: Checks all pairs (no missed connections)
✅ **Robust**: Works with low-overlap + low-texture
✅ **White-box**: Fully mathematical, debuggable
✅ **Adaptive**: Automatically adjusts to dataset
✅ **Provable**: Algebraic certificates for every decision

### Cons
❌ **Slower**: ~2-3× slower than current (28s vs 11s for 100 images)
❌ **More GPU**: 2.5× more GPU time (25s vs 10s)
❌ **Complexity**: More sophisticated algorithm, harder to maintain
❌ **Scaling**: O(N²) pairs (current is O(N×K) where K=200)

### When to Use

**Use Algebraic Graph if**:
- Dataset has low-overlap regions (< 30% overlap)
- Low-texture environments (indoor, sky, uniform surfaces)
- Need explainable/debuggable pipeline
- Quality > Speed (willing to wait 3× longer for better results)

**Use Current Pipeline if**:
- Standard datasets with good overlap (> 50%)
- Speed is critical (real-time or near-real-time)
- Vocab tree performs well (tested on similar data)
- Don't need mathematical guarantees

---

## Optimization Potential

With optimizations (batching, multiprocessing, early stopping):

| N | Current | Algebraic (Naive) | Algebraic (Optimized) |
|---|---------|-------------------|-----------------------|
| 50 | 5s | 6s | **4s** ✅ |
| 100 | 11s | 28s | **12s** ≈ |
| 200 | 22s | 106s | **40s** |
| 500 | 55s | 662s | **180s** |

**Conclusion**: With engineering effort, can match current speed while gaining robustness.

---

## Next Steps

### Immediate (Prototyping)
1. Implement `AlgebraicPreFilter` with orientation compatibility
2. Benchmark on 100-image dataset
3. Compare pair selection quality (algebraic vs vocab tree)

### Medium-term (Integration)
1. Integrate into `sfm_pipeline.py` as alternative mode
2. Add `--matching-mode` flag: `vocab_tree` | `algebraic_graph`
3. Benchmark on low-overlap datasets (ETH3D, Tanks & Temples)

### Long-term (Publication)
1. Implement global optimization (cycle consistency)
2. Theoretical analysis of completeness guarantees
3. Write up as CVPR paper: "Algebraic Correspondence Graphs for Robust SfM"

---

## Code Skeleton

```python
# sfm/core/algebraic_graph.py

class AlgebraicPreFilter:
    """Fast orientation/spatial compatibility check."""

    def check_compatibility(self, feat1, feat2) -> Tuple[bool, float]:
        """
        Returns:
            (compatible, confidence_score)

        Time: O(0.1ms) per pair
        """
        pass

class MinimalVerifier:
    """3-point polynomial verification."""

    def verify_minimal(self, correspondences) -> VerificationResult:
        """
        Time: O(1ms) per pair
        """
        pass

class AlgebraicGraph:
    """Main graph construction and optimization."""

    def build_graph(self, all_features) -> Graph:
        """
        Stage 1: Pre-filter all pairs
        Stage 2: Minimal verification
        Stage 3: (Optional) Global consistency

        Returns verified pairs for downstream matching.
        """
        pass

    def global_optimization(self, graph) -> Graph:
        """
        Cycle consistency check and outlier removal.
        """
        pass
```

---

**Status**: Proposal stage
**Priority**: High (solves critical low-overlap/low-texture problem)
**Risk**: Medium (more complex, needs engineering for speed)
**Reward**: High (unified mathematical framework, provably robust)
