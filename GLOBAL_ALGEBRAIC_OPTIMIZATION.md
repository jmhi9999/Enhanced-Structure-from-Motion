# Stage 4: Global Algebraic Optimization

**목표**: Pairwise verification을 통과한 edge들 중에서도 **globally inconsistent**한 것을 찾아서 제거.

---

## 문제 정의

### Pairwise vs Global Consistency

**Pairwise consistency**: 각 edge (i,j)가 개별적으로 algebraic consensus 통과
```
Image A ↔ Image B: ✓ (inlier ratio 0.8)
Image B ↔ Image C: ✓ (inlier ratio 0.7)
Image C ↔ Image A: ✓ (inlier ratio 0.75)
```

하지만 이것만으로는 **global structure가 consistent**한지 알 수 없음!

**Global inconsistency 예시**:
```
T_AB: A를 B로 변환 (90° 회전 + translation)
T_BC: B를 C로 변환 (90° 회전 + translation)
T_CA: C를 A로 변환 (90° 회전 + translation)

Composition: T_AB ∘ T_BC ∘ T_CA = 270° rotation
Expected: Identity (0° rotation)

→ INCONSISTENT! (At least one edge is wrong)
```

---

## 수학적 원리

### Cycle Consistency Constraint

**정의**: Graph에서 cycle C = (v₁ → v₂ → ... → vₖ → v₁)에 대해:
```
T_{v₁,v₂} ∘ T_{v₂,v₃} ∘ ... ∘ T_{vₖ,v₁} = I (Identity)
```

Affine transformation T는 matrix 형태:
```
T = [A | t]  where A ∈ ℝ²ˣ², t ∈ ℝ²
    [0 | 1]

Composition: T₁ ∘ T₂ = [A₁A₂ | A₁t₂ + t₁]
                        [0    | 1        ]
```

**Cycle constraint를 polynomial로**:
```
For cycle of length 3: T_AB ∘ T_BC ∘ T_CA = I

A_AB · A_BC · A_CA = I₂      (rotation part)
A_AB · A_BC · t_CA + A_AB · t_BC + t_AB = 0  (translation part)
```

이것들은 **polynomial equations** in the entries of A and t!

### Ideal Membership Test (Gröbner Basis)

**핵심 아이디어**: 모든 cycle constraints를 모아서 polynomial ideal I를 만든다.

```
I = ⟨g₁, g₂, ..., gₘ⟩

where each gᵢ is a polynomial from a cycle constraint
```

**Theorem**: System is **globally consistent** ⟺ **1 ∉ I**

즉, Gröbner basis를 계산해서 1이 포함되면 → **모순** → 적어도 하나의 edge가 틀렸다!

---

## 알고리즘 1: Simple Cycle Consistency (빠름)

### Step 1: Find All Cycles

```python
def find_cycles(graph, max_length=4):
    """
    Find all simple cycles up to max_length.

    For N=100 images:
    - 3-cycles: ~100K
    - 4-cycles: ~1M

    We focus on 3-cycles (triangles) for speed.
    """
    cycles = []

    # For each node
    for v1 in graph.nodes():
        # For each pair of neighbors
        for v2 in graph.neighbors(v1):
            for v3 in graph.neighbors(v2):
                if v3 in graph.neighbors(v1) and v3 != v1:
                    # Found triangle: v1 → v2 → v3 → v1
                    cycles.append((v1, v2, v3))

    return cycles
```

**Complexity**:
- Triangles: O(N · d²) where d = average degree
- For d=20, N=100: ~40K triangles

### Step 2: Compute Cycle Residual

```python
def cycle_residual(T_ab, T_bc, T_ca):
    """
    Compute how far the cycle deviates from identity.

    Args:
        T_ab, T_bc, T_ca: 3x3 affine matrices

    Returns:
        residual: float (0 = perfect consistency)
    """
    # Compose transformations
    T_composed = T_ab @ T_bc @ T_ca

    # Should equal identity
    I = np.eye(3)

    # Frobenius norm of difference
    residual = np.linalg.norm(T_composed - I, 'fro')

    return residual
```

**Mathematical detail**:
```
T_composed = [A_comp | t_comp]
             [0      | 1      ]

Residual = ||A_comp - I₂||_F + ||t_comp||₂

For perfect consistency: residual = 0
In practice: residual < ε (threshold, e.g., 0.01)
```

### Step 3: Detect Inconsistent Cycles

```python
def detect_inconsistent_cycles(graph, threshold=0.01):
    """
    Find cycles with residual > threshold.
    """
    cycles = find_cycles(graph, max_length=3)

    inconsistent = []

    for (v1, v2, v3) in cycles:
        T_12 = graph.get_transformation(v1, v2)
        T_23 = graph.get_transformation(v2, v3)
        T_31 = graph.get_transformation(v3, v1)

        residual = cycle_residual(T_12, T_23, T_31)

        if residual > threshold:
            inconsistent.append({
                'cycle': (v1, v2, v3),
                'residual': residual,
                'edges': [(v1, v2), (v2, v3), (v3, v1)]
            })

    return inconsistent
```

### Step 4: Identify Bad Edges

**문제**: Cycle이 inconsistent하면, 어느 edge가 잘못됐는지 어떻게 알까?

**Solution 1: Voting (빠름)**
```python
def find_bad_edges_voting(inconsistent_cycles):
    """
    Count how many inconsistent cycles each edge appears in.
    """
    edge_fault_count = defaultdict(int)

    for cycle_info in inconsistent_cycles:
        for edge in cycle_info['edges']:
            edge_fault_count[edge] += 1

    # Edges appearing in many inconsistent cycles are likely bad
    bad_edges = sorted(
        edge_fault_count.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return bad_edges
```

**Idea**: 만약 edge (A,B)가 100개 cycle 중 80개에서 inconsistent하면, 이 edge가 문제일 확률이 높음.

**Solution 2: Algebraic Certificate (정확함)**
```python
def find_bad_edges_algebraic(graph, inconsistent_cycles):
    """
    For each edge, compute its 'algebraic inconsistency score'.
    """
    edge_scores = {}

    for edge in graph.edges():
        # Remove this edge temporarily
        graph_minus_edge = graph.copy()
        graph_minus_edge.remove_edge(*edge)

        # Re-check inconsistent cycles
        remaining_inconsistent = detect_inconsistent_cycles(graph_minus_edge)

        # Score: how many cycles become consistent when this edge is removed?
        score = len(inconsistent_cycles) - len(remaining_inconsistent)
        edge_scores[edge] = score

    # Higher score = removing this edge fixes more cycles
    return sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
```

**Problem**: 이건 너무 느림 (각 edge마다 전체 cycle 재검사)

**Solution 2b: Incremental Algebraic (타협)**
```python
def algebraic_score_per_edge(graph, cycle):
    """
    For each edge in the cycle, compute its contribution to residual.

    Mathematical idea:
    - Perturb each transformation slightly
    - Measure sensitivity of cycle residual
    - High sensitivity = this edge is critical
    """
    v1, v2, v3 = cycle
    edges = [(v1, v2), (v2, v3), (v3, v1)]

    T_12 = graph.get_transformation(v1, v2)
    T_23 = graph.get_transformation(v2, v3)
    T_31 = graph.get_transformation(v3, v1)

    base_residual = cycle_residual(T_12, T_23, T_31)

    sensitivities = {}
    epsilon = 0.01

    for edge_idx, edge in enumerate(edges):
        # Perturb this transformation
        if edge_idx == 0:
            T_12_pert = perturb_transformation(T_12, epsilon)
            pert_residual = cycle_residual(T_12_pert, T_23, T_31)
        elif edge_idx == 1:
            T_23_pert = perturb_transformation(T_23, epsilon)
            pert_residual = cycle_residual(T_12, T_23_pert, T_31)
        else:
            T_31_pert = perturb_transformation(T_31, epsilon)
            pert_residual = cycle_residual(T_12, T_23, T_31_pert)

        # Sensitivity: how much residual changes
        sensitivities[edge] = abs(pert_residual - base_residual) / epsilon

    return sensitivities

def perturb_transformation(T, epsilon):
    """Add small random noise to transformation."""
    noise = np.random.randn(3, 3) * epsilon
    noise[2, :] = 0  # Keep last row as [0, 0, 1]
    noise[:, 2] *= 2  # Translation can vary more
    return T + noise
```

**Interpretation**:
- High sensitivity → small change in this edge causes big change in residual
- → This edge is **tightly constrained** by the cycle
- → If cycle is inconsistent, this edge is likely **wrong**

---

## 알고리즘 2: Gröbner Basis (수학적으로 완벽)

### Step 1: Construct Polynomial System

```python
from sympy import symbols, groebner, Matrix

def build_cycle_polynomial_system(cycles, transformations):
    """
    Build polynomial system from cycle constraints.

    Returns:
        polynomials: List of SymPy polynomials
        variables: List of transformation parameters
    """
    # Variables: all transformation parameters
    # For N edges, 6N variables (each affine has 6 DOF)

    variables = []
    for edge in transformations:
        # a11, a12, a21, a22, t1, t2 for each edge
        vars_edge = symbols(f'a11_{edge} a12_{edge} a21_{edge} a22_{edge} t1_{edge} t2_{edge}')
        variables.extend(vars_edge)

    polynomials = []

    # For each cycle, add composition constraint
    for cycle in cycles:
        v1, v2, v3 = cycle

        # Get variable names for each edge
        a11_12, a12_12, a21_12, a22_12, t1_12, t2_12 = get_vars_for_edge((v1, v2))
        a11_23, a12_23, a21_23, a22_23, t1_23, t2_23 = get_vars_for_edge((v2, v3))
        a11_31, a12_31, a21_31, a22_31, t1_31, t2_31 = get_vars_for_edge((v3, v1))

        # Matrix for T_12
        A_12 = Matrix([[a11_12, a12_12], [a21_12, a22_12]])
        t_12 = Matrix([t1_12, t2_12])

        # Similarly for T_23, T_31
        # ...

        # Composition: T_12 ∘ T_23 ∘ T_31
        A_composed = A_12 * A_23 * A_31
        t_composed = A_12 * A_23 * t_31 + A_12 * t_23 + t_12

        # Constraint: A_composed = I, t_composed = 0
        polynomials.append(A_composed[0,0] - 1)  # a11 = 1
        polynomials.append(A_composed[0,1] - 0)  # a12 = 0
        polynomials.append(A_composed[1,0] - 0)  # a21 = 0
        polynomials.append(A_composed[1,1] - 1)  # a22 = 1
        polynomials.append(t_composed[0])        # t1 = 0
        polynomials.append(t_composed[1])        # t2 = 0

    return polynomials, variables
```

### Step 2: Compute Gröbner Basis

```python
def check_consistency_groebner(polynomials, variables):
    """
    Compute Gröbner basis and check if 1 ∈ I.

    Returns:
        consistent: bool
        basis: Gröbner basis (for debugging)
    """
    # Add known constraints (from pairwise verification)
    # For example: T_12 should be close to the verified transformation

    # Compute Gröbner basis
    basis = groebner(polynomials, variables, order='lex')

    # Check if 1 is in the basis (system is inconsistent)
    if 1 in basis:
        return False, basis
    else:
        return True, basis
```

**Problem**: Gröbner basis는 **매우 느림**!
- N=100 images, E=500 edges → 3000 variables
- 40K cycles → 240K polynomials
- Gröbner basis: **수 시간** 소요

### Step 3: Localize Inconsistency

```python
def find_inconsistent_edge_groebner(graph, cycles):
    """
    Use Gröbner basis to find which edge is inconsistent.

    Strategy: Binary search
    """
    # Start with all edges
    edge_set = set(graph.edges())

    # Binary search: which subset is inconsistent?
    def is_subset_consistent(edges):
        subgraph = graph.edge_subgraph(edges)
        cycles_sub = find_cycles(subgraph)
        polys, vars = build_cycle_polynomial_system(cycles_sub, edges)
        consistent, _ = check_consistency_groebner(polys, vars)
        return consistent

    # If full graph is consistent, done
    if is_subset_consistent(edge_set):
        return None

    # Binary search for minimal inconsistent set
    while len(edge_set) > 1:
        # Split in half
        half = len(edge_set) // 2
        edges_list = list(edge_set)
        left = set(edges_list[:half])
        right = set(edges_list[half:])

        if not is_subset_consistent(left):
            edge_set = left
        elif not is_subset_consistent(right):
            edge_set = right
        else:
            # Both halves consistent, but together inconsistent
            # → Need both halves, problem is in their interaction
            # Pick edge with lowest confidence from original verification
            return min(edge_set, key=lambda e: graph[e[0]][e[1]]['confidence'])

    return list(edge_set)[0]
```

**This is too slow for practical use!**

---

## 알고리즘 3: Practical Hybrid (추천) ⭐

**Idea**: Combine fast heuristics with selective algebraic verification

```python
class GlobalAlgebraicOptimizer:
    """
    Hybrid approach:
    1. Fast cycle residual check (all cycles)
    2. Voting-based edge ranking
    3. Greedy removal with re-verification
    """

    def optimize(self, graph, max_iterations=10):
        """
        Iteratively remove bad edges until globally consistent.
        """
        for iteration in range(max_iterations):
            # Step 1: Find all inconsistent cycles
            inconsistent = self.detect_inconsistent_cycles(graph)

            if len(inconsistent) == 0:
                print(f"✓ Globally consistent after {iteration} iterations")
                break

            print(f"Iteration {iteration}: {len(inconsistent)} inconsistent cycles")

            # Step 2: Rank edges by fault score
            edge_scores = self.compute_edge_fault_scores(graph, inconsistent)

            # Step 3: Remove worst edge
            worst_edge = max(edge_scores, key=edge_scores.get)
            print(f"  Removing edge {worst_edge} (score: {edge_scores[worst_edge]:.2f})")

            graph.remove_edge(*worst_edge)

            # Step 4: Check if graph is still connected
            if not self.is_well_connected(graph):
                print("  ⚠ Graph becoming disconnected, stopping")
                break

        return graph

    def compute_edge_fault_scores(self, graph, inconsistent_cycles):
        """
        Combine multiple signals:
        1. Vote count (how many bad cycles)
        2. Sensitivity (analytical derivative)
        3. Original verification confidence (lower = more suspect)
        """
        edge_scores = defaultdict(float)

        for cycle_info in inconsistent_cycles:
            cycle = cycle_info['cycle']
            residual = cycle_info['residual']

            # Compute sensitivity for each edge in cycle
            sensitivities = self.algebraic_score_per_edge(graph, cycle)

            for edge, sensitivity in sensitivities.items():
                # Vote contribution
                vote = 1.0

                # Sensitivity contribution (normalized)
                sens_normalized = sensitivity / (sum(sensitivities.values()) + 1e-8)

                # Confidence penalty (lower confidence = higher score)
                confidence = graph[edge[0]][edge[1]].get('confidence', 0.5)
                conf_penalty = 1 - confidence

                # Combined score
                edge_scores[edge] += (
                    vote * 0.4 +
                    sens_normalized * 0.4 +
                    conf_penalty * 0.2
                ) * residual  # Weight by how bad the cycle is

        return edge_scores

    def is_well_connected(self, graph, min_connectivity=0.8):
        """
        Check if graph is well-connected.

        Criterion: Largest connected component has >= 80% of nodes
        """
        import networkx as nx
        components = list(nx.connected_components(graph.to_undirected()))
        largest_component_size = max(len(c) for c in components)
        connectivity = largest_component_size / graph.number_of_nodes()
        return connectivity >= min_connectivity
```

### Example Usage

```python
# After Stage 3: We have a graph with verified edges
graph = build_graph_from_verified_pairs(verified_pairs)

# Stage 4: Global optimization
optimizer = GlobalAlgebraicOptimizer(threshold=0.01)
optimized_graph = optimizer.optimize(graph, max_iterations=10)

# Result: Graph with globally consistent edges
print(f"Removed {len(graph.edges()) - len(optimized_graph.edges())} inconsistent edges")
```

---

## 시간 복잡도

### Algorithm 1: Simple Voting
- Find cycles: O(N·d²) ≈ 40K cycles for N=100, d=20
- Compute residuals: O(cycles) ≈ 40K × (0.01ms) = **0.4s**
- Voting: O(cycles × 3) = **0.5s**
- **Total: ~1s**

### Algorithm 2: Gröbner Basis
- Build polynomial system: O(cycles × vars) ≈ 40K × 3K = 120M operations
- Gröbner basis: **Exponential** (수 시간)
- **Total: Impractical**

### Algorithm 3: Hybrid
- Cycle detection: **0.5s**
- Sensitivity analysis: 40K cycles × 3 edges × 0.1ms = **12s**
- Greedy removal: 10 iterations × (0.5s + 12s) = **125s**

**With optimization**:
- Parallelize sensitivity: 12s → **2s** (6 cores)
- Early stopping: 3-5 iterations average → **15s**
- **Practical: ~2s** per optimization round

---

## 실제 예시

### Before Global Optimization
```
Graph: 100 images, 500 edges
Inconsistent cycles: 127 / 40000 (0.3%)

Bad edge candidates:
  (img_042, img_051): fault_score = 15.2 (in 38 bad cycles)
  (img_078, img_103): fault_score = 12.7 (in 31 bad cycles)
  (img_015, img_029): fault_score = 8.1 (in 19 bad cycles)
```

### After Iteration 1: Remove (img_042, img_051)
```
Graph: 100 images, 499 edges
Inconsistent cycles: 89 / 39850 (0.22%)  ← Reduced by 30%!
```

### After Iteration 3: Remove 3 edges
```
Graph: 100 images, 497 edges
Inconsistent cycles: 0 / 39700 (0%)  ✓ Globally consistent!
```

---

## 핵심 수학

**Theorem (Cycle Consistency)**:
```
A set of transformations {T_ij} is globally consistent
⟺ For all cycles C, ∏_{(i,j)∈C} T_ij = I
```

**Corollary (Outlier Detection)**:
```
If a cycle has residual > ε, at least one edge in the cycle is incorrect.

The edge with highest sensitivity is most likely the outlier.
```

**Proof sketch**:
- If all edges were correct, composition would equal identity (residual = 0)
- Non-zero residual → at least one edge is wrong
- Sensitivity measures ∂(residual)/∂T → which edge has most impact
- Highest impact edge is most likely source of error

---

## 요약

### 추천 방식: Algorithm 3 (Hybrid)

```
1. Fast cycle enumeration (triangles only): 0.5s
2. Residual computation (vectorized): 0.5s
3. Sensitivity analysis (parallelized): 2s
4. Greedy edge removal (3-5 iterations): 3×3s = 9s

Total: ~12s for 100 images
```

### 언제 유용한가?

✅ **Large-scale reconstruction**: 많은 이미지 → 많은 cycle → inconsistency 가능성 높음
✅ **Low-quality matches**: Pairwise verification이 약간 부정확해도 global에서 수정 가능
✅ **Loop closure**: Long sequence에서 마지막 이미지가 처음과 연결될 때 drift 보정

### Trade-off

- **Pros**: Mathematically principled, provably finds inconsistencies
- **Cons**: Slower (~2-12s overhead), complex implementation
- **Alternative**: Skip Stage 4, let COLMAP's bundle adjustment handle it
  - COLMAP는 이미 global optimization 함 (photometric reprojection)
  - 하지만 COLMAP은 **numerical** (gradient descent)
  - Stage 4는 **algebraic** (provable outlier detection)

**결론**: Stage 4는 선택적으로 사용. 품질이 critical하면 켜고, 속도가 중요하면 끄기.
