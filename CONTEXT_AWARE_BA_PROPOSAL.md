# Context-Aware Bundle Adjustment Pipeline

## Overview

This document proposes a novel Bundle Adjustment approach that replaces COLMAP's traditional BA with a **context-aware optimization** that leverages **global scene understanding** through scene graphs and confidence weighting.

### Key Innovation

**Traditional BA**: Each observation is optimized independently, treating all cameras and points equally.

**Context-Aware BA**: Observations are weighted based on global scene structure, automatically down-weighting unreliable cameras and uncertain points.

---

## Motivation

### Limitations of Traditional Bundle Adjustment

#### 1. Local Minima Problem
Traditional BA optimizes local reprojection errors independently, which can lead to globally inconsistent reconstructions even when each camera pair is locally consistent.

```
Example:
Camera 1-2: reprojection error = 0.5px ✓ (locally good)
Camera 2-3: reprojection error = 0.5px ✓ (locally good)
Camera 3:   180° rotation error ✗ (globally wrong!)
```

#### 2. Isolated Optimization
BA treats each observation independently, ignoring relationships between cameras:
- No use of covisibility information
- All cameras treated equally regardless of connection strength
- Cannot distinguish reliable vs unreliable views

#### 3. Outlier Sensitivity
```
Camera 1: 200 good points + 5 outliers
Camera 2: 200 good points + 100 outliers (textureless wall)

Traditional BA: All observations weighted equally
→ Camera 2's outliers distort the entire structure

Desired behavior: Automatically detect and down-weight Camera 2
```

---

## Proposed Solution Architecture

### Pipeline Overview

```
Input: Images
    ↓
[SuperPoint/ALIKED Feature Extraction]  ← Keep existing
    ↓
[LightGlue Feature Matching]            ← Keep existing
    ↓
[Geometric Verification (MAGSAC)]       ← Keep existing
    ↓
[Scene Graph Construction]              ← NEW!
    ↓
[Confidence Computation]                ← NEW!
    ↓
[Context-Aware Bundle Adjustment]       ← NEW! (replaces COLMAP)
    ↓
Output: Camera Poses + Sparse Points
```

### Key Stages

#### Stage 1: Scene Graph Construction

Build a graph representation of the reconstruction problem:

**Nodes:**
- Camera nodes: One per image
- Point nodes: One per 3D point (optional)

**Edges:**
- Camera-Camera: Weighted by covisibility (number of shared points)
- Camera-Point: Observations
- Point-Point: Spatial proximity (optional)

**Node Features:**
```python
Camera node feature:
  - Pooled SuperPoint descriptors (mean/max pooling)
  - Number of keypoints
  - Spatial distribution statistics
  - Match quality statistics

Point node feature (optional):
  - Number of observations
  - Reprojection error
  - Triangulation angle
```

#### Stage 2: Confidence Computation (Rule-Based)

**No training required!** Use hand-crafted heuristics to compute confidence scores.

##### Camera Confidence Factors:

1. **Covisibility** (25% weight)
   - How many other cameras share points with this camera?
   - High covisibility = well-connected = more reliable

2. **Match Quality** (20% weight)
   - Average matching score from LightGlue
   - High scores = distinctive features = reliable

3. **Feature Density** (15% weight)
   - Number of keypoints detected
   - More features = better coverage = reliable

4. **Spatial Uniformity** (15% weight)
   - How evenly are keypoints distributed?
   - Uniform distribution = good coverage = reliable

5. **Multi-hop Connectivity** (15% weight)
   - 2-hop neighbors (friends of friends)
   - High indirect connectivity = stable position in graph

6. **Geometric Consistency** (10% weight)
   - Inlier ratio from MAGSAC
   - High inliers = geometrically consistent = reliable

**Formula:**
```python
confidence(camera) =
    0.25 × covisibility_score +
    0.20 × match_quality +
    0.15 × feature_density +
    0.15 × uniformity +
    0.15 × two_hop_connectivity +
    0.10 × inlier_ratio
```

##### Point Confidence Factors:

1. **Track Length** (50% weight)
   - Number of cameras observing this point
   - More observations = more certain 3D position

2. **Reprojection Error** (30% weight)
   - Lower error = more accurate
   - `error_score = 1.0 / (1.0 + error)`

3. **Triangulation Angle** (20% weight)
   - Angle between observing cameras
   - Larger angles = better triangulation

#### Stage 3: Context-Aware Bundle Adjustment

**Objective Function:**
```
Traditional BA:
  minimize Σ ||π(P_i, X_j) - x_ij||²

Context-Aware BA:
  minimize Σ w_i × w_j × ||π(P_i, X_j) - x_ij||²
           i,j

where:
  w_i = confidence(camera_i)
  w_j = confidence(point_j)
  π() = projection function
```

**Effect:**
- High-confidence cameras/points: Large weight → stronger influence
- Low-confidence cameras/points: Small weight → less influence
- Outliers automatically down-weighted
- Reliable structure preserved

---

## Implementation Plan

### Phase 1: Core Components (Week 1)

#### 1.1 Scene Graph Builder
```python
# File: sfm/core/scene_graph.py

class SceneGraphBuilder:
    """Build graph from features and matches"""

    def build_graph(self, features, matches) -> Dict:
        """
        Returns:
          {
            'camera_nodes': {img_path: feature_vector},
            'camera_edges': [(img1, img2, weight), ...],
            'num_cameras': int,
            'num_edges': int
          }
        """
```

#### 1.2 Rule-Based Confidence
```python
# File: sfm/core/confidence.py

class RuleBasedConfidence:
    """Compute confidence without training"""

    def compute_camera_confidence(self, camera_id, features, matches, graph) -> float:
        """Return confidence ∈ [0, 1]"""

    def compute_point_confidence(self, point_id, points3d) -> float:
        """Return confidence ∈ [0, 1]"""
```

#### 1.3 Context-Aware BA Optimizer
```python
# File: sfm/core/context_aware_ba.py

class ContextAwareBundleAdjustment:
    """BA with confidence weighting"""

    def optimize(self, features, matches, initial_cameras,
                 initial_images, initial_points3d):
        """
        Returns:
          optimized_points3d, optimized_cameras, optimized_images
        """
```

### Phase 2: Integration (Week 2)

#### 2.1 Drop-in Replacement
```python
# File: sfm/core/context_aware_ba.py

def context_aware_reconstruction(features, matches, output_path, image_dir):
    """
    Drop-in replacement for colmap_binary_reconstruction()

    Compatible signature - no changes to sfm_pipeline.py needed!
    """
```

#### 2.2 Pipeline Flag
```python
# File: sfm_pipeline.py

parser.add_argument(
    "--use_context_ba",
    action="store_true",
    help="Use context-aware BA instead of COLMAP"
)

# Usage in pipeline
if args.use_context_ba:
    from sfm.core.context_aware_ba import context_aware_reconstruction
    sparse_points, cameras, images = context_aware_reconstruction(...)
else:
    from sfm.core.colmap_binary import colmap_binary_reconstruction
    sparse_points, cameras, images = colmap_binary_reconstruction(...)
```

### Phase 3: Testing & Validation (Week 3)

#### 3.1 Unit Tests
- Test confidence computation on synthetic data
- Verify graph construction
- Check BA convergence

#### 3.2 Integration Tests
- Compare with COLMAP on benchmark datasets
- Measure reprojection errors
- Measure pose accuracy

#### 3.3 Performance Tests
- Runtime comparison
- Memory usage
- Scalability (small vs large datasets)

---

## Usage Examples

### Basic Usage
```bash
# Use context-aware BA instead of COLMAP
python sfm_pipeline.py \
    --input_dir images/ \
    --output_dir output/ \
    --use_context_ba
```

### With Custom Feature Extractor
```bash
python sfm_pipeline.py \
    --input_dir images/ \
    --output_dir output/ \
    --feature_extractor aliked \
    --use_context_ba
```

### For 3D Gaussian Splatting
```bash
python sfm_pipeline.py \
    --input_dir images/ \
    --output_dir output/ \
    --use_context_ba \
    --copy_to_3dgs_dir ../gaussian-splatting/data/my_scene/
```

---

## Expected Performance Improvements

### Scenario 1: Partial Poor Quality
```
Dataset: 100 images, 20 are blurry/motion blur

Traditional COLMAP:
  - All cameras equally weighted
  - Poor cameras distort structure
  - Mean reprojection error: 1.2px

Context-Aware BA:
  - Poor cameras auto-detected (low feature density + uniformity)
  - Down-weighted → less influence
  - Mean reprojection error: 0.6px ✓
```

### Scenario 2: Sequential Video Capture
```
Dataset: 500 video frames with temporal order

Traditional COLMAP:
  - Independent optimization
  - No temporal consistency
  - Trajectory jitter

Context-Aware BA:
  - High covisibility along temporal chain
  - Sequential cameras mutually reinforce
  - Smooth trajectory ✓
```

### Scenario 3: Large-Scale Scenes
```
Dataset: 1000+ images, outdoor scene

Traditional COLMAP:
  - Global structure drift
  - Accumulating errors
  - Takes 2-3 hours

Context-Aware BA:
  - Global context prevents drift
  - Covisibility graph stabilizes structure
  - Takes 1-2 hours ✓
```

### Scenario 4: Textureless Regions
```
Dataset: Indoor with white walls

Traditional COLMAP:
  - False matches on walls
  - BA struggles with ambiguous features
  - Failed reconstruction or high error

Context-Aware BA:
  - Low confidence on wall cameras (low uniformity, weak matches)
  - Auto down-weighted
  - Successful reconstruction ✓
```

---

## Advantages Over Traditional BA

### 1. No Additional Feature Extraction
- ✓ Uses existing SuperPoint/ALIKED descriptors
- ✓ No DINO, no SAM, no extra models
- ✓ Minimal overhead (just graph construction)

### 2. No Training Required
- ✓ Rule-based confidence = zero training time
- ✓ Works out of the box
- ✓ No overfitting to specific datasets
- ✓ Interpretable (can debug each factor)

### 3. Automatic Outlier Handling
- ✓ Poor cameras automatically down-weighted
- ✓ Uncertain points given less importance
- ✓ No manual parameter tuning

### 4. Global Consistency
- ✓ Scene-level understanding prevents drift
- ✓ Covisibility graph enforces structural consistency
- ✓ Better for large-scale reconstructions

### 5. Lightweight
- ✓ No deep learning models
- ✓ Fast graph construction (O(n²) or O(n log n) with vocab tree)
- ✓ Standard scipy optimization (no GPU needed for BA)

---

## Future Enhancements (Optional)

### Enhancement 1: Hybrid Confidence (Optional)
Add a lightweight learned combiner for rule-based features:

```python
class HybridConfidence:
    """Rule-based features + 129-parameter MLP combiner"""

    def __init__(self):
        self.rule_computer = RuleBasedConfidence()

        # Tiny MLP (only 129 parameters!)
        self.combiner = nn.Sequential(
            nn.Linear(6, 16),   # 6 rule-based features
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def compute_confidence(self, camera_id, features, matches, graph):
        # Extract rule-based features
        feature_vec = self.rule_computer.compute_all_features(...)

        # Learned combination
        return self.combiner(feature_vec)
```

**Training:**
- Use COLMAP results as pseudo-GT
- Low error cameras → label = 1.0
- High error cameras → label = 0.0
- Train on 50-100 scenes (a few hours)

### Enhancement 2: GNN-based Context (Research)
Replace rule-based with learned GNN for complex pattern recognition:

```python
class GNNConfidence:
    """Full end-to-end learning (research-level)"""

    def __init__(self):
        self.gnn = GraphAttentionNetwork(
            input_dim=256,
            hidden_dim=128,
            output_dim=64,
            num_layers=3
        )
        self.confidence_head = nn.Linear(64, 1)
```

**Requirements:**
- Large-scale training (1000+ scenes)
- Ground truth poses
- GPU for training
- Several days of training time

---

## Technical Details

### Scene Graph Construction Time Complexity

```python
# Camera nodes: O(n) where n = num_images
for img in images:
    node_feature = pool_descriptors(img)  # O(k) where k = keypoints
    graph.add_node(img, node_feature)

# Camera-camera edges: O(m) where m = num_matches
for (img1, img2) in matches:
    graph.add_edge(img1, img2, weight=num_matches)

# Total: O(n × k + m)
# For typical case: n=100, k=4096, m=500
# → 100 × 4096 + 500 = ~410k operations (milliseconds)
```

### Confidence Computation Time Complexity

```python
# Per camera: O(degree × k)
# - degree: number of connected cameras (~10-50)
# - k: number of keypoints (~4096)

# Total: O(n × degree × k)
# For typical case: 100 × 20 × 4096 = ~8M operations (seconds)
```

### BA Optimization Time Complexity

```python
# Same as traditional BA: O(iterations × observations)
# But with confidence weighting (negligible overhead)

# Typical: 50 iterations × 10000 observations = 500k evaluations
# With context: +5% overhead for weight computation
```

---

## File Structure

```
sfm/
├── core/
│   ├── feature_extractor.py          # Existing
│   ├── feature_matcher.py            # Existing
│   ├── geometric_verification.py     # Existing
│   ├── colmap_binary.py              # Existing (fallback)
│   ├── scene_graph.py                # NEW: Graph construction
│   ├── confidence.py                 # NEW: Rule-based confidence
│   └── context_aware_ba.py           # NEW: Context-aware BA
├── utils/
│   └── ...                            # Existing utilities
└── ...

# New test files
tests/
├── test_scene_graph.py
├── test_confidence.py
└── test_context_ba.py
```

---

## Backward Compatibility

The new pipeline is **fully backward compatible**:

```python
# Old usage (still works)
python sfm_pipeline.py --input_dir images/ --output_dir output/
→ Uses COLMAP BA (default)

# New usage (opt-in)
python sfm_pipeline.py --input_dir images/ --output_dir output/ --use_context_ba
→ Uses Context-Aware BA

# Mixed usage
python sfm_pipeline.py --input_dir images/ --output_dir output1/
python sfm_pipeline.py --input_dir images/ --output_dir output2/ --use_context_ba
→ Compare results side-by-side
```

---

## Evaluation Metrics

### 1. Reconstruction Quality
- Mean reprojection error (pixels)
- Median reprojection error
- Percentage of points with error < 1px

### 2. Pose Accuracy (if GT available)
- Rotation error (degrees)
- Translation error (meters or %)
- ATE (Absolute Trajectory Error)

### 3. Completeness
- Number of registered images
- Number of 3D points
- Mean track length

### 4. Robustness
- Success rate on challenging datasets
- Handling of outliers (manual inspection)

### 5. Performance
- Total runtime (seconds)
- Peak memory usage (GB)
- Scalability (small vs large datasets)

---

## References & Inspiration

### Traditional Bundle Adjustment
- Triggs et al. "Bundle Adjustment — A Modern Synthesis" (2000)
- COLMAP: Structure-from-Motion Revisited (Schönberger & Frahm, 2016)

### Graph-Based Optimization
- Carlone et al. "Attention and Anticipation in Fast Visual-Inertial Navigation" (2019)
- Barsan et al. "Learning to Localize Through Compressed Binary Maps" (2018)

### Context-Aware Reconstruction
- Neural Correspondence Field (Li et al., 2021)
- NeRF--: Neural Radiance Fields Without Known Camera Parameters (Wang et al., 2021)
- BARF: Bundle-Adjusting Neural Radiance Fields (Lin et al., 2021)

### Scene Understanding
- Graph Attention Networks (Veličković et al., 2018)
- LoFTR: Detector-Free Local Feature Matching (Sun et al., 2021)

---

## Conclusion

This proposal presents a **practical, training-free** approach to improve Bundle Adjustment through global scene understanding. By leveraging the scene graph structure and rule-based confidence computation, we can automatically identify and down-weight unreliable observations, leading to more robust and accurate reconstructions.

**Key Takeaways:**
- ✓ Drop-in replacement for COLMAP BA
- ✓ No training required (rule-based)
- ✓ No additional feature extraction (uses existing SuperPoint/ALIKED)
- ✓ Automatic outlier handling
- ✓ Global consistency enforcement
- ✓ Lightweight and fast
- ✓ Fully backward compatible

**Next Steps:**
1. Implement rule-based confidence computation
2. Build scene graph constructor
3. Integrate with existing pipeline
4. Benchmark on standard datasets
5. (Optional) Add learned components for further improvement
