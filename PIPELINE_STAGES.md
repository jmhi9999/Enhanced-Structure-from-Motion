# Complete Pipeline: Feature Types & Usage

## Stage-by-Stage Feature Usage

### Stage 1: Feature Extraction
```
Input: Images
Output: Local features (keypoints + descriptors)

Model: SuperPoint or ALIKED
Feature Type: LOCAL features
  - Keypoints: [N, 2] (x, y coordinates)
  - Descriptors: [256, N] (local patch descriptors)
  - Scores: [N] (confidence scores)

Purpose: Find corresponding points between images
```

**DINOv2 NOT used here!**

---

### Stage 2: Feature Matching
```
Input: Local features from Stage 1
Output: Correspondences between image pairs

Model: LightGlue
Uses: Local descriptors [256, N]

Purpose: Match keypoints between images
```

**DINOv2 NOT used here!**

---

### Stage 3: Geometric Verification
```
Input: Matched keypoints
Output: Inlier matches (geometrically consistent)

Algorithm: MAGSAC
Uses: Keypoint coordinates only

Purpose: Remove outlier matches
```

**DINOv2 NOT used here!**

---

### Stage 4: Scene Graph Construction
```
Input: Verified matches
Output: Graph structure (cameras as nodes)

Uses: Covisibility (number of shared points)
      Match quality (from LightGlue)

Purpose: Understand scene topology
```

**DINOv2 NOT used here!**

---

### Stage 5: Loop Closure Detection ← DINOv2 Used HERE!
```
Input: Images (or SuperPoint features)
Output: Additional image pairs (temporal loops)

Model: DINOv2 (optional) OR SuperPoint pooling (default)
Feature Type: GLOBAL descriptors
  - One descriptor per image: [384] (DINOv2) or [256] (SuperPoint pooling)

Purpose: Find similar images that are temporally far apart
         (e.g., Image 1 ↔ Image 100 in loop capture)

Algorithm:
  1. Extract global descriptor for each image
  2. Compute cosine similarity between all pairs
  3. If similarity > threshold AND temporal_gap > 30:
       → Loop closure candidate
  4. Verify with LightGlue feature matching
  5. Add verified loops to scene graph
```

#### Two Options for Global Descriptors

**Option A: SuperPoint Pooling (Default)**
```python
# Reuse existing SuperPoint features
global_desc = np.max(features['descriptors'], axis=1)  # [256]
global_desc = global_desc / np.linalg.norm(global_desc)

# Pros: No additional model, fast
# Cons: Lower quality than DINOv2
```

**Option B: DINOv2 (Optional)**
```python
# New forward pass with DINOv2
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
global_desc = model(image_tensor)  # [384]

# Pros: Higher quality, better loop detection
# Cons: Requires GPU, additional time
```

---

### Stage 6: Component Analysis
```
Input: Scene graph (with loop closures added)
Output: Connected components

Algorithm: BFS/DFS
Uses: Graph connectivity only

Purpose: Find disconnected parts of reconstruction
```

**DINOv2 NOT used here!**

---

### Stage 7: Pose Graph Optimization
```
Input: Relative poses (essential matrices)
Output: Initial camera poses

Algorithm: Motion averaging
Uses: Rotation/translation from essential matrices

Purpose: Initialize camera poses for each component
```

**DINOv2 NOT used here!**

---

### Stage 8: Component Merging
```
Input: Initial poses + loop closures
Output: Merged poses in global coordinate system

Algorithm: Sim(3) alignment (Umeyama)
Uses: Camera centers from loop closure pairs

Purpose: Merge disconnected components
```

**DINOv2 NOT used here!**

---

### Stage 9: Context-Aware Bundle Adjustment
```
Input: Merged camera poses + 3D points
Output: Refined reconstruction

Algorithm: Weighted BA with confidence
Uses: All features (covisibility, match quality, etc.)

Purpose: Final optimization
```

**DINOv2 NOT used here!**

---

## Summary: When is DINOv2 Used?

| Stage | Model | Feature Type | DINOv2? |
|-------|-------|--------------|---------|
| 1. Feature Extraction | SuperPoint/ALIKED | Local | ❌ No |
| 2. Feature Matching | LightGlue | Local | ❌ No |
| 3. Geometric Verification | MAGSAC | Local | ❌ No |
| 4. Scene Graph | - | - | ❌ No |
| **5. Loop Closure** | **DINOv2 (optional)** | **Global** | **✅ Yes (optional)** |
| 6. Component Analysis | - | - | ❌ No |
| 7. PGO | - | - | ❌ No |
| 8. Component Merging | - | - | ❌ No |
| 9. Context-Aware BA | - | - | ❌ No |

---

## Key Distinction: Local vs Global Features

### Local Features (SuperPoint/ALIKED)
```
Purpose: Find corresponding POINTS between images
Input: Image patch (e.g., 32×32 pixels)
Output: Descriptor for that specific patch [256]

Example:
  Image A: Corner of window → descriptor [256]
  Image B: Same corner → similar descriptor [256]
  → Match!

Used for: Feature matching, triangulation, BA
```

### Global Features (DINOv2)
```
Purpose: Find similar IMAGES (image retrieval)
Input: Entire image (e.g., 1920×1080)
Output: Single descriptor for whole image [384]

Example:
  Image 1: Building front view → descriptor [384]
  Image 100: Same building front view → similar descriptor [384]
  → Loop closure!

Used for: Loop detection, image retrieval, place recognition
```

---

## Recommended Configuration

### For Most Cases: SuperPoint Pooling
```python
loop_detector = LoopClosureDetector(descriptor_type='superpoint_pooling')

# Pros:
# ✅ No additional model
# ✅ Fast (reuse existing features)
# ✅ Works well for sequential video

# Use when:
# - Sequential capture (video, drone)
# - Limited compute (CPU-only)
# - Already have good scene graph connectivity
```

### For Challenging Cases: DINOv2
```python
loop_detector = LoopClosureDetector(descriptor_type='dinov2')

# Pros:
# ✅ Superior global representation
# ✅ Better for large temporal gaps
# ✅ Robust to appearance changes

# Use when:
# - Unordered images (photo collections)
# - Large-scale scenes (>500 images)
# - Severe viewpoint/lighting changes
# - GPU available
```

---

## Example Workflow

```python
# Stage 1-4: Standard SfM pipeline
features = extract_features(images)          # SuperPoint
matches = match_features(features)           # LightGlue
verified = geometric_verification(matches)   # MAGSAC
scene_graph = build_graph(verified)

# Stage 5: Loop closure detection
# Option A: SuperPoint pooling (fast, no additional model)
loop_detector = LoopClosureDetector('superpoint_pooling')
global_desc = loop_detector.extract_global_descriptors(
    images=None,      # Not needed
    features=features # Reuse SuperPoint features!
)

# Option B: DINOv2 (slower, better quality)
loop_detector = LoopClosureDetector('dinov2')
global_desc = loop_detector.extract_global_descriptors(
    images=images,    # Need original images
    features=None     # Not used
)

# Find loops (same code for both options)
loops = loop_detector.detect_loop_closures(scene_graph, global_desc)
verified_loops = loop_detector.verify_loop_closures(loops, features, matcher)

# Add to scene graph
add_loop_closures_to_scene_graph(scene_graph, verified_loops, features, matches)

# Stage 6-9: Continue with rest of pipeline
components = detect_components(scene_graph)
initial_poses = pgo_initialize(features, matches, scene_graph)
merged_poses = merge_components(components, initial_poses, verified_loops)
final_result = context_aware_ba(merged_poses, ...)
```

---

## Time Complexity Comparison

### SuperPoint Pooling
```
Feature extraction: 100ms/image (SuperPoint - already done)
Global descriptor:  0.1ms/image (just max pooling)
Similarity search:  O(N²) comparisons (fast, CPU)

Total overhead: ~negligible (already have features)
```

### DINOv2
```
Feature extraction: 100ms/image (SuperPoint - already done)
Global descriptor:  50ms/image (DINOv2 forward pass, GPU)
Similarity search:  O(N²) comparisons (fast, CPU)

Total overhead: 50ms × N images
  100 images → +5 seconds
  500 images → +25 seconds
```

---

## Conclusion

**DINOv2는 feature 추출 단계가 아니라, loop closure detection 단계에서만 선택적으로 사용됩니다.**

- **Feature extraction**: SuperPoint/ALIKED (unchanged)
- **Feature matching**: LightGlue (unchanged)
- **Loop detection**: SuperPoint pooling (default) OR DINOv2 (optional)

**Default recommendation**: SuperPoint pooling으로 시작하고, loop detection 품질이 부족하면 DINOv2로 업그레이드.