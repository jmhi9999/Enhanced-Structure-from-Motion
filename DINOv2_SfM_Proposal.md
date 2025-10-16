# DINOv2-based Global Structure-from-Motion (SfM)

## Overview

We propose a **DINOv2-driven, context-aware SfM pipeline** that replaces traditional keypoint-based feature extraction (e.g., SIFT, SuperPoint) with **Transformer-derived patch embeddings and attention priors** from DINOv2. The system achieves robust scene reconstruction even under weak overlap, textureless regions, or large viewpoint changes, while maintaining full geometric interpretability.

---

## Motivation

Traditional SfM pipelines rely on sparse keypoint extraction and descriptor matching. However, CNN-based extractors fail under illumination and texture variance, while ViT-based global descriptors have not been effectively integrated into full geometric pipelines.

### Problems in Traditional SfM

1. **Weak Overlap Dependency** — Matching fails if <15 shared points between views.
2. **Scale Drift** — Independent incremental reconstructions drift across scales.
3. **Global Context Ignorance** — Each observation treated independently in BA.
4. **Feature Fragility** — Local CNN features fail on repetitive or low-texture surfaces.

---

## DINOv2-SfM: Architecture Overview

```text
[Image Frames]
     ↓
[DINOv2 Feature Extraction]
     ├─ CLS Token → Global Descriptor (Scene-level)
     └─ Patch Tokens → Local Descriptors (Feature Grid)
     ↓
[Attention-guided Patch Sampling]
     ↓
[CLS-based Top-K Retrieval (FAISS)]
     ↓
[LoFTR Matching + MAGSAC Verification]
     ↓
[Scene Graph Construction (Weighted by CLS Similarity)]
     ↓
[Loop Closure Detection (CLS Token Retrieval)]
     ↓
[Component Merging (Sim(3) Alignment)]
     ↓
[Pose Graph Optimization]
     ↓
[Graph-aware Context-Aware Bundle Adjustment (DINO-weighted)]
```

---

## 1. DINOv2 Feature Extraction

### Key Idea

Use **patch embeddings** as dense local features and **attention weights** as per-feature confidence priors.

```python
from sfm.features.dinov2_feature_extractor import DINOv2Extractor
dino = DINOv2Extractor(model_name='dinov2_vits14')
feat = dino.extract(image)
# feat['cls'], feat['patch'], feat['attn'], feat['grid_hw']
```

Each patch token acts as a **keypoint-free descriptor**, allowing uniform coverage of the image. The attention map provides confidence used later in weighting Bundle Adjustment residuals.

---

## 2. Attention-Guided Patch Matching

### Mutual NN Matching

```python
from sfm.matching.dino_matcher import match_dino_patches
matches = match_dino_patches(feat_i, feat_j, cos_thresh=0.8, topk_i=800, topk_j=800)
```

- Matching score = cosine similarity of normalized patch embeddings.
- Mutual nearest filtering ensures geometric consistency.
- Attention-weighted sampling keeps high-saliency tokens only.

## 2c. Image Retrieval and Pair Selection (CLS-based Top-K)

Since LoFTR is detector-free, we replace the traditional **vocabulary tree** approach with a **DINO CLS token–based retrieval pipeline**. Each image’s CLS embedding is used as a compact global descriptor for Top‑K candidate selection.

**Pipeline Overview**

```text
[Images]
  ↓
[DINOv2 CLS Embeddings]
  ↓
[FAISS Index + Reciprocal Filtering]
  ↓
[Top-K1 Candidates (e.g., 60)]
  ↓
[Re-ranking: DINO Patch / LoFTR quick-check]
  ↓
[Final Top-K (e.g., 20) for Full LoFTR Matching]
```

**Steps**

1. **CLS Embedding Extraction**

   - Extract L2-normalized CLS embeddings from DINOv2 for each image.
   - (Optional) Apply PCA to 256D + whitening for speed/memory efficiency.

2. **Initial Retrieval**

   - Build a FAISS index (e.g., `IndexFlatIP` or `IVF-PQ`).
   - Query each CLS vector to get Top‑K1=60 candidates.
   - Apply reciprocal filtering and temporal diversity constraints to reduce redundancy.
   - Final Top‑K ≈ 20.

3. **Re-ranking (Optional)**

   - **A)** Patch‑level mutual NN check (Top‑400 patches, avg cosine of top‑50 mutuals).
   - **B)** Low‑res LoFTR (e.g., 448 px) + quick MAGSAC verification for inlier count.

4. **Full Matching**
   - Run LoFTR on the Top‑K pairs (e.g., 640–840 px) followed by MAGSAC++.
   - Accept pairs with ≥ 15 inliers and triangulation angle ≥ 2°.

**Advantages**

- Compatible with detector‑free LoFTR.
- Robust under viewpoint and illumination changes.
- CLS similarity correlates well with scene overlap, enabling efficient pair curation.

---

## 3. Scene Graph Construction

Each image is a node. Edges represent verified DINO matches.

Edge weight:
\[ w*{ij} = N*{inlier} \times \cos(\text{CLS}\_i, \text{CLS}\_j) \]

This fuses local geometry with global semantics (scene-level proximity).

---

## 4. Component Merging via Sim(3)

Disconnected subgraphs are aligned using **Sim(3)** estimated from camera centers of verified inter-component links:

\[
T\_{s→t} = (s, R, t) \text{ where } C_t \approx s R C_s + t
\]

Implementation uses Umeyama alignment with scale normalization.

---

## 5. Graph-Aware Context-Aware Bundle Adjustment

We redefine BA as a **graph-weighted robust M-estimator**:

\[
E(Θ) = \sum*{i,j} w*{ij}(G, \text{DINO}) \cdot ρ(\|r\_{ij}(Θ)\|^2)
\]

### Weight Design

\[
S*{ij} = 0.5\,S^{\text{loc}}*{ij} + 0.5\,S^{\text{glob}}_{ij},\quad
w_{ij} = 0.35\,C*i + 0.25\,A*{ij} + 0.25\,S*{ij} + 0.15\,T*{ij}
\]

| Symbol                    | Meaning                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| \(C_i\)                   | Camera-level confidence (covisibility centrality, inlier ratio, track length) |
| \(A\_{ij}\)               | Attention-derived confidence of the two patches (avg. DINO attention at obs)  |
| \(S^{\text{loc}}\_{ij}\)  | Local match quality (LightGlue/LoFTR confidence, normalized)                  |
| \(S^{\text{glob}}\_{ij}\) | Global similarity (CLS cosine between images)                                 |
| \(T\_{ij}\)               | Triangulation angle quality                                                   |

**Implementation note.** We normalize \(S^{\text{loc}}\_{ij}\) by the matcher’s native confidence percentile per scene (e.g., 10–90% → [0,1]). CLS cosine is clamped to [0,1]. Attention values are min–max normalized over each image.

The IRLS solver reweights residuals iteratively, down-weighting ambiguous or low-texture observations.

---

## 6. Loop Closure & Scene Merging

Global CLS descriptors naturally support long-range loop closure detection:

1. Retrieve top-K nearest CLS embeddings via FAISS (same index used for pair selection).
2. Retrieve top-K nearest CLS embeddings via FAISS (same index used for pair selection).
3. Verify via LightGlue or DINO patch re-matching.
4. Add inter-component Sim(3) edges.
5. Re-run PGO + BA.

This enables **semantic loop closure** without requiring overlapping 3D points.

---

## 7. Implementation Highlights

- **Pair Retrieval:** DINOv2 CLS embedding–based FAISS Top‑K selection (reciprocal, diverse)
- **Local Matching:** LoFTR (default, detector-free); LightGlue optional fallback; MAGSAC++ verification
- **Global Similarity:** DINOv2 CLS token cosine for retrieval & loop closure
- **DINO Patches:** Attention‑guided Top‑K sampling (e.g., 800) for optional dense-ish assists
- **Graph Backend:** Weighted pose graph (PGO + IRLS), Sim(3) component alignment
- **Context-Aware BA:** Combined local score + CLS + attention + tri‑angle weighting
- **Compatibility:** Drop‑in with traditional SfM; local modules are swappable

---

## 8. Experiments

### Datasets

- ETH3D, ScanNet, MegaDepth, Tanks & Temples

### Metrics

- Registered Images ↑
- Split Reconstructions ↓
- Global Scale Consistency ↑
- ATE / RPE ↓
- Reprojection Error ↓
- Runtime / Memory Cost
- Pair Retrieval Precision / Recall
- Inlier Rate after Retrieval

### Baselines

| Method                | Descriptor / Matcher          | Loop Closure | Scene Merge | Context-Aware BA |
| --------------------- | ----------------------------- | ------------ | ----------- | ---------------- |
| COLMAP                | SIFT + NN                     | ✗            | ✗           | ✗                |
| HLOC                  | NetVLAD + SuperPoint/NN       | ✓            | partial     | ✗                |
| LightGlue             | SuperPoint/ALIKED + LightGlue | ✓            | partial     | ✗                |
| **LoFTR (Ours base)** | LoFTR (detector-free)         | ✓            | ✓           | ✓                |
| VGGSfM                | ViT Global                    | ✓            | partial     | ✗                |
| **Ours (DINOv2‑SfM)** | DINOv2 (CLS+patch) + LoFTR    | ✓            | ✓           | ✓                |

---

## 9. Key Results (Expected)

| Metric            | COLMAP | HLOC | Ours (DINOv2-SfM) |
| ----------------- | ------ | ---- | ----------------- |
| Registered Images | 68%    | 81%  | **95%**           |
| Mean ATE ↓        | 0.39   | 0.31 | **0.21**          |
| Mean RPE ↓        | 0.25   | 0.22 | **0.15**          |
| Split Count ↓     | 3.1    | 1.7  | **1.0**           |
| Runtime (rel.)    | 1.0x   | 1.6x | 2.2x              |

_All reported numbers for **Ours** use LoFTR (detector-free) as primary matcher with MAGSAC++ verification; DINOv2 provides retrieval and weighting._

---

## 10. Novelty Summary (CVPR Justification)

| Innovation                             | Description                                                                  | Why It Matters                                            |
| -------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Attention-guided Keypoint-Free SfM** | Replaces discrete keypoints with DINO patch embeddings weighted by attention | Enables continuous dense coverage & confidence estimation |
| **Graph-aware M-Estimator BA**         | Incorporates global + local priors into weighting                            | Improves convergence, robustness, and merging stability   |
| **Semantic Loop Closure**              | Uses CLS token retrieval instead of shared 3D points                         | Solves traditional merging failure cases                  |
| **Unified Global Context**             | One ViT backbone for global and local cues                                   | Simplifies architecture, reduces model fragmentation      |

---

## 11. Future Work

- Hybrid DINO + SuperPoint fusion for lightweight deployment.
- Self-supervised fine-tuning on video sequences.
- End-to-end differentiable SfM training with DINO frozen backbone.

---

## 12. Conclusion

> **DINOv2-SfM** unifies global semantic reasoning and geometric consistency in a single Transformer-based architecture, achieving robust, scalable, and context-aware 3D reconstruction.

**Core Contributions:**

1. Keypoint-free feature extraction using DINOv2 patch embeddings.
2. Attention-guided confidence weighting in BA.
3. Graph-aware optimization for stable scene merging.

This pipeline provides both a _conceptual_ and _practical_ leap toward unified Transformer-based visual reconstruction.

Additionally, by replacing the traditional vocabulary-tree retrieval with **DINO CLS–based FAISS Top‑K selection**, the pipeline achieves detector-free scalability and robust pair selection compatible with LoFTR.
