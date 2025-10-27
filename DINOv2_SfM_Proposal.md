# DINOv3-based Global Structure-from-Motion (SfM)

## Overview

We propose a **DINOv3-driven, context-aware SfM pipeline** that replaces traditional keypoint-based feature extraction (e.g., SIFT, SuperPoint) with **Transformer-derived patch embeddings and attention priors** from DINOv3. The system achieves robust scene reconstruction even under weak overlap, textureless regions, or large viewpoint changes, while maintaining full geometric interpretability.

---

## Motivation

Traditional SfM pipelines rely on sparse keypoint extraction and descriptor matching. However, CNN-based extractors fail under illumination and texture variance, while ViT-based global descriptors have not been effectively integrated into full geometric pipelines.

### Problems in Traditional SfM

1. **Weak Overlap Dependency** — Matching fails if <15 shared points between views.
2. **Scale Drift** — Independent incremental reconstructions drift across scales.
3. **Global Context Ignorance** — Each observation treated independently in BA.
4. **Feature Fragility** — Local CNN features fail on repetitive or low-texture surfaces.

---

## DINOv3-SfM: Architecture Overview

```text
[Image Frames]
     ↓
[DINOv3 Feature Extraction]
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

## 1. DINOv3 Feature Extraction

### Key Idea

Use **patch embeddings** as dense local features and **attention weights** as per-feature confidence priors.

```python
from sfm.features.dino_feature_extractor import DINOFeatureExtractor
dino = DINOFeatureExtractor(model_family='dinov3', model_name='dinov3_vitl14')
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
- Patch correspondences are not fed directly to triangulation; instead they provide matchability priors, inlier proposals, and fallback correspondences when LoFTR fails locally.

### Integration with Detector-Free Matching

**LoFTR is the primary matcher** for generating metric correspondences. DINO patch matches influence the pipeline as follows:

1. **Pre-filtering.** Images with low average DINO cosine score skip heavy LoFTR processing, conserving runtime on non-overlapping pairs.
2. **Guided LoFTR.** High-attention DINO patches define spatial masks that bias LoFTR's coarse stage, improving robustness in textureless regions.
3. **Pose-only Estimation (Fallback).** When LoFTR produces <15 inliers, DINO patch matches (verified with MAGSAC++) are used to estimate relative pose and add edges to the pose graph, but **NOT for direct 3D triangulation** due to spatial coarseness (±7px uncertainty from 14×14 patches). These edges enable Sim(3) merging without contributing BA residuals.
4. **Residual Weighting.** For successful LoFTR matches, DINO attention scores at corresponding patch locations become `A_{ij}` terms inside bundle adjustment weights (Section 5).

## 2c. Image Retrieval and Pair Selection (CLS-based Top-K)

Since LoFTR is detector-free, we replace the traditional **vocabulary tree** approach with a **DINO CLS token–based retrieval pipeline**. Each image’s CLS embedding is used as a compact global descriptor for Top‑K candidate selection.

**Pipeline Overview**

```text
[Images]
  ↓
[DINOv3 CLS Embeddings]
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

   - Extract L2-normalized CLS embeddings from DINOv3 for each image.
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

## 2d. Global Graph-Based Reconstruction

We adopt a **global SfM** approach (specifically GLOMAP) rather than incremental registration. This aligns naturally with our graph-aware design and addresses Problem 2 (scale drift).

### Why Global SfM?

**GLOMAP (Global Structure-from-Motion)** optimizes all camera poses simultaneously on a view graph, ensuring:

1. **Global scale consistency** — No incremental drift; all scales resolved jointly
2. **Natural component merging** — Disconnected subgraphs aligned via Sim(3) in a unified framework
3. **Explicit graph structure** — View graph serves as backbone for DINO weighting

### Pipeline Integration

```text
[LoFTR Matches + MAGSAC++]
  ↓
[View Graph Construction]
  - Nodes: Cameras
  - Edges: Relative poses (E matrix decomposition)
  - Weights: N_inlier × cos(CLS_i, CLS_j)  ← DINO enhancement!
  ↓
[Rotation Averaging (graph-based)]
  - Minimize: Σ_edges w_ij · || R_j - R_i * R_ij ||²
  - DINO weights prioritize high-confidence edges
  ↓
[Translation Averaging (graph-based)]
  - Solve for global camera positions under scale constraints
  ↓
[Triangulation using global poses]
  ↓
[Context-Aware Bundle Adjustment (Section 5)]
```

### Advantages over Incremental SfM

| | Incremental (COLMAP) | Global (GLOMAP + Ours) |
|---|---|---|
| **Scale consistency** | ⚠️ Drift accumulates | ✅ Globally consistent |
| **Component merging** | ⚠️ Difficult | ✅ Native Sim(3) support |
| **Graph structure** | ❌ Implicit | ✅ Explicit view graph |
| **DINO integration** | Limited | ✅ Natural edge weighting |

**Implementation note:** We use GLOMAP as the reconstruction backend, modified to accept our DINO-enhanced edge weights during rotation/translation averaging.

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
S_{ij} = 0.5\,S^{\text{loc}}_{ij} + 0.5\,S^{\text{glob}}_{ij},\quad
C_{ij} = 0.5\,(C_i + C_j),\quad
w_{ij} = 0.35\,C_{ij} + 0.25\,A_{ij} + 0.25\,S_{ij} + 0.15\,T_{ij}
\]

| Symbol                    | Meaning                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| \(C_i\)                   | Camera-level confidence (covisibility centrality, inlier ratio, track length) |
| \(A\_{ij}\)               | Attention-derived confidence of the two patches (avg. DINO attention at obs)  |
| \(S^{\text{loc}}\_{ij}\)  | Local match quality (LightGlue/LoFTR confidence, normalized)                  |
| \(S^{\text{glob}}\_{ij}\) | Global similarity (CLS cosine between images)                                 |
| \(T\_{ij}\)               | Triangulation angle quality                                                   |

**Implementation notes.**

- \(S^{\text{loc}}\_{ij}\) is normalized by the matcher’s native confidence percentile per scene (e.g., 10–90% → [0,1]), while \(S^{\text{glob}}\_{ij}\) uses a CLS cosine that is first whitened, then clamped to \([0, 0.9]\) to avoid overweighting perceptually similar yet distinct structures.
- \(C_i\) aggregates per-camera metrics (track count, reprojection residual percentile, pose graph degree). We smooth \(C_{ij}\) with an exponential moving average across optimization iterations to suppress jitter.
- Attention values are min–max normalized per image and capped by the local entropy of the attention map so repetitive textures do not dominate.
- Residuals tied to pairs with large baseline but low semantic affinity preserve their weight via a floor \(w_{ij} \geq 0.05\) to protect valid but dissimilar observations.

The IRLS solver reweights residuals iteratively, down-weighting ambiguous or low-texture observations.

---

## 6. Loop Closure & Scene Merging

Global CLS descriptors naturally support long-range loop closure detection:

1. Retrieve top-K nearest CLS embeddings via FAISS (same index used for pair selection).
2. Verify via LoFTR or DINO patch re-matching.
3. Add inter-component Sim(3) edges to the view graph.
4. Re-run rotation/translation averaging + BA.

This enables **semantic loop closure** without requiring overlapping 3D points.

---

## 7. Implementation Highlights

- **Reconstruction Backend:** GLOMAP (global SfM) with DINO-enhanced edge weights
- **Pair Retrieval:** DINOv3 CLS embedding–based FAISS Top‑K selection (reciprocal, diverse)
- **Primary Matcher:** LoFTR (detector-free, dense) + MAGSAC++ verification
- **Ablation Matcher:** LightGlue (sparse, for speed/quality trade-off comparison)
- **Global Similarity:** DINOv3 CLS token cosine for retrieval & loop closure
- **DINO Patches:** Attention‑guided Top‑K sampling (e.g., 800) for pre-filtering and pose-graph fallback
- **Graph Operations:** Rotation/translation averaging with DINO-weighted edges, Sim(3) component alignment
- **Context-Aware BA:** PyTorch-based custom BA with combined local score + CLS + attention + tri‑angle weighting
- **Modularity:** DINO weighting layer is matcher-agnostic and can be applied to any feature-based pipeline

---

## 8. Experiments

### Datasets

- **ETH3D (train/test split).** Use the official training split for hyper-parameter tuning, hold out the 7 evaluation scenes for reporting.
- **ScanNet (scene subset).** 100 indoor scenes for retrieval analysis, 25 disjoint scenes for final metrics; depth + poses serve as ground truth.
- **MegaDepth (outdoor).** 150 validation image collections, filtered for ≥20 views; calibration derived from COLMAP bundle.
- **Tanks & Temples.** Intermediate set for cross-dataset generalization; camera intrinsics supplied by dataset.

All image sets are undistorted and resized so the longer edge ≤ 1600 px unless otherwise noted. Intrinsics are either provided or estimated once with COLMAP and reused across methods for fairness.

### Metrics

- Registered Images ↑
- Split Reconstructions ↓
- Global Scale Consistency ↑
- ATE / RPE ↓
- Reprojection Error ↓
- Runtime / Memory Cost
- Pair Retrieval Precision / Recall
- Inlier Rate after Retrieval
- LoFTR failure cases rescued by DINO patches (count)

### Experimental Setup

- **Hardware.** 1× RTX 4080 16 GB GPU + 1× 16-core CPU (Ryzen 9 7950X equivalent), 64 GB RAM; CLS extraction batched to fit 4080 memory budget.
- **Software.** PyTorch 2.1 + FAISS GPU; GLOMAP for reconstruction backend; COLMAP 3.9 for baselines only; custom PyTorch BA implementation.
- **Hyper-parameters.** CLS Top-K1 = 60, final Top-K = 20, LoFTR resolution {448, 832} px; MAGSAC++ thresholds tuned on ETH3D train.

### Protocol

1. **Retrieval Ablation.** Compare NetVLAD, DINO CLS, and hybrid (CLS+NetVLAD concat) on pair-retrieval precision/recall.
2. **Matcher Ablation.** Compare LoFTR (dense, primary) vs. LightGlue (sparse) to demonstrate:
   - DINO weighting benefits are larger with dense matchers (LoFTR)
   - Framework is matcher-agnostic but shows synergy with detector-free approaches
   - Speed/quality trade-offs in textureless regions
3. **DINO Integration Ablation.** Evaluate on LoFTR: (a) vanilla LoFTR, (b) +attention-guided masks, (c) +DINO fallback, (d) full pipeline.
4. **Weighting Ablation.** Run BA with (a) uniform weights, (b) only geometric terms (C_ij + T_ij), (c) full DINO-aware weighting (all terms in Eq. Section 5).
5. **Loop Closure Stress Test.** Inject long-range loops (ScanNet hallway, MegaDepth skyline) and measure Sim(3) drift before/after closure.
6. **Runtime Study.** Report per-stage runtime, GPU memory for each dataset; include breakdown of CLS extraction, FAISS queries, LoFTR, GLOMAP, BA.

### Baselines

| Method                | Descriptor / Matcher          | SfM Type    | Loop Closure | Scene Merge | Context-Aware BA |
| --------------------- | ----------------------------- | ----------- | ------------ | ----------- | ---------------- |
| COLMAP                | SIFT + NN                     | Incremental | ✗            | ✗           | ✗                |
| HLOC                  | NetVLAD + SuperPoint/NN       | Incremental | ✓            | partial     | ✗                |
| GLOMAP                | NetVLAD + SuperPoint          | Global      | ✓            | ✓           | ✗                |
| LightGlue             | SuperPoint/ALIKED + LightGlue | Incremental | ✓            | partial     | ✗                |
| VGGSfM                | ViT Global                    | Global      | ✓            | partial     | ✗                |
| **Ours (DINOv3‑SfM)** | DINOv3 (CLS+patch) + LoFTR    | Global      | ✓            | ✓           | ✓                |

### Timeline / Deliverables

- **Month 1.** Reproduce COLMAP, HLOC, GLOMAP baselines; implement DINO CLS retrieval and validate on ETH3D train scenes.
- **Month 2.** Integrate GLOMAP with DINO-weighted edges; implement custom PyTorch BA with attention weighting; run full pipeline on ETH3D/ScanNet validation.
- **Month 3.** Execute cross-dataset evaluation (MegaDepth, Tanks & Temples); run all ablations; finalize runtime study and qualitative visualizations.

---

## 9. Key Results (Expected)

| Metric            | COLMAP | HLOC | GLOMAP | Ours (DINOv3-SfM) |
| ----------------- | ------ | ---- | ------ | ----------------- |
| Registered Images | 68%    | 81%  | 89%    | **93%**           |
| Mean ATE ↓        | 0.39   | 0.31 | 0.26   | **0.19**          |
| Mean RPE ↓        | 0.25   | 0.22 | 0.18   | **0.14**          |
| Split Count ↓     | 3.1    | 1.7  | 1.2    | **1.0**           |
| Runtime (rel.)    | 1.0x   | 1.6x | 1.4x   | 2.1x              |

_Numbers for **Ours** use LoFTR (detector-free) as primary matcher with MAGSAC++ verification on GLOMAP backend; DINOv3 provides CLS retrieval, edge weighting, and BA attention weights._

**Matcher Comparison (Ours only):**

| Matcher   | Registered | ATE ↓ | Textureless ATE ↓ | Runtime (rel.) |
| --------- | ---------- | ----- | ----------------- | -------------- |
| LightGlue | 90%        | 0.22  | 0.35              | 1.5x           |
| LoFTR     | **93%**    | **0.19** | **0.24**       | 2.1x           |

_Shows DINO weighting benefits are larger with dense matchers in challenging regions._

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

> **DINOv3-SfM** unifies global semantic reasoning and geometric consistency in a single Transformer-based architecture, achieving robust, scalable, and context-aware 3D reconstruction.

**Core Contributions:**

1. **Keypoint-free dense matching** using DINOv3 patch embeddings + LoFTR, replacing traditional sparse keypoint extraction.
2. **Attention-guided confidence weighting** in bundle adjustment, leveraging DINO's learned semantic priors.
3. **Graph-aware global optimization** via GLOMAP with DINO-enhanced edge weights, ensuring scale consistency and robust component merging.

This pipeline provides both a _conceptual_ and _practical_ leap toward unified Transformer-based visual reconstruction. By building on GLOMAP's global SfM framework and enhancing it with DINOv3's multi-scale semantic understanding, we address fundamental limitations of incremental approaches (scale drift, fragmented reconstructions) while maintaining full geometric interpretability.

The system is **matcher-agnostic** in principle, but achieves strongest performance with dense detector-free matchers (LoFTR) that complement DINO's continuous patch representation. The DINO CLS–based FAISS retrieval replaces traditional vocabulary trees, enabling efficient pair selection at scale.
