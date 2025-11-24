# SARA (Maximum-Parallax Augment) Overview

SARA is a pair-selection module designed to front-load geometry-aware reasoning _before_ expensive matching or SfM stages. It builds a sparse but information-rich graph by scoring candidate image pairs with a lightweight overlap–parallax proxy, then maximizes the total score with a maximum spanning tree and a small loop augmentation budget. The result is a plug-in `pairs.csv` that feeds matchers such as LightGlue and improves pose stability while cutting redundant matching calls.

## Core Motivation

Traditional SfM pipelines retrieve top‑K neighbours with pure appearance similarity (kNN/BoW). They defer geometric filtering until after feature matching, wasting compute on low-parallax or redundant edges and risking chain-like graphs that drift. SARA reframes the objective: we want to maximize expected triangulation gain per edge _before_ committing to a matcher.

### Triangulation Gain Proxy

For an image pair `(i, j)` we define

```
score_ij = overlap_ij^α × parallax_ij^β
```

* `overlap` ≈ inlier ratio estimate from mutual-NN descriptors after a mini-RANSAC.
* `parallax` ≈ median sine of the angle between bearing rays (intrinsics-aware when provided).
* `α`, `β` tune the relative emphasis.

This proxy is consistent with standard triangulation uncertainty: depth variance ∝ 1/(baseline × sin θ). Higher overlap raises confidence in the baseline observation; higher parallax increases effective baseline. Edges below thresholds `τ_overlap` or `τ_parallax` are discarded as they contribute little to reconstruction.

## Pipeline Stages

SARA runs entirely prior to heavy matching:

1. **Embeddings** – load or compute DINOv2 CLS embeddings (`timm`) for every image.
2. **Candidates** – cosine kNN (`knn_k`) over embeddings to form a symmetric candidate edge list.
3. **Fast Pre-Matching** – load ALIKED features (`features/{stem}.npz`), gather top mutual descriptor matches (`top_t_mutual`), and fit an 8-point fundamental matrix with a mini-RANSAC (`min_nn_for_ransac`, `ransac_iters`, `ransac_conf`).
4. **Overlap × Parallax Scoring** – estimate overlap from inlier ratio; compute parallax proxy via normalized bearing angles (`parallax.py`) with optional focal length hints (`fx`/`fy` when intrinsics are missing).
5. **Maximum Spanning Tree** – select `N−1` edges maximizing total score using `networkx.maximum_spanning_tree`.
6. **Leaf Augmentation** – connect degree-1 nodes with their best remaining edges, respecting an optional degree cap (`deg_cap`).
7. **Triangle Loop Augment** – add up to `ceil(loop_budget_per_node × N)` extra edges, prioritizing high-score edges that close loops longer than length-1 (preference for large baselines).
8. **Persist Outputs** – write `pairs.csv` (`i, j, score, overlap, parallax`) and `pairs_for_matcher.jsonl` for matcher adapters.

The exported graph typically contains `(N−1) + O(N)` edges, yielding order-of-magnitude fewer match invocations coSARAred with fixed-k retrieval, while retaining high-parallax constraints for stability.

## Implementation Notes

| Module | Responsibility |
| --- | --- |
| `SARA/config.py` | Dataclass with hyperparameters, cache resolution helpers. |
| `SARA/dino_embed.py` | DINOv2 embedding loader/cacher. |
| `SARA/aliked_io.py` | Reads ALIKED `.npz` caches, verifies coverage. |
| `SARA/candidates.py` | Cosine kNN candidate generation. |
| `SARA/mini_ransac.py` | Mutual nearest neighbour selection + OpenCV mini-RANSAC. |
| `SARA/parallax.py` | Bearing vector normalization and parallax proxy. |
| `SARA/scoring.py` | Overlap ratio + edge scoring logic. |
| `SARA/mst.py` | `networkx` maximum spanning tree wrapper. |
| `SARA/augment.py` | Leaf reinforcement and triangle loop augmentation. |
| `SARA/cli.py` | Orchestrates the full flow, produces pair files. |
| `SARA/io_utils.py` | Utility I/O helpers (image listing, CSV/JSONL saving). |

Dependencies introduced by SARA: `timm`, `networkx`, `pandas`, and `opencv-python` (already used elsewhere). All computations proceed on CPU by default; DINO can run on GPU when available.

## Key Hyperparameters

* `knn_k` – candidate neighbours per image (default 30).
* `top_t_mutual` – mutual descriptor cap for mini-RANSAC (default 128).
* `min_nn_for_ransac` – minimum matches to attempt RANSAC (default 32).
* `tau_overlap`, `tau_parallax` – pruning thresholds for overlap/parallax.
* `loop_budget_per_node` – additional loops per node (default 0.5).
* `deg_cap` – optional maximum degree during leaf augmentation (default 6).

These map to CLI flags in `sfm_pipeline.py` (`--SARA_knn_k`, `--SARA_top_t_mutual`, etc.).

## Outputs

* `out_dir/pairs.csv` – canonical pair list with scores used to drive matching.
* `out_dir/pairs_for_matcher.jsonl` – convenience jsonl for LightGlue/other matchers expecting JSON records.
* Optional caches in `out_dir/embeddings` and `out_dir/features` with DINO and ALIKED data for reuse.

Downstream tooling consumes `pairs.csv` to constrain matching (e.g., LightGlue uses `predefined_pairs`). COLMAP integration remains unchanged beyond the reduced match workload and improved graph quality.

## Practical Tips

* **Missing intrinsics** – set `--SARA_disable_intrinsics` if K is unreliable; the proxy falls back to normalized image coordinates.
* **Low-parallax scenes** – lower `tau_parallax` and consider relaxing `tau_overlap` to avoid empty MSTs.
* **Diagnostics** – inspect `pairs.csv` summary statistics (edge count, mean parallax). Sparse graphs with very low parallax usually indicate RANSAC thresholds are too strict or features are insufficient.
* **Scaling** – DINO embedding caching is the dominant upfront cost; reuse `out_dir/embeddings` between runs to avoid recomputation.
* **Integration** – `sfm_pipeline.py` automatically exports ALIKED caches to `out/SARA/features` and runs `SARA.cli.run_SARA` when the pipeline executes Stage 3.

SARA’s simplicity—one score, MST, and a handful of loop edges—keeps runtime low while providing strong theoretical backing for improved triangulation fidelity. Use it whenever you want balanced coverage with minimal matching overhead. 
