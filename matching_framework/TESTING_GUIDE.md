# Dataset Testing Guide

This guide explains how to use the benchmarking system to evaluate feature matching methods on standard datasets.

## Overview

The testing framework provides:

- **Dataset loaders** for ETH3D, ScanNet, and MegaDepth
- **Evaluation metrics** for match quality and pose estimation
- **Automated benchmarking** with comprehensive logging
- **Standardized output** for easy comparison

## Quick Start

```bash
# Basic benchmark on ETH3D
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path /path/to/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --output results/eth3d_superpoint_lightglue

# Compare different configurations
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path /path/to/eth3d \
    --extractor sift \
    --matcher nn \
    --output results/eth3d_sift_nn
```

## Supported Datasets

### ETH3D

**Type:** Indoor and outdoor scenes with high-quality reconstruction

**Download:** https://www.eth3d.net/datasets

**Expected structure:**
```
eth3d/
├── scene1/
│   ├── images/
│   │   ├── image0000.jpg
│   │   └── ...
│   ├── calibration.txt      # Contains: fx fy cx cy
│   └── poses.txt            # Contains: image_name \n 4x4_matrix \n ...
└── scene2/
    └── ...
```

**Calibration format (`calibration.txt`):**
```
fx fy cx cy
```

**Poses format (`poses.txt`):**
```
image0000
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
image0001
r11 r12 r13 tx
...
```

### ScanNet

**Type:** Indoor RGB-D scans

**Download:** http://www.scan-net.org/

**Expected structure:**
```
scannet/
├── scene0000_00/
│   ├── color/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── depth/
│   │   ├── 0.png           # 16-bit PNG, depth in millimeters
│   │   ├── 1.png
│   │   └── ...
│   ├── pose/
│   │   ├── 0.txt           # 4x4 camera-to-world matrix
│   │   ├── 1.txt
│   │   └── ...
│   └── intrinsic.txt       # 4x4 intrinsic matrix
└── scene0001_00/
    └── ...
```

**Intrinsic format (`intrinsic.txt`):**
```
fx  0  cx  0
0  fy  cy  0
0   0   1  0
0   0   0  1
```

**Pose format (`0.txt`, etc.):**
```
r11 r12 r13 tx
r21 r22 r23 ty
r31 r32 r33 tz
0   0   0   1
```

### MegaDepth

**Type:** Large-scale outdoor landmarks

**Download:** https://www.cs.cornell.edu/projects/megadepth/

**Expected structure:**
```
megadepth/
├── scene_info/
│   ├── scene0000.npz       # Contains: K_{image_name}, T_{image_name}
│   ├── scene0001.npz
│   └── ...
├── pairs.json              # List of {"scene_id": "...", "image0": "...", "image1": "..."}
└── scene0000/
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

**Scene info format (`scene0000.npz`):**
```python
{
    "K_image1.jpg": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
    "T_image1.jpg": np.array([[4x4 camera pose]]),
    # ... for each image
}
```

**Pairs format (`pairs.json`):**
```json
[
    {
        "scene_id": "scene0000",
        "image0": "image1.jpg",
        "image1": "image2.jpg",
        "overlap": 0.75
    },
    ...
]
```

### COLMAP Scene (local sparse reconstructions)

**Type:** Any scene that already has a COLMAP sparse model (no dense data required)

**Expected structure:**
```
scene_root/
├── images_2/               # RGB frames (auto-detected if named differently)
│   ├── DSCF5565.JPG
│   ├── DSCF5566.JPG
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

**Usage:**
```bash
python -m matching_framework.testing.benchmark \
    --dataset colmap_scene \
    --dataset_path ImageInputs/bonsai \
    --extractor superpoint \
    --matcher lightglue \
    --output results/bonsai_sp_lg \
    --colmap_min_shared_points 1000
```

Use `--colmap_min_shared_points` to require a minimum number of shared 3D tracks per pair.

To sweep all extractor/pair-selector combinations (matching automatically
selected as LightGlue for learning-based extractors, NN for classical ones):

```bash
python -m matching_framework.testing.run_benchmark_suite \
    --images-root ImageInputs \
    --output-root results/benchmark_suite \
    --colmap-min-shared-points 1000
```

Each run stores `summary.json` and `summary.csv` under
`results/benchmark_suite/<scene>/<extractor>_<matcher>_<pair_selector>/`.

## Evaluation Metrics

### Match Quality Metrics

1. **Number of Matches**: Total feature correspondences found
2. **Epipolar Error**:
   - Mean and median point-to-epipolar-line distance (pixels)
   - Computed using ground truth fundamental matrix
3. **Inlier Ratio**:
   - Fraction of matches within epipolar threshold
   - Default threshold: 1.0 pixels
4. **Depth Reprojection Error** (if depth available):
   - 3D distance after reprojecting using depth maps
   - Only for ScanNet dataset

### Pose Estimation Metrics

1. **Success Rate**: Fraction of pairs with successful pose recovery
2. **Rotation Error**: Angular difference between estimated and ground truth rotation (degrees)
3. **Translation Error**: Angular difference between translation directions (degrees)
4. **AUC@θ**: Area Under Curve at threshold θ
   - Fraction of samples with both rotation and translation errors < θ
   - Reported for θ = 5°, 10°, 20°

## Command-Line Options

### Dataset Options
```bash
--dataset {eth3d,scannet,megadepth}   # Dataset name
--dataset_path PATH                    # Root directory
--split {train,val,test}               # Dataset split (default: test)
--num_samples N                        # Limit to N samples (for quick tests)
```

### Feature Extraction
```bash
--extractor {sift,orb,superpoint,aliked,disk}
--max_keypoints N                      # Max keypoints per image (default: 4096)
--resize_max N                         # Resize max dimension (default: None)
```

### Matching
```bash
--matcher {nn,lightglue}
--distance_threshold FLOAT             # Lowe's ratio for NN (default: 0.7)
```

### Evaluation Thresholds
```bash
--epipolar_threshold FLOAT             # Epipolar error threshold in pixels (default: 1.0)
--ransac_threshold FLOAT               # RANSAC threshold in pixels (default: 1.0)

# MAGSAC Filtering (same as sfm_pipeline for realistic evaluation)
--use_magsac_filtering                 # Apply MAGSAC filtering (matches sfm_pipeline behavior)
--magsac_threshold FLOAT               # MAGSAC reprojection threshold in pixels (default: 2.0)
--magsac_confidence FLOAT              # MAGSAC confidence level (default: 0.999)
--magsac_max_iters INT                 # MAGSAC maximum iterations (default: 1000)
```

### Output
```bash
--output PATH                          # Output directory
--device {cuda,cpu}                    # Device (default: cuda)
```

## MAGSAC Filtering Option

**NEW:** The benchmark now supports MAGSAC filtering to match the behavior of `sfm_pipeline.py`.

### Why Use MAGSAC Filtering?

- **Raw evaluation (default):** Measures pure matching quality without geometric verification
- **MAGSAC evaluation (`--use_magsac_filtering`):** Simulates real SfM pipeline performance

### Comparison: Raw vs MAGSAC Evaluation

```bash
# 1. Raw matching evaluation (default)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --output results/eth3d_raw

# 2. Realistic SfM evaluation (with MAGSAC)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --use_magsac_filtering \
    --output results/eth3d_magsac
```

**Expected differences:**
- MAGSAC filtering typically retains 60-90% of matches
- Epipolar error will be lower (outliers removed)
- Pose estimation success rate may be higher (better initialization)

### MAGSAC Parameter Tuning

```bash
# Strict filtering (same as sfm_pipeline default)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor sift \
    --matcher nn \
    --use_magsac_filtering \
    --magsac_threshold 2.0 \
    --magsac_confidence 0.999 \
    --magsac_max_iters 1000 \
    --output results/eth3d_strict

# Very strict filtering (low noise tolerance)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor sift \
    --matcher nn \
    --use_magsac_filtering \
    --magsac_threshold 0.8 \
    --magsac_confidence 0.9999 \
    --magsac_max_iters 5000 \
    --output results/eth3d_very_strict
```

## Example Workflows

### 1. Compare Extractors on ETH3D

```bash
# SuperPoint + LightGlue (raw evaluation)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --output results/eth3d_superpoint_lg

# SIFT + NN (raw evaluation)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor sift \
    --matcher nn \
    --output results/eth3d_sift_nn

# ORB + NN (with MAGSAC - realistic SfM)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor orb \
    --matcher nn \
    --use_magsac_filtering \
    --output results/eth3d_orb_nn_magsac
```

### 2. Quick Test with Limited Samples

```bash
python -m matching_framework.testing.benchmark \
    --dataset scannet \
    --dataset_path data/scannet \
    --extractor superpoint \
    --matcher lightglue \
    --num_samples 50 \
    --output results/scannet_test
```

### 3. CPU-Only Evaluation

```bash
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor sift \
    --matcher nn \
    --device cpu \
    --output results/eth3d_cpu
```

## Understanding Output

### Console Output

```
==============================================================
BENCHMARK RESULTS
==============================================================
Dataset: eth3d
Extractor: superpoint
Matcher: lightglue
Samples: 150
--------------------------------------------------------------
Match Metrics:
  Avg Matches: 1247.3
  Avg Epipolar Error: 0.42 px
  Median Epipolar Error: 0.28 px
  Inlier Ratio: 94.23%
--------------------------------------------------------------
Pose Metrics:
  Success Rate: 97.33%
  Avg Rotation Error: 1.23°
  Avg Translation Error: 2.15°
  Median Rotation Error: 0.87°
  Median Translation Error: 1.45°
  AUC@5°: 85.33%
  AUC@10°: 92.67%
  AUC@20°: 96.00%
==============================================================
```

### Output Files

**`results.json`**: Per-sample detailed results
```json
[
  {
    "sample_idx": 0,
    "image0": "image0000.jpg",
    "image1": "image0001.jpg",
    "metadata": {"scene": "scene1"},
    "match_metrics": {
      "num_matches": 1523,
      "epipolar_error": 0.35,
      "median_epipolar_error": 0.21,
      "inlier_ratio": 0.96
    },
    "pose_metrics": {
      "pose_estimated": true,
      "rotation_error": 0.87,
      "translation_error": 1.42,
      "num_inliers": 1461
    }
  },
  ...
]
```

**`summary.json`**: Aggregated metrics
```json
{
  "dataset": "eth3d",
  "dataset_path": "/path/to/eth3d",
  "num_samples": 150,
  "extractor": "superpoint",
  "matcher": "lightglue",
  "config": {
    "max_keypoints": 4096,
    "distance_threshold": 0.7,
    "epipolar_threshold": 1.0,
    "ransac_threshold": 1.0
  },
  "metrics": {
    "avg_num_matches": 1247.3,
    "avg_epipolar_error": 0.42,
    "median_epipolar_error": 0.28,
    "avg_inlier_ratio": 0.9423,
    "pose_success_rate": 0.9733,
    "avg_rotation_error": 1.23,
    "avg_translation_error": 2.15,
    "median_rotation_error": 0.87,
    "median_translation_error": 1.45,
    "auc_5deg": 0.8533,
    "auc_10deg": 0.9267,
    "auc_20deg": 0.96
  }
}
```

## Python API

You can also use the benchmark programmatically:

```python
from matching_framework.testing.benchmark import run_benchmark
from argparse import Namespace

# Create args
args = Namespace(
    dataset="eth3d",
    dataset_path="data/eth3d",
    split="test",
    extractor="superpoint",
    matcher="lightglue",
    max_keypoints=4096,
    resize_max=None,
    distance_threshold=0.7,
    epipolar_threshold=1.0,
    ransac_threshold=1.0,
    num_samples=None,
    output="results/eth3d",
    device="cuda"
)

# Run benchmark
summary = run_benchmark(args)

# Access results
print(f"AUC@10°: {summary['metrics']['auc_10deg']:.2%}")
```

## Tips

1. **Start small**: Use `--num_samples 10` for quick tests
2. **Check data**: Ensure dataset paths and formats are correct
3. **GPU memory**: Reduce `--max_keypoints` if running out of memory
4. **CPU fallback**: Use `--device cpu` if CUDA not available
5. **Compare fairly**: Use same thresholds when comparing methods

## Troubleshooting

**"Dataset path does not exist"**
- Check `--dataset_path` points to the correct directory
- Ensure dataset is downloaded and extracted

**"No samples found"**
- Verify dataset structure matches expected format
- Check file extensions (.jpg, .png, .txt)

**"CUDA out of memory"**
- Reduce `--max_keypoints` (try 2048 or 1024)
- Use `--device cpu` for CPU-only processing

**"No matches for sample X"**
- Normal for low-overlap pairs
- Check if images are valid and not corrupted

## Citation

If you use this benchmarking framework in your research, please cite:

```bibtex
@software{matching_framework,
  title={Modular Feature Matching Framework},
  author={Enhanced SfM Team},
  year={2025},
  url={https://github.com/...}
}
```
