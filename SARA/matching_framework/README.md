# Modular Feature Matching Framework

A professional, hloc-inspired framework for feature extraction, matching, and pair selection with dynamic component switching.

## Features

- 🎯 **Modular Architecture**: Clean separation of extractors, matchers, and pair selectors
- 🔄 **Dynamic Switching**: Change components via CLI without code modification
- 🏭 **Factory Pattern**: Easy instantiation with factories
- 📊 **Dataset Testing**: Built-in benchmarking support
- 🎨 **Clean API**: Simple and intuitive interface

## Architecture

```
matching_framework/
├── extractors/         # Feature extraction (SIFT, ORB, SuperPoint, ALIKED, DISK)
├── matchers/           # Feature matching (NN, LightGlue)
├── pair_selectors/     # Pair selection (Exhaustive, VocabTree, MPA)
├── utils/              # Utilities
├── testing/            # Dataset testing framework
└── cli.py              # Command-line interface
```

## Installation

```bash
# Required
pip install opencv-python numpy torch

# Optional (for learning-based methods)
pip install lightglue

# For MPA pair selection
# (MPA already in parent directory)
```

## Usage

### Command Line

#### Example 1: SIFT + NN Matcher + Exhaustive Pairs
```bash
python -m matching_framework.cli \
    --images data/images \
    --output results \
    --extractor sift \
    --matcher nn \
    --pair_selector exhaustive \
    --geometric_verification \
    --save_visualization
```

#### Example 2: SuperPoint + LightGlue + MPA
```bash
python -m matching_framework.cli \
    --images data/images \
    --output results \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --max_keypoints 4096 \
    --device cuda
```

#### Example 3: ORB + NN + Exhaustive (fast, CPU-friendly)
```bash
python -m matching_framework.cli \
    --images data/images \
    --output results \
    --extractor orb \
    --matcher nn \
    --pair_selector exhaustive \
    --device cpu
```

### Python API

```python
from matching_framework import (
    ExtractorFactory, ExtractorConfig,
    MatcherFactory, MatcherConfig,
    PairSelectorFactory, PairSelectorConfig
)
from pathlib import Path

# 1. Extract features
extractor_config = ExtractorConfig(
    max_keypoints=4096,
    device="cuda"
)
extractor = ExtractorFactory.create("superpoint", extractor_config)

image_list = list(Path("data/images").glob("*.jpg"))
features_dict = extractor.extract_batch(image_list)

# 2. Select pairs
pair_selector_config = PairSelectorConfig(max_pairs_per_image=20)
pair_selector = PairSelectorFactory.create("mpa", pair_selector_config)

pairs = pair_selector.select(image_list, features=features_dict)

# 3. Match features
matcher_config = MatcherConfig(device="cuda")
matcher = MatcherFactory.create(
    "lightglue",
    matcher_config,
    extractor.get_config_dict()
)

matches_dict = matcher.match_pairs(features_dict, pairs)

# 4. Geometric verification (optional)
for (path0, path1), matches in matches_dict.items():
    verified, F = matcher.geometric_verification(
        features_dict[path0],
        features_dict[path1],
        matches,
        threshold=1.0
    )
    print(f"{path0.name} <-> {path1.name}: {len(verified)} inliers")
```

## Supported Components

### Feature Extractors

| Name | Type | Descriptor | Speed | Quality |
|------|------|------------|-------|---------|
| `sift` | Classical | 128-dim float | Medium | Good |
| `orb` | Classical | 256-bit binary | Fast | Medium |
| `superpoint` | Learning | 256-dim float | Medium | Excellent |
| `aliked` | Learning | 128-dim float | Fast | Excellent |
| `disk` | Learning | 128-dim float | Medium | Excellent |

### Matchers

| Name | Type | Supports | Speed | Quality |
|------|------|----------|-------|---------|
| `nn` | Traditional | All | Fast | Good |
| `lightglue` | Learning | SuperPoint, ALIKED, DISK | Medium | Excellent |

### Pair Selectors

| Name | Complexity | Best For |
|------|------------|----------|
| `exhaustive` | O(n²) | Small datasets (<100 images) |
| `vocab_tree` | O(n log n) | Medium datasets (100-1000 images) |
| `mpa` | O(n log n) | Large datasets (1000+ images) |

## Configuration

### Extractor Configuration

```python
ExtractorConfig(
    max_keypoints=4096,          # Max keypoints per image
    detection_threshold=0.005,    # Detection confidence threshold
    edge_threshold=10.0,          # Edge rejection threshold
    resize_max=None,              # Resize images (None = no resize)
    grayscale=True,               # Convert to grayscale
    device="cuda",                # Device
    extra={}                      # Extractor-specific params
)
```

### Matcher Configuration

```python
MatcherConfig(
    distance_threshold=0.7,       # Lowe's ratio or distance threshold
    mutual_check=True,            # Enable mutual nearest neighbor check
    max_matches=None,             # Limit matches (None = unlimited)
    device="cuda",
    extra={}                      # Matcher-specific params
)
```

### Pair Selector Configuration

```python
PairSelectorConfig(
    max_pairs_per_image=20,       # Max pairs per image
    min_pairs_per_image=1,        # Min pairs for connectivity
    device="cuda",
    extra={}                      # Selector-specific params
)
```

## Extending the Framework

### Adding a New Extractor

```python
from matching_framework.extractors.base import BaseFeatureExtractor, FeatureData

class MyExtractor(BaseFeatureExtractor):
    def _setup(self):
        # Initialize your extractor
        pass

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        # Extract features
        return FeatureData(
            keypoints=kpts,
            descriptors=desc,
            scores=scores,
            image_shape=image.shape[:2]
        )

    @property
    def descriptor_dim(self) -> int:
        return 128  # Your descriptor dimension

    @property
    def descriptor_type(self) -> str:
        return "float"  # or "binary"

    @property
    def supports_batching(self) -> bool:
        return False
```

Then register in `extractors/__init__.py`:
```python
from .my_extractor import MyExtractor

class ExtractorFactory:
    _extractors = {
        # ...
        "myextractor": MyExtractor,
    }
```

## SfM Reconstruction Pipeline

The framework includes a complete Structure-from-Motion reconstruction pipeline that uses modular extractors, matchers, and pair selectors to perform full 3D reconstruction and collect statistics (even without ground truth).

### Quick Start

```bash
# Run reconstruction with SuperPoint + LightGlue
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple_sp_lg \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --device cuda

# Run with SIFT + NN (CPU-friendly)
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple_sift_nn \
    --extractor sift \
    --matcher nn \
    --pair_selector exhaustive \
    --device cpu
```

### Output Statistics (No GT Required)

The pipeline collects the following statistics automatically:

**Feature Extraction & Matching:**
- Number of images
- Number of pairs
- Total matches
- Average matches per pair
- Extraction time
- Matching time

**Reconstruction Quality:**
- Number of registered images
- Registration rate (%)
- Number of 3D points
- Number of observations
- Mean/median track length
- Mean/median reprojection error (px)
- Reconstruction time

**Output Files:**
- `statistics.json` - Full statistics in JSON format
- `statistics.csv` - Statistics in CSV format for easy comparison
- `sparse/` - COLMAP sparse reconstruction
- `database.db` - COLMAP database

### Comparing Multiple Configurations

Run multiple experiments and compare results:

```bash
# Experiment 1: SuperPoint + LightGlue
python -m matching_framework.sfm_pipeline \
    --images data/temple --output results/sp_lg \
    --extractor superpoint --matcher lightglue

# Experiment 2: SIFT + NN
python -m matching_framework.sfm_pipeline \
    --images data/temple --output results/sift_nn \
    --extractor sift --matcher nn

# Experiment 3: ORB + NN
python -m matching_framework.sfm_pipeline \
    --images data/temple --output results/orb_nn \
    --extractor orb --matcher nn

# Compare all experiments
python -m matching_framework.testing.compare_reconstructions \
    results/sp_lg/statistics.csv \
    results/sift_nn/statistics.csv \
    results/orb_nn/statistics.csv \
    --output comparison.csv
```

The comparison tool will display a side-by-side table and automatically identify:
- Best registration rate
- Most 3D points
- Lowest reprojection error
- Fastest configuration

## Dataset Testing

The framework includes a comprehensive benchmarking system for evaluating feature matching on standard datasets (with ground truth).

### Supported Datasets

| Dataset | Type | Metrics | Ground Truth |
|---------|------|---------|--------------|
| ETH3D | Indoor/Outdoor | Pose, Epipolar | Camera poses + intrinsics |
| ScanNet | Indoor RGB-D | Pose, Epipolar, Depth | Camera poses + depth maps |
| MegaDepth | Outdoor landmarks | Pose, Epipolar | Camera poses + intrinsics |

### Quick Start

```bash
# Run on ETH3D dataset (raw matching evaluation)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --output results/eth3d

# Run with MAGSAC filtering (realistic SfM evaluation)
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --use_magsac_filtering \
    --output results/eth3d_magsac

# Run on ScanNet with depth evaluation
python -m matching_framework.testing.benchmark \
    --dataset scannet \
    --dataset_path data/scannet \
    --extractor sift \
    --matcher nn \
    --output results/scannet

# Run on MegaDepth with limited samples
python -m matching_framework.testing.benchmark \
    --dataset megadepth \
    --dataset_path data/megadepth \
    --extractor superpoint \
    --matcher lightglue \
    --num_samples 100 \
    --output results/megadepth
```

### Evaluation Modes

**Two evaluation modes are available:**

1. **Raw Matching Evaluation (default):** Evaluates pure matching quality without geometric filtering
2. **Realistic SfM Evaluation (`--use_magsac_filtering`):** Applies MAGSAC filtering (same as `sfm_pipeline.py`) for realistic SfM performance prediction

**When to use MAGSAC filtering:**
- Testing how a matcher performs in actual SfM reconstruction
- Comparing against production SfM systems
- Evaluating robustness to outliers

**When NOT to use MAGSAC filtering:**
- Measuring raw matching discriminative power
- Comparing pure matcher performance
- Analyzing matching behavior before geometric verification

### Evaluation Metrics

The benchmark computes the following metrics:

**Match Quality Metrics:**
- Number of matches per pair
- Epipolar error (mean and median, in pixels)
- Inlier ratio (fraction within epipolar threshold)
- Depth reprojection error (if depth maps available)

**Pose Estimation Metrics:**
- Pose estimation success rate
- Rotation error (degrees)
- Translation error (angular, degrees)
- AUC@5°, AUC@10°, AUC@20° (pose accuracy thresholds)

### Output

The benchmark produces two output files:

1. **`results.json`**: Detailed per-sample results
2. **`summary.json`**: Aggregated metrics and configuration

Example summary output:
```json
{
  "dataset": "eth3d",
  "extractor": "superpoint",
  "matcher": "lightglue",
  "num_samples": 150,
  "metrics": {
    "avg_num_matches": 1247.3,
    "avg_epipolar_error": 0.42,
    "avg_inlier_ratio": 0.94,
    "pose_success_rate": 0.97,
    "avg_rotation_error": 1.23,
    "avg_translation_error": 2.15,
    "auc_5deg": 0.85,
    "auc_10deg": 0.92,
    "auc_20deg": 0.96
  }
}
```

### Dataset Format

Each dataset should follow the expected directory structure:

**ETH3D:**
```
eth3d/
├── scene1/
│   ├── images/
│   │   ├── image0000.jpg
│   │   └── ...
│   ├── calibration.txt      # fx fy cx cy
│   └── poses.txt            # Image name + 4x4 pose matrix
└── scene2/
    └── ...
```

**ScanNet:**
```
scannet/
├── scene0000_00/
│   ├── color/
│   │   ├── 0.jpg
│   │   └── ...
│   ├── depth/
│   │   ├── 0.png           # Depth in mm (uint16)
│   │   └── ...
│   ├── pose/
│   │   ├── 0.txt           # 4x4 pose matrix
│   │   └── ...
│   └── intrinsic.txt       # 4x4 intrinsic matrix
└── scene0001_00/
    └── ...
```

**MegaDepth:**
```
megadepth/
├── scene_info/
│   ├── scene0000.npz       # K, T for each image
│   └── ...
├── pairs.json              # List of image pairs
└── scene0000/
    └── images/
        ├── image1.jpg
        └── ...
```
