# Modular Feature Matching Framework - Complete Overview

## What is This?

A professional, hloc-inspired framework for feature extraction, matching, and pair selection with:

- **Modular design** with clean parent/child class architecture
- **Dynamic component switching** via CLI
- **Factory pattern** for easy instantiation
- **Comprehensive dataset testing** on ETH3D, ScanNet, MegaDepth
- **Production-ready** with proper abstractions and interfaces

## Complete Architecture

```
matching_framework/
├── extractors/                    # Feature Extraction
│   ├── base.py                    # BaseFeatureExtractor + FeatureData + ExtractorConfig
│   ├── sift.py                    # SIFT extractor (128-dim float)
│   ├── orb.py                     # ORB extractor (256-bit binary)
│   ├── superpoint.py              # SuperPoint extractor (256-dim float)
│   └── __init__.py                # ExtractorFactory
│
├── matchers/                      # Feature Matching
│   ├── base.py                    # BaseMatcher + MatchData + MatcherConfig
│   ├── nn.py                      # Nearest Neighbor (L2/Hamming)
│   ├── lightglue.py               # LightGlue (attention-based)
│   └── __init__.py                # MatcherFactory
│
├── pair_selectors/                # Pair Selection
│   ├── base.py                    # BasePairSelector + PairSelectorConfig
│   ├── exhaustive.py              # O(n²) exhaustive pairs
│   ├── mpa.py                     # O(n log n) MPA integration
│   └── __init__.py                # PairSelectorFactory
│
├── testing/                       # Dataset Testing Framework
│   ├── datasets/
│   │   ├── base.py                # BaseDataset + DatasetSample
│   │   ├── eth3d.py               # ETH3D loader
│   │   ├── scannet.py             # ScanNet loader
│   │   ├── megadepth.py           # MegaDepth loader
│   │   └── __init__.py            # DatasetFactory
│   ├── metrics.py                 # Evaluation metrics (epipolar, pose, AUC)
│   ├── benchmark.py               # Main benchmark script
│   └── __init__.py
│
├── cli.py                         # Command-line interface
├── example.py                     # Quick example script
├── __init__.py                    # Main exports
├── README.md                      # Main documentation
├── TESTING_GUIDE.md               # Dataset testing guide
└── OVERVIEW.md                    # This file
```

## Key Design Principles

### 1. Parent-Child Class Architecture

**Base Classes:**
- `BaseFeatureExtractor`: Abstract extractor interface
- `BaseMatcher`: Abstract matcher interface
- `BasePairSelector`: Abstract pair selector interface
- `BaseDataset`: Abstract dataset loader interface

**Child Classes Inherit:**
- SIFT, ORB, SuperPoint inherit from `BaseFeatureExtractor`
- NN, LightGlue inherit from `BaseMatcher`
- Exhaustive, MPA inherit from `BasePairSelector`
- ETH3D, ScanNet, MegaDepth inherit from `BaseDataset`

### 2. Factory Pattern

All components use factories for dynamic instantiation:

```python
# Create any extractor by name
extractor = ExtractorFactory.create("superpoint", config)

# Create any matcher by name
matcher = MatcherFactory.create("lightglue", config, extractor_info)

# Create any pair selector by name
selector = PairSelectorFactory.create("mpa", config)

# Create any dataset by name
dataset = DatasetFactory.create("eth3d", dataset_path)
```

### 3. Information Flow

**Extractor → Matcher:**
- Matcher receives descriptor dimension and type from extractor
- Enables automatic distance metric selection (L2 for float, Hamming for binary)

```python
# Extractor provides config
extractor_config = extractor.get_config_dict()
# Contains: descriptor_dim, descriptor_type, name

# Matcher uses this info
matcher = MatcherFactory.create("nn", matcher_config, extractor_config)
```

### 4. Configuration System

Type-safe dataclasses for all configurations:

```python
@dataclass
class ExtractorConfig:
    max_keypoints: int = 4096
    device: str = "cuda"
    # ... other params

@dataclass
class MatcherConfig:
    distance_threshold: float = 0.7
    mutual_check: bool = True
    # ... other params

@dataclass
class PairSelectorConfig:
    max_pairs_per_image: int = 20
    # ... other params
```

## Component Overview

### Feature Extractors

| Name | Parent | Descriptor | Type | Speed |
|------|--------|------------|------|-------|
| SIFT | BaseFeatureExtractor | 128-dim | float | Medium |
| ORB | BaseFeatureExtractor | 256-bit | binary | Fast |
| SuperPoint | BaseFeatureExtractor | 256-dim | float | Medium |

**Key Methods:**
- `extract(image_path)` → `FeatureData`
- `extract_batch(image_list)` → `Dict[Path, FeatureData]`
- `get_config_dict()` → `dict` (for matcher)

### Feature Matchers

| Name | Parent | Supports | Method |
|------|--------|----------|--------|
| NN | BaseMatcher | All extractors | Nearest neighbor + Lowe's ratio |
| LightGlue | BaseMatcher | SuperPoint, ALIKED, DISK | Attention-based |

**Key Methods:**
- `match(features0, features1)` → `MatchData`
- `match_pairs(features_dict, pairs)` → `Dict[tuple, MatchData]`
- `geometric_verification(...)` → Verified matches + F-matrix

### Pair Selectors

| Name | Parent | Complexity | Best For |
|------|--------|------------|----------|
| Exhaustive | BasePairSelector | O(n²) | Small datasets (<100) |
| MPA | BasePairSelector | O(n log n) | Large datasets (1000+) |

**Key Methods:**
- `select(image_list, features=None)` → `List[Tuple[Path, Path]]`
- `get_statistics(image_list, pairs)` → Statistics dict

### Dataset Loaders

| Name | Parent | Type | Ground Truth |
|------|--------|------|--------------|
| ETH3D | BaseDataset | Indoor/Outdoor | Poses + intrinsics |
| ScanNet | BaseDataset | RGB-D | Poses + depth + intrinsics |
| MegaDepth | BaseDataset | Outdoor | Poses + intrinsics |

**Key Methods:**
- `__len__()` → Number of samples
- `__getitem__(idx)` → `DatasetSample`

## Usage Examples

### 1. Basic Feature Matching

```bash
python -m matching_framework.cli \
    --images data/images \
    --output results \
    --extractor sift \
    --matcher nn \
    --pair_selector exhaustive
```

### 2. High-Quality GPU Matching

```bash
python -m matching_framework.cli \
    --images data/images \
    --output results \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --device cuda
```

### 3. Dataset Benchmarking

```bash
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor superpoint \
    --matcher lightglue \
    --output results/eth3d
```

### 4. Python API

```python
from matching_framework import (
    ExtractorFactory, ExtractorConfig,
    MatcherFactory, MatcherConfig,
)

# Extract features
config = ExtractorConfig(max_keypoints=4096, device="cuda")
extractor = ExtractorFactory.create("superpoint", config)
features = extractor.extract(image_path)

# Match
matcher_config = MatcherConfig(device="cuda")
matcher = MatcherFactory.create(
    "lightglue",
    matcher_config,
    extractor.get_config_dict()
)
matches = matcher.match(features0, features1)
```

## Evaluation Metrics

### Match Quality
- **Number of matches**: Total correspondences
- **Epipolar error**: Distance to epipolar line (pixels)
- **Inlier ratio**: Fraction within threshold
- **Depth error**: 3D reprojection error (if depth available)

### Pose Estimation
- **Success rate**: Fraction with valid pose
- **Rotation error**: Angular difference (degrees)
- **Translation error**: Angular difference (degrees)
- **AUC@θ**: Fraction with errors < θ (θ = 5°, 10°, 20°)

## Extension Guide

### Adding a New Extractor

1. Create `extractors/my_extractor.py`:

```python
from .base import BaseFeatureExtractor, FeatureData

class MyExtractor(BaseFeatureExtractor):
    def _setup(self):
        # Initialize your model
        pass

    def _extract_impl(self, image: np.ndarray) -> FeatureData:
        # Extract features
        return FeatureData(keypoints, descriptors, scores, image_shape)

    @property
    def descriptor_dim(self) -> int:
        return 128

    @property
    def descriptor_type(self) -> str:
        return "float"  # or "binary"
```

2. Register in `extractors/__init__.py`:

```python
from .my_extractor import MyExtractor

class ExtractorFactory:
    _extractors = {
        # ...
        "my_extractor": MyExtractor,
    }
```

### Adding a New Dataset

1. Create `testing/datasets/my_dataset.py`:

```python
from .base import BaseDataset, DatasetSample

class MyDataset(BaseDataset):
    def _load_samples(self) -> List[DatasetSample]:
        # Load your dataset
        samples = []
        # ... parse dataset files
        for ... in ...:
            samples.append(DatasetSample(
                image0=img0_path,
                image1=img1_path,
                K0=K0,
                K1=K1,
                T_0to1=T_0to1,
            ))
        return samples
```

2. Register in `testing/datasets/__init__.py`:

```python
from .my_dataset import MyDataset

class DatasetFactory:
    _datasets = {
        # ...
        "my_dataset": MyDataset,
    }
```

## File Descriptions

### Core Files

- **`__init__.py`**: Main exports (factories, configs, data classes)
- **`cli.py`**: CLI for running feature matching pipelines
- **`example.py`**: Quick example demonstrating the framework

### Extractor Files

- **`extractors/base.py`**: Abstract base class + data structures
- **`extractors/sift.py`**: OpenCV SIFT implementation
- **`extractors/orb.py`**: OpenCV ORB implementation
- **`extractors/superpoint.py`**: LightGlue SuperPoint integration

### Matcher Files

- **`matchers/base.py`**: Abstract base class + match data structures
- **`matchers/nn.py`**: Nearest neighbor with Lowe's ratio test
- **`matchers/lightglue.py`**: LightGlue attention-based matcher

### Pair Selector Files

- **`pair_selectors/base.py`**: Abstract base class
- **`pair_selectors/exhaustive.py`**: All-pairs selection
- **`pair_selectors/mpa.py`**: MPA integration

### Testing Files

- **`testing/benchmark.py`**: Main benchmarking script
- **`testing/metrics.py`**: Evaluation metrics implementation
- **`testing/datasets/base.py`**: Dataset base class
- **`testing/datasets/eth3d.py`**: ETH3D dataset loader
- **`testing/datasets/scannet.py`**: ScanNet dataset loader
- **`testing/datasets/megadepth.py`**: MegaDepth dataset loader

### Documentation

- **`README.md`**: Main documentation with usage examples
- **`TESTING_GUIDE.md`**: Comprehensive guide for dataset testing
- **`OVERVIEW.md`**: This file (architecture overview)

## Dependencies

### Required
```bash
pip install numpy opencv-python torch
```

### Optional
```bash
# For LightGlue matcher
pip install lightglue

# For dataset testing
pip install tqdm
```

## Quick Reference

### List Available Components

```python
from matching_framework import (
    ExtractorFactory,
    MatcherFactory,
    PairSelectorFactory,
)
from matching_framework.testing.datasets import DatasetFactory

print("Extractors:", ExtractorFactory.list_extractors())
# ['sift', 'orb', 'superpoint']

print("Matchers:", MatcherFactory.list_matchers())
# ['nn', 'lightglue']

print("Pair Selectors:", PairSelectorFactory.list_selectors())
# ['exhaustive', 'mpa']

print("Datasets:", DatasetFactory.list_datasets())
# ['eth3d', 'scannet', 'megadepth']
```

### Switch Components Dynamically

```bash
# SIFT + NN
python -m matching_framework.cli --extractor sift --matcher nn ...

# SuperPoint + LightGlue
python -m matching_framework.cli --extractor superpoint --matcher lightglue ...

# ORB + NN
python -m matching_framework.cli --extractor orb --matcher nn ...
```

### Run Quick Test

```bash
# Run example script
python matching_framework/example.py

# Run small benchmark
python -m matching_framework.testing.benchmark \
    --dataset eth3d \
    --dataset_path data/eth3d \
    --extractor sift \
    --matcher nn \
    --num_samples 10 \
    --output results/test
```

## Performance Recommendations

| Scenario | Extractor | Matcher | Pair Selector | Device |
|----------|-----------|---------|---------------|--------|
| CPU-only | SIFT or ORB | NN | Exhaustive | CPU |
| Small dataset (<100) | SIFT | NN | Exhaustive | CPU/GPU |
| Medium dataset (100-1000) | SuperPoint | LightGlue | MPA | GPU |
| Large dataset (1000+) | SuperPoint | LightGlue | MPA | GPU |
| Best quality | SuperPoint | LightGlue | MPA | GPU |
| Fastest | ORB | NN | Exhaustive | CPU |

## License

MIT License

## Citation

```bibtex
@software{matching_framework,
  title={Modular Feature Matching Framework},
  author={Enhanced SfM Team},
  year={2025},
  url={https://github.com/...}
}
```

## Contact

For issues, questions, or contributions, please see the repository.
