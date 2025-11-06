# Enhanced SfM Pipeline for 3D Gaussian Splatting

### Quick Install
```bash
# Clone repository
git clone https://github.com/jmhi9999/Enhanced-Structure-from-Motion
cd Enhanced-Structure-from-Motion

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install cupy-cuda12x faiss-gpu 
pip install timm networkx pandas

## Quick Start

### Basic SfM Reconstruction
```bash
python sfm_pipeline.py --input_dir path/to/images --output_dir path/to/output --feature_extractor aliked --use_brute_force
```

## Pipeline Architecture

```
Enhanced SfM Pipeline
├── Image Preprocessing → sfm/utils/image_utils.py
├── Feature Extraction → sfm/core/feature_extractor.py (SuperPoint/ALIKED)
├── Pair Selection → mpa/cli.py (DINOv2 embeddings + MST-Parallax Augment graph)
├── Feature Matching → sfm/core/feature_matcher.py (LightGlue + GPU brute force fallback)
├── Geometric Verification → sfm/core/geometric_verification.py (cv2.USAC_MAGSAC)
├── SfM Reconstruction → sfm/core/colmap_binary.py (COLMAP binary execution)
└── COLMAP Output → 3DGS Ready
```

## Configuration Options

### Feature Extraction
```bash
--feature_extractor superpoint     # SuperPoint, ALIKED, or DISK
--max_image_size 1600              # Maximum image size for processing
--max_keypoints 4096               # Maximum keypoints per image
```

### Pair Selection and Matching
```bash
--use_brute_force                  # GPU brute force matching (default)
--max_total_pairs 10000            # Maximum total pairs for brute force
--mpa_knn_k 30                     # DINO k-NN candidates per image
--mpa_top_t_mutual 128             # Mutual descriptor matches per edge
--mpa_loop_budget_per_node 0.5     # High-parallax loop augmentation budget
--mpa_tau_overlap 0.10             # Minimum overlap threshold
--mpa_tau_parallax 0.05            # Minimum parallax threshold
--mpa_device cuda                  # Device for DINO embedding extraction
```
## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+ -> 1.26.4 recommended
- timm 0.9+
- networkx 3.2+
- pandas 2.1+
- COLMAP binary (must be in PATH)

