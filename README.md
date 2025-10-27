# Enhanced Structure-from-Motion with DINOv3

**A global SfM pipeline powered by DINOv3 features, LoFTR matching, and GLOMAP reconstruction.**

This repository implements the DINOv3-SfM system described in `DINOv2_SfM_Proposal.md`, featuring:
- 🔍 **CLS token–based retrieval** (FAISS) for efficient pair selection
- 🎯 **LoFTR dense matching** with DINO attention guidance
- 📊 **Context-aware Bundle Adjustment** with attention weighting
- 🌐 **Global SfM** via GLOMAP for scale consistency and robust merging

## Quick Install
```bash
# Clone repository
git clone https://github.com/jmhi9999/Enhanced-Structure-from-Motion
cd Enhanced-Structure-from-Motion

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install cupy-cuda12x faiss-gpu

# Install GLOMAP (global SfM backend)
# Option 1: Download precompiled binary to bin/ directory
# Windows: Place glomap.exe in bin/
# Linux/Mac: Place glomap in bin/ and chmod +x bin/glomap
# Download from: https://github.com/colmap/glomap/releases

# Option 2: Build from source
# Follow instructions at: https://github.com/colmap/glomap
```

## Quick Start

### Basic Usage (DINOv3-SfM Pipeline)
```bash
python sfm_pipeline.py \
  --input_dir path/to/images \
  --output_dir path/to/output
```

### Advanced Configuration
```bash
python sfm_pipeline.py \
  --input_dir path/to/images \
  --output_dir path/to/output \
  --dino_model_family dinov3 \
  --dino_model_name dinov3_vitl14 \
  --dino_topk_primary 60 \
  --dino_topk_final 20 \
  --loftr_pretrained outdoor \
  --confidence_mode rule_based
```

## Pipeline Architecture

### DINOv3-SfM Pipeline (Primary)
```
DINOv3-SfM Pipeline
├── DINOv3 Feature Extraction → sfm/features/dino_feature_extractor.py
│   ├─ CLS Token (global descriptor)
│   └─ Patch Tokens + Attention Maps
├── CLS-based Retrieval → sfm/retrieval/dino_retriever.py (FAISS Top-K)
├── LoFTR Matching → sfm/matching/loftr_matcher.py (attention-guided)
├── DINO Patch Fallback → sfm/matching/dino_matcher.py (coarse geometry)
├── MAGSAC++ Verification → sfm/core/geometric_verification.py
├── Scene Graph → sfm/core/context_ba/scene_graph.py
├── Global Reconstruction → GLOMAP (rotation/translation averaging)
├── Context-Aware BA → sfm/core/context_ba/optimizer.py
└── Output → 3DGS Ready
```

### Legacy Pipeline (Traditional)
```
Traditional SfM Pipeline
├── Feature Extraction → sfm/core/feature_extractor.py (SuperPoint/ALIKED)
├── Pair Selection → sfm/core/gpu_vocabulary_tree.py (FAISS vocab tree)
├── Feature Matching → sfm/core/feature_matcher.py (LightGlue)
├── Geometric Verification → sfm/core/geometric_verification.py
├── SfM Reconstruction → sfm/core/colmap_binary.py (incremental)
└── Output → 3DGS Ready
```

## Configuration Options

### DINOv3 Feature Extraction
```bash
--dino_model_family dinov3         # dinov3 or dinov2 (default: dinov3)
--dino_model_name dinov3_vitl14    # Model variant (default: dinov3_vitl14)
--max_image_size 1600              # Maximum image size for processing
```

### CLS-based Retrieval
```bash
--dino_topk_primary 60             # Initial CLS retrieval candidates (default: 60)
--dino_topk_final 20               # Final pairs after re-ranking (default: 20)
```

### LoFTR Matching
```bash
--loftr_pretrained outdoor         # LoFTR weights: outdoor/indoor (default: outdoor)
--loftr_max_long_edge 832          # LoFTR maximum long edge (default: 832)
--dino_use_attention_guidance      # Enable attention-guided LoFTR (default: True)
```

### Geometric Verification
```bash
--min_inliers 15                   # Minimum inliers for valid pair (default: 15)
--magsac_threshold 2.0             # MAGSAC++ threshold (default: 2.0)
```

### Context-Aware BA
```bash
--confidence_mode rule_based       # rule_based or hybrid (default: rule_based)
--context_ba_checkpoint path       # Hybrid MLP checkpoint (optional)
```
## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+ (with CUDA recommended)
- OpenCV 4.8+
- NumPy 1.26.4
- FAISS-GPU (for CLS retrieval)
- Kornia 0.6.12+ (for LoFTR)
- GLOMAP (for global reconstruction) - See installation guide
- Optional: COLMAP (for legacy pipeline)

