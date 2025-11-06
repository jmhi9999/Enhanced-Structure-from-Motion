# SfM Reconstruction Pipeline Guide

## Overview

The `sfm_pipeline.py` script provides a complete Structure-from-Motion reconstruction pipeline using `matching_framework` components. It performs full 3D reconstruction and collects comprehensive statistics **without requiring ground truth data**.

## Key Features

- ✅ **Modular components**: Use any extractor/matcher/pair selector from matching_framework
- ✅ **COLMAP integration**: Leverages existing `sfm/core/colmap_binary.py` for reconstruction
- ✅ **Comprehensive statistics**: Collects 15+ metrics automatically
- ✅ **CSV export**: Easy comparison across experiments
- ✅ **Comparison tool**: Automatic side-by-side analysis
- ✅ **No GT required**: Works on any image folder

## Usage

### Basic Reconstruction

```bash
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --device cuda
```

### All Options

```bash
python -m matching_framework.sfm_pipeline \
    --images INPUT_DIR \
    --output OUTPUT_DIR \
    \
    # Feature extractor
    --extractor {sift,orb,superpoint,aliked,disk} \
    --max_keypoints 4096 \
    --resize_max 1024 \
    \
    # Matcher
    --matcher {nn,lightglue} \
    --distance_threshold 0.7 \
    \
    # Pair selector
    --pair_selector {exhaustive,mpa} \
    --max_pairs_per_image 20 \
    \
    # MAGSAC filtering (optional, applied by COLMAP internally)
    --use_magsac_filtering \
    --magsac_threshold 2.0 \
    --magsac_confidence 0.999 \
    --magsac_max_iters 1000 \
    \
    # COLMAP
    --colmap_executable colmap \
    --skip_reconstruction  # Skip COLMAP reconstruction (features/matching only)
    \
    # Device
    --device {cuda,cpu}
```

## Output Statistics

### Feature Extraction & Matching

| Metric | Description |
|--------|-------------|
| `num_images` | Total number of images |
| `num_pairs` | Number of image pairs selected |
| `num_matches_total` | Total feature matches across all pairs |
| `avg_matches_per_pair` | Average matches per pair |
| `feature_extraction_time` | Time spent extracting features (seconds) |
| `matching_time` | Time spent matching features (seconds) |

### Reconstruction Quality

| Metric | Description |
|--------|-------------|
| `num_registered_images` | Number of images successfully registered |
| `registration_rate` | Fraction of images registered (0-1) |
| `num_3d_points` | Number of 3D points reconstructed |
| `num_observations` | Total 2D-3D correspondences |
| `mean_track_length` | Average number of views per 3D point |
| `median_track_length` | Median track length |
| `mean_reprojection_error` | Mean reprojection error (pixels) |
| `median_reprojection_error` | Median reprojection error (pixels) |
| `reconstruction_time` | Time spent in COLMAP reconstruction (seconds) |
| `total_time` | Total pipeline time (seconds) |

## Output Files

```
output_dir/
├── statistics.json           # Full statistics in JSON
├── statistics.csv            # Statistics in CSV (for comparison)
├── database.db               # COLMAP database
├── sparse/                   # COLMAP sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── (optional) dense/         # Dense reconstruction (if run separately)
```

## Comparing Experiments

### Step 1: Run Multiple Experiments

```bash
# Experiment 1: SuperPoint + LightGlue + MPA
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple_sp_lg_mpa \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --device cuda

# Experiment 2: SIFT + NN + Exhaustive
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple_sift_nn_exh \
    --extractor sift \
    --matcher nn \
    --pair_selector exhaustive \
    --device cpu

# Experiment 3: ORB + NN + Exhaustive (fastest)
python -m matching_framework.sfm_pipeline \
    --images data/temple \
    --output results/temple_orb_nn_exh \
    --extractor orb \
    --matcher nn \
    --pair_selector exhaustive \
    --device cpu
```

### Step 2: Compare Results

```bash
python -m matching_framework.testing.compare_reconstructions \
    results/temple_sp_lg_mpa/statistics.csv \
    results/temple_sift_nn_exh/statistics.csv \
    results/temple_orb_nn_exh/statistics.csv \
    --output temple_comparison.csv
```

### Example Output

```
======================================================================================================================
RECONSTRUCTION COMPARISON
======================================================================================================================
Metric                                  temple_sp_lg_mpa  temple_sift_nn_exh  temple_orb_nn_exh
----------------------------------------------------------------------------------------------------------------------
extractor                                     superpoint            sift            orb
matcher                                       lightglue              nn              nn
pair_selector                                       mpa      exhaustive      exhaustive
device                                             cuda             cpu             cpu
num_images                                           50              50              50
num_pairs                                          1000            1225            1225
num_matches_total                                125430           98234           72145
avg_matches_per_pair                              125.4            80.2            58.9
feature_extraction_time                           12.34           23.45           15.67
matching_time                                     45.67           32.10           28.90
num_registered_images                                48              45              42
registration_rate                                 0.960           0.900           0.840
num_3d_points                                     45230           38120           29840
num_observations                                 342150          285900          223680
mean_track_length                                  7.56            7.50            7.50
median_track_length                                   7               7               7
mean_reprojection_error                           0.342           0.456           0.523
median_reprojection_error                         0.298           0.401           0.478
reconstruction_time                               89.12           67.34           54.21
total_time                                       147.13          122.89          98.78
======================================================================================================================

======================================================================================================================
KEY INSIGHTS
======================================================================================================================
Best registration rate: temple_sp_lg_mpa
  superpoint + lightglue
  0.960 registration rate

Most 3D points: temple_sp_lg_mpa
  45230 points

Lowest reprojection error: temple_sp_lg_mpa
  0.342 px

Fastest: temple_orb_nn_exh
  98.78s total
======================================================================================================================
```

## Use Cases

### 1. Algorithm Selection

**Goal:** Choose the best extractor/matcher for your dataset

```bash
# Test different extractors with same matcher
python -m matching_framework.sfm_pipeline --images data/mine --output results/sift --extractor sift --matcher nn
python -m matching_framework.sfm_pipeline --images data/mine --output results/orb --extractor orb --matcher nn
python -m matching_framework.sfm_pipeline --images data/mine --output results/sp --extractor superpoint --matcher lightglue

# Compare
python -m matching_framework.testing.compare_reconstructions results/*/statistics.csv
```

**Look at:** `registration_rate`, `num_3d_points`, `mean_reprojection_error`

### 2. Speed Optimization

**Goal:** Find fastest configuration with acceptable quality

```bash
# Test different devices and pair selectors
python -m matching_framework.sfm_pipeline --images data/mine --output results/gpu_mpa --device cuda --pair_selector mpa
python -m matching_framework.sfm_pipeline --images data/mine --output results/gpu_exh --device cuda --pair_selector exhaustive
python -m matching_framework.sfm_pipeline --images data/mine --output results/cpu_exh --device cpu --pair_selector exhaustive

# Compare
python -m matching_framework.testing.compare_reconstructions results/*/statistics.csv
```

**Look at:** `total_time`, `feature_extraction_time`, `matching_time`

### 3. Quality vs Speed Trade-off

**Goal:** Balance reconstruction quality and computational cost

```bash
# High quality (slow)
python -m matching_framework.sfm_pipeline \
    --images data/mine --output results/high_quality \
    --extractor superpoint --matcher lightglue --max_keypoints 8192

# Medium quality (balanced)
python -m matching_framework.sfm_pipeline \
    --images data/mine --output results/medium_quality \
    --extractor sift --matcher nn --max_keypoints 4096

# Fast (lower quality)
python -m matching_framework.sfm_pipeline \
    --images data/mine --output results/fast \
    --extractor orb --matcher nn --max_keypoints 2048

# Compare
python -m matching_framework.testing.compare_reconstructions results/*/statistics.csv
```

**Look at:** Balance between `total_time`, `registration_rate`, `num_3d_points`, `mean_reprojection_error`

### 4. Hyperparameter Tuning

**Goal:** Optimize parameters for specific dataset

```bash
# Test different keypoint counts
python -m matching_framework.sfm_pipeline --images data/mine --output results/kp_2048 --max_keypoints 2048 --extractor sift --matcher nn
python -m matching_framework.sfm_pipeline --images data/mine --output results/kp_4096 --max_keypoints 4096 --extractor sift --matcher nn
python -m matching_framework.sfm_pipeline --images data/mine --output results/kp_8192 --max_keypoints 8192 --extractor sift --matcher nn

# Compare
python -m matching_framework.testing.compare_reconstructions results/kp_*/statistics.csv
```

**Look at:** `avg_matches_per_pair`, `num_3d_points`, `feature_extraction_time`

## Integration with Existing Pipeline

The `sfm_pipeline.py` uses the existing `sfm/core/colmap_binary.py` module for COLMAP reconstruction:

1. **Feature extraction**: Uses `matching_framework` extractors
2. **Pair selection**: Uses `matching_framework` pair selectors
3. **Matching**: Uses `matching_framework` matchers
4. **Format conversion**: Converts to `colmap_binary.py` format
5. **Reconstruction**: Calls `colmap_binary_reconstruction()` from existing module
6. **Statistics extraction**: Parses COLMAP output

This ensures:
- ✅ Consistency with existing `sfm_pipeline.py` behavior
- ✅ MAGSAC filtering is applied (inside `colmap_binary_reconstruction`)
- ✅ Same COLMAP parameters
- ✅ Compatible output format

## Tips

1. **Start small**: Test with 10-20 images first using `--skip_reconstruction` to check features/matching
2. **Use exhaustive for small datasets**: `--pair_selector exhaustive` for <100 images
3. **Use MPA for large datasets**: `--pair_selector mpa` for 100+ images
4. **GPU acceleration**: Use `--device cuda` for learning-based methods (SuperPoint, LightGlue)
5. **CPU fallback**: SIFT/ORB work well on CPU with `--device cpu`
6. **Compare systematically**: Change one variable at a time for clearer insights

## Troubleshooting

**"Invalid feature data from superpoint" or "No features detected"**
- **Common causes**:
  - Images are too blurry, dark, or low texture
  - Images are corrupted or in unsupported format
  - Images are too small after resizing
- **Solutions**:
  1. Check image quality: `ls -lh ImageInputs/images_drjohnson/` (look for unusually small files)
  2. Try different extractor: `--extractor sift` (SIFT is more robust to poor images)
  3. Lower detection threshold: Use extractor config with lower threshold
  4. Remove problematic images manually
- **Expected behavior**: Pipeline will automatically skip images with <10 features and continue
- **Check logs**: Look for summary like "Feature extraction complete: 250/263 images successful, 13 with no features"

**"Too many images removed (insufficient features)"**
- **If >50% of images have no features**:
  1. **Wrong extractor for your images**:
     - Blurry/dark images → Try SIFT: `--extractor sift`
     - Texture-less images → Try ORB: `--extractor orb`
     - High-quality images → Use SuperPoint: `--extractor superpoint`
  2. **Images need preprocessing**:
     - Resize very large images: `--resize_max 1600`
     - Check image format (PNG, JPG, etc.)
  3. **Corrupted dataset**: Check first few images manually

**"COLMAP reconstruction failed"**
- Check that COLMAP is installed: `colmap --version`
- Try `--skip_reconstruction` to test features/matching only
- Check logs for specific COLMAP errors
- Ensure at least 10-20 images with sufficient features

**"No matches after MAGSAC filtering"**
- Reduce `--magsac_threshold` (try 1.0 instead of 2.0)
- Increase `--max_keypoints` for more features
- Check if images actually overlap

**"Low registration rate"**
- Try different extractor/matcher combination
- Increase `--max_pairs_per_image` for more connectivity
- Check image quality and overlap
- Verify images are from same scene/object

**"Out of memory"**
- Reduce `--max_keypoints` (try 2048 or 1024)
- Use `--device cpu` instead of `cuda`
- Use `--pair_selector exhaustive` only for small datasets
- Process images in smaller batches

## Example Workflows

### Workflow 1: Initial Exploration

```bash
# Quick test without reconstruction
python -m matching_framework.sfm_pipeline \
    --images data/mine \
    --output results/test \
    --extractor sift \
    --matcher nn \
    --skip_reconstruction

# Check statistics.csv for num_matches_total and avg_matches_per_pair
# If good, run full reconstruction
python -m matching_framework.sfm_pipeline \
    --images data/mine \
    --output results/full \
    --extractor sift \
    --matcher nn
```

### Workflow 2: Method Comparison

```bash
# Run all combinations
for ext in sift orb superpoint; do
  for mat in nn lightglue; do
    python -m matching_framework.sfm_pipeline \
      --images data/mine \
      --output results/${ext}_${mat} \
      --extractor $ext \
      --matcher $mat
  done
done

# Compare all
python -m matching_framework.testing.compare_reconstructions \
    results/*/statistics.csv \
    --output comparison.csv
```

### Workflow 3: Production Pipeline

```bash
# Use best configuration from comparison
python -m matching_framework.sfm_pipeline \
    --images data/production \
    --output results/production_$(date +%Y%m%d) \
    --extractor superpoint \
    --matcher lightglue \
    --pair_selector mpa \
    --max_keypoints 8192 \
    --device cuda

# Monitor statistics.csv for quality metrics
```

## Next Steps

After reconstruction, you can:
1. **Visualize**: Use COLMAP GUI to view sparse reconstruction
2. **Dense reconstruction**: Run COLMAP dense reconstruction separately
3. **Export**: Convert to other formats (PLY, OBJ, etc.)
4. **Analyze**: Use statistics.csv for quality analysis
5. **Compare**: Run multiple configurations and use comparison tool
6. **Leverage GPU matching**: Exhaustive + LightGlue on CUDA automatically switches to the GPU brute-force matcher for faster pairwise matching.

## Batch Automation (with Ground Truth)

For scenes that already contain COLMAP sparse models (e.g. `ImageInputs/bonsai`),
use the automation helper to run every extractor/matcher/pair-selection combination
and collect both SfM pipeline metrics and ground-truth benchmarks in one pass:

```bash
python -m matching_framework.testing.run_colmap_suite \
    --images-root ImageInputs \
    --output-root results/colmap_suite \
    --device cuda \
    --colmap_min_shared_points 1000 \
    --pipeline-use-magsac \
    --benchmark-use-magsac
```

Results are grouped per scene/configuration under `results/colmap_suite/<scene>/<extractor>_<matcher>_<pair_selector>/`
with:
- `sfm/` holding pipeline outputs (`statistics.json`, `statistics.csv`, sparse model, etc.)
- `benchmark/` holding ground-truth metrics (`summary.json`, `results.json`)
- `suite_summary.json` at the root capturing the success status of every run.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{matching_framework_sfm,
  title={Modular Feature Matching Framework with SfM Pipeline},
  author={Enhanced SfM Team},
  year={2025},
  url={https://github.com/...}
}
```
