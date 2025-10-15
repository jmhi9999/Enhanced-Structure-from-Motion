# Gemini Code Guide: Enhanced Structure-from-Motion Pipeline

This document provides a guide for interacting with the "Enhanced Structure-from-Motion" codebase. It summarizes the project's purpose, architecture, and development conventions.

## Project Overview

This is a Python-based, GPU-accelerated Structure-from-Motion (SfM) pipeline designed to produce high-quality camera poses, particularly for use in 3D Gaussian Splatting.

The key innovation is a **Context-Aware Bundle Adjustment (BA)** module that serves as a drop-in replacement for the traditional BA performed by COLMAP. This module analyzes the global structure of the scene to identify and down-weight unreliable cameras (e.g., from blurry images or textureless surfaces) and uncertain 3D points during optimization, leading to more robust and accurate reconstructions.

The core technologies used are Python, PyTorch, OpenCV, and FAISS for GPU-accelerated search. It integrates modern feature extractors like ALIKED and SuperPoint with the LightGlue matcher.

The theoretical foundation and future research directions for the context-aware BA are detailed in `CONTEXT_AWARE_BA_PROPOSAL.md`.

## Building and Running

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jmhi9999/Enhanced-Structure-from-Motion
    cd Enhanced-Structure-from-Motion
    ```

2.  **Install core dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install GPU-accelerated libraries (recommended):**
    ```bash
    pip install cupy-cuda12x faiss-gpu
    ```
    *Note: The COLMAP binary must be installed and available in the system's `PATH`.*

### Running the Pipeline

The main pipeline is executed via `sfm_pipeline.py`.

**Standard (COLMAP) Reconstruction:**
```bash
python sfm_pipeline.py \
    --input_dir path/to/your/images \
    --output_dir path/to/your/output \
    --feature_extractor aliked
```

**Using the Context-Aware Bundle Adjustment (Key Feature):**

To enable the novel context-aware BA, add the `--use_context_ba` flag.
```bash
python sfm_pipeline.py \
    --input_dir path/to/your/images \
    --output_dir path/to/your/output \
    --feature_extractor aliked \
    --use_context_ba
```

The pipeline is also installed as a console script, so you can use `sfm-pipeline` instead of `python sfm_pipeline.py`.

### Running Tests

The project uses `pytest` for testing. Tests are located in the `tests/` directory.

To run all tests:
```bash
pytest
```

## Development Conventions

### Code Structure

-   **`sfm_pipeline.py`**: The main entry point and orchestrator of the entire SfM workflow.
-   **`sfm/core/`**: Contains the core algorithmic components of the pipeline (feature extraction, matching, etc.).
-   **`sfm/core/context_ba/`**: The implementation of the novel Context-Aware Bundle Adjustment. This is the project's primary research area.
    -   **`scene_graph.py`**: Builds the graph representation of the scene used for context analysis.
    -   **`confidence/`**: Contains the logic for calculating confidence scores. `rule_based.py` is the default, training-free method, while `hybrid.py` is an optional learned approach.
    -   **`optimizer.py`**: The core weighted BA solver, which uses `scipy.optimize.least_squares`.
-   **`CONTEXT_AWARE_BA_PROPOSAL.md`**: The technical and research proposal for the context-aware BA. This document should be consulted for understanding the mathematical formulation and future plans.
-   **`tests/`**: Contains unit and integration tests. New features should be accompanied by corresponding tests.

### Contribution Guidelines

-   New functionality should be integrated into `sfm_pipeline.py` and exposed via command-line flags.
-   When modifying the context-aware BA, refer to the principles and formulas outlined in `CONTEXT_AWARE_BA_PROPOSAL.md`.
-   The context-aware BA has two confidence modes: `rule_based` (default) and `hybrid` (optional, requires PyTorch). The hybrid model can be trained using the methods in `sfm/core/context_ba/confidence/hybrid.py`.
-   Always add tests for new code in the `tests/` directory to ensure correctness and prevent regressions.
