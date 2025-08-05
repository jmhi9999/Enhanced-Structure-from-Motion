#!/bin/bash

# Enhanced SfM Installation Script
# Installs all dependencies and sets up the environment

set -e  # Exit on any error

echo "🚀 Installing Enhanced SfM Pipeline..."
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
    echo "✅ CUDA available: $cuda_version"
    CUDA_AVAILABLE=true
else
    echo "⚠️  CUDA not available, will use CPU only"
    CUDA_AVAILABLE=false
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv enhanced_sfm_env
source enhanced_sfm_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "💻 Installing PyTorch (CPU only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Install GPU-specific dependencies if CUDA is available
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Installing GPU acceleration libraries..."
    
    # Install CuPy
    pip install cupy-cuda12x
    
    # Install FAISS GPU
    pip install faiss-gpu
    
    # Install PyCeres (if available)
    pip install pyceres || echo "⚠️  PyCeres not available, using CPU fallback"
fi

# Install additional dependencies for depth estimation
echo "🏔️  Installing depth estimation dependencies..."
pip install transformers timm

# Install development dependencies
echo "🔧 Installing development dependencies..."
pip install pytest black flake8 mypy memory-profiler line-profiler

# Install visualization dependencies
echo "📊 Installing visualization dependencies..."
pip install plotly seaborn

# Install hloc for comparison
echo "🔍 Installing hloc for comparison..."
pip install git+https://github.com/cvg/Hierarchical-Localization.git

# Install LightGlue
echo "🔗 Installing LightGlue..."
pip install git+https://github.com/cvg/LightGlue.git

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p results
mkdir -p logs

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import cv2
import numpy as np
import hloc
from transformers import DPTFeatureExtractor

print('✅ Core dependencies imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Create activation script
echo "📝 Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating Enhanced SfM environment..."
source enhanced_sfm_env/bin/activate
echo "✅ Environment activated!"
echo "Run 'deactivate' to exit the environment"
EOF

chmod +x activate_env.sh

# Create quick start script
echo "📝 Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
source enhanced_sfm_env/bin/activate

echo "🚀 Enhanced SfM Quick Start"
echo "=========================="
echo ""
echo "1. Basic usage:"
echo "   python sfm_pipeline.py --input_dir data/images --output_dir results"
echo ""
echo "2. With GPU acceleration:"
echo "   python sfm_pipeline.py --input_dir data/images --output_dir results --use_gpu_ba --use_vocab_tree"
echo ""
echo "3. Benchmark against hloc:"
echo "   python benchmark_comparison.py --dataset data/images --output benchmark_results"
echo ""
echo "4. Profile performance:"
echo "   python sfm_pipeline.py --input_dir data/images --output_dir results --profile"
echo ""
echo "For more options, run: python sfm_pipeline.py --help"
EOF

chmod +x quick_start.sh

# Create requirements check script
echo "📝 Creating requirements check script..."
cat > check_requirements.py << 'EOF'
#!/usr/bin/env python3
"""
Check if all requirements are properly installed
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name}")
        return False

def main():
    print("🔍 Checking Enhanced SfM requirements...")
    print("=" * 40)
    
    requirements = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("kornia", "Kornia"),
        ("faiss", "FAISS"),
        ("cupy", "CuPy"),
        ("hloc", "hloc"),
        ("transformers", "Transformers"),
        ("timm", "timm"),
        ("plotly", "Plotly"),
        ("seaborn", "Seaborn"),
    ]
    
    all_good = True
    for module, name in requirements:
        if not check_import(module, name):
            all_good = False
    
    print("=" * 40)
    if all_good:
        print("🎉 All requirements installed successfully!")
    else:
        print("⚠️  Some requirements are missing. Run: pip install -r requirements.txt")
    
    return all_good

if __name__ == "__main__":
    main()
EOF

chmod +x check_requirements.py

echo ""
echo "🎉 Installation completed successfully!"
echo "======================================"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment: source activate_env.sh"
echo "2. Check requirements: python check_requirements.py"
echo "3. Quick start: ./quick_start.sh"
echo ""
echo "📁 Directory structure:"
echo "├── data/           # Put your images here"
echo "├── results/        # Output will be saved here"
echo "├── logs/           # Log files"
echo "└── enhanced_sfm_env/  # Virtual environment"
echo ""
echo "🚀 Ready to use Enhanced SfM!"
echo ""
echo "Example usage:"
echo "source activate_env.sh"
echo "python sfm_pipeline.py --input_dir data/images --output_dir results --use_gpu_ba" 