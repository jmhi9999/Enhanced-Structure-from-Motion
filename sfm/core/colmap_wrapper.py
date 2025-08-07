"""
Safe COLMAP wrapper that handles CUDA library path issues
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

def setup_cuda_environment():
    """Setup CUDA environment variables before importing pycolmap"""
    # Set CUDA paths for compatibility
    cuda_12_path = '/usr/local/cuda-12.2/lib64'
    
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
    
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if cuda_12_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f'{cuda_12_path}:{current_ld_path}'

def safe_colmap_reconstruction(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any], 
                              output_path: Path, image_dir: Path, device: str = "cpu") -> Tuple[Dict, Dict, Dict]:
    """Safely run COLMAP reconstruction with proper CUDA setup"""
    
    # Setup environment
    setup_cuda_environment()
    
    try:
        # Import COLMAP after environment setup
        from .colmap_reconstruction import COLMAPReconstruction
        
        reconstruction = COLMAPReconstruction(
            output_path=output_path,
            device=device
        )
        
        return reconstruction.reconstruct(features, matches, image_dir)
        
    except ImportError as e:
        if 'libcublas' in str(e) or 'cublasSetEnvironmentMode' in str(e):
            print(f"CUDA library conflict detected: {e}")
            print("Attempting to restart Python process with correct CUDA path...")
            
            # Try subprocess approach as last resort
            import subprocess
            import pickle
            import tempfile
            
            # Save data to temporary files
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                pickle.dump({'features': features, 'matches': matches, 
                           'output_path': str(output_path), 'image_dir': str(image_dir), 
                           'device': device}, f)
                temp_file = f.name
            
            # Create script to run in subprocess
            script = f'''
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import sys
sys.path.insert(0, '/mnt/d/Github/Enhanced-Structure-from-Motion')

import pickle
from pathlib import Path
from sfm.core.colmap_reconstruction import COLMAPReconstruction

# Load data
with open('{temp_file}', 'rb') as f:
    data = pickle.load(f)

reconstruction = COLMAPReconstruction(
    output_path=Path(data['output_path']),
    device=data['device']
)

result = reconstruction.reconstruct(
    data['features'], 
    data['matches'], 
    Path(data['image_dir'])
)

# Save result
with open('{temp_file}_result', 'wb') as f:
    pickle.dump(result, f)
'''
            
            # Run in subprocess
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                script_file = f.name
                
            try:
                result = subprocess.run([sys.executable, script_file], 
                                      capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    # Load result
                    with open(f'{temp_file}_result', 'rb') as f:
                        return pickle.load(f)
                else:
                    print(f"Subprocess failed: {result.stderr}")
                    raise RuntimeError("COLMAP subprocess execution failed")
                    
            finally:
                # Cleanup
                os.unlink(temp_file)
                os.unlink(script_file)
                if os.path.exists(f'{temp_file}_result'):
                    os.unlink(f'{temp_file}_result')
        else:
            raise