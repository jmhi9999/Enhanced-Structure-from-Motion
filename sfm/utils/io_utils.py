"""
I/O utilities for saving SfM results in COLMAP format
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import json
import shutil
import struct


def save_colmap_format(
    cameras: Dict,
    images: Dict,
    points3d: Dict,
    output_dir: str,
    source_sparse_dir: Optional[Path] = None,
) -> None:
    """Prepare COLMAP-format outputs for downstream tools (e.g., 3DGS).

    We copy the original COLMAP binary/text files when available to guarantee
    compatibility and still emit a JSON summary for quick inspection.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_sparse_path = Path(source_sparse_dir) if source_sparse_dir else None
    if source_sparse_path and source_sparse_path.exists():
        for filename in [
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
            "project.ini",
        ]:
            src_file = source_sparse_path / filename
            if src_file.exists():
                shutil.copy2(src_file, output_path / filename)

    save_reconstruction_info(
        output_path / "reconstruction_info.json",
        cameras,
        images,
        points3d,
    )


def save_reconstruction_info(filepath: Path, cameras: Dict, images: Dict, points3d: Dict):
    """Save reconstruction information in JSON format"""
    
    def convert_to_json_serializable(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    
    info = {
        'num_cameras': len(cameras),
        'num_images': len(images),
        'num_points3d': len(points3d),
        'cameras': convert_to_json_serializable(cameras),
        'images': {str(k): convert_to_json_serializable(v) for k, v in images.items()},
        'points3d': convert_to_json_serializable(points3d)
    }
    
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)


def load_colmap_format(input_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load SfM results from COLMAP format"""
    input_path = Path(input_dir)
    
    # Load cameras
    cameras = load_cameras_bin(input_path / "cameras.bin")
    
    # Load images
    images = load_images_bin(input_path / "images.bin")
    
    # Load 3D points
    points3d = load_points3d_bin(input_path / "points3D.bin")
    
    return cameras, images, points3d


def load_cameras_bin(filepath: Path) -> Dict:
    """Load cameras from COLMAP binary format"""
    cameras = {}
    
    with open(filepath, 'rb') as f:
        # Read number of cameras
        num_cameras = int.from_bytes(f.read(8), byteorder='little')
        
        for _ in range(num_cameras):
            # Read camera ID
            camera_id = int.from_bytes(f.read(4), byteorder='little')
            
            # Read model name
            model_name_len = int.from_bytes(f.read(8), byteorder='little')
            model_name = f.read(model_name_len).decode('utf-8')
            
            # Read width and height
            width = int.from_bytes(f.read(8), byteorder='little')
            height = int.from_bytes(f.read(8), byteorder='little')
            
            # Read parameters
            params_len = int.from_bytes(f.read(8), byteorder='little')
            params = []
            for _ in range(params_len):
                params.append(struct.unpack('d', f.read(8))[0])
            
            cameras[camera_id] = {
                'model': model_name,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def load_images_bin(filepath: Path) -> Dict:
    """Load images from COLMAP binary format"""
    images = {}
    
    with open(filepath, 'rb') as f:
        # Read number of images
        num_images = int.from_bytes(f.read(8), byteorder='little')
        
        for _ in range(num_images):
            # Read image ID
            image_id = int.from_bytes(f.read(4), byteorder='little')
            
            # Read camera ID
            camera_id = int.from_bytes(f.read(4), byteorder='little')
            
            # Read image name
            image_name_len = int.from_bytes(f.read(8), byteorder='little')
            image_name = f.read(image_name_len).decode('utf-8')
            
            # Read quaternion
            qvec = []
            for _ in range(4):
                qvec.append(struct.unpack('d', f.read(8))[0])
            
            # Read translation vector
            tvec = []
            for _ in range(3):
                tvec.append(struct.unpack('d', f.read(8))[0])
            
            # Read number of points
            num_points = int.from_bytes(f.read(8), byteorder='little')
            
            # Read 2D points
            xys = []
            for _ in range(num_points):
                x, y = struct.unpack('dd', f.read(16))
                xys.append([x, y])
            
            # Read point3D IDs
            point3d_ids = []
            for _ in range(num_points):
                point3d_id = int.from_bytes(f.read(4), byteorder='little')
                point3d_ids.append(point3d_id)
            
            images[image_name] = {
                'camera_id': camera_id,
                'qvec': qvec,
                'tvec': tvec,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3d_ids
            }
    
    return images


def load_points3d_bin(filepath: Path) -> Dict:
    """Load 3D points from COLMAP binary format"""
    points3d = {}
    
    with open(filepath, 'rb') as f:
        # Read number of points
        num_points = int.from_bytes(f.read(8), byteorder='little')
        
        for _ in range(num_points):
            # Read point ID
            point_id = int.from_bytes(f.read(8), byteorder='little')
            
            # Read XYZ coordinates
            xyz = []
            for _ in range(3):
                xyz.append(struct.unpack('d', f.read(8))[0])
            
            # Read RGB color
            rgb = []
            for _ in range(3):
                rgb.append(int.from_bytes(f.read(1), byteorder='little'))
            
            # Read error
            error = struct.unpack('d', f.read(8))[0]
            
            # Read track length
            track_len = int.from_bytes(f.read(8), byteorder='little')
            
            # Read track elements
            track = []
            for _ in range(track_len):
                image_id = int.from_bytes(f.read(4), byteorder='little')
                point2d_id = int.from_bytes(f.read(4), byteorder='little')
                track.append((image_id, point2d_id))
            
            points3d[point_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': track
            }
    
    return points3d


# Import struct for binary operations
import struct


def load_images(input_dir: str) -> List[str]:
    """Load image paths from directory"""
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_path.glob(f"*{ext}"))
        image_paths.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted([str(p) for p in image_paths])


def save_features(features: Dict[str, Any], filepath: Path):
    """Save features in H5 format"""
    import h5py
    import numbers
    
    with h5py.File(filepath, 'w') as f:
        for img_path, feat_data in features.items():
            grp = f.create_group(str(img_path))
            for key, value in feat_data.items():
                if key == 'image_shape':
                    grp.attrs['image_shape'] = value
                elif isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                elif isinstance(value, (numbers.Number, str)):
                    grp.attrs[key] = value
                # Skip non-serializable runtime helpers silently


def load_features(filepath: Path) -> Dict[str, Any]:
    """Load features from H5 format"""
    import h5py
    
    features = {}
    with h5py.File(filepath, 'r') as f:
        for img_path in f.keys():
            grp = f[img_path]
            features[img_path] = {
                key: grp[key][:] for key in grp.keys()
            }
            if 'image_shape' in grp.attrs:
                features[img_path]['image_shape'] = tuple(grp.attrs['image_shape'])
            for attr_key, attr_val in grp.attrs.items():
                if attr_key != 'image_shape':
                    features[img_path][attr_key] = attr_val
    
    return features


def save_matches(matches: Dict[Tuple[str, str], Any], filepath: Path):
    """Save matches in H5 format"""
    import h5py
    
    with h5py.File(filepath, 'w') as f:
        for i, (pair, match_data) in enumerate(matches.items()):
            grp = f.create_group(f'match_{i}')
            grp.attrs['img1'] = pair[0]
            grp.attrs['img2'] = pair[1]
            
            for key, value in match_data.items():
                if isinstance(value, np.ndarray):
                    grp.create_dataset(key, data=value)
                else:
                    grp.attrs[key] = value


def load_matches(filepath: Path) -> Dict[Tuple[str, str], Any]:
    """Load matches from H5 format"""
    import h5py
    
    matches = {}
    with h5py.File(filepath, 'r') as f:
        for group_name in f.keys():
            grp = f[group_name]
            img1 = grp.attrs['img1']
            img2 = grp.attrs['img2']
            
            match_data = {}
            for key in grp.keys():
                match_data[key] = grp[key][:]
            
            for attr_key in grp.attrs.keys():
                if attr_key not in ['img1', 'img2']:
                    match_data[attr_key] = grp.attrs[attr_key]
            
            matches[(img1, img2)] = match_data
    
    return matches 
