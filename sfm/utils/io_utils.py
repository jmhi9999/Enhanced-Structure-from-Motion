"""
I/O utilities for saving SfM results in COLMAP format
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
import pickle


def save_colmap_format(output_dir: str, cameras: Dict, images: Dict, 
                      points3d: Dict, dense_depth_maps: Dict[str, np.ndarray] = None):
    """Save SfM results in COLMAP format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cameras.bin
    save_cameras_bin(output_path / "cameras.bin", cameras)
    
    # Save images.bin
    save_images_bin(output_path / "images.bin", images)
    
    # Save points3D.bin
    save_points3d_bin(output_path / "points3D.bin", points3d)
    
    # Save dense depth maps if provided
    if dense_depth_maps:
        save_dense_depth_maps(output_path / "depth_maps", dense_depth_maps)
    
    # Save reconstruction info
    save_reconstruction_info(output_path / "reconstruction_info.json", 
                           cameras, images, points3d)
    
    print(f"SfM results saved to {output_path}")


def save_cameras_bin(filepath: Path, cameras: Dict):
    """Save cameras in COLMAP binary format"""
    with open(filepath, 'wb') as f:
        # Write number of cameras
        num_cameras = len(cameras)
        f.write(num_cameras.to_bytes(8, byteorder='little'))
        
        for camera_id, camera in cameras.items():
            # Write camera ID
            f.write(camera_id.to_bytes(4, byteorder='little'))
            
            # Write model name
            model_name = camera['model'].encode('utf-8')
            f.write(len(model_name).to_bytes(8, byteorder='little'))
            f.write(model_name)
            
            # Write width and height
            f.write(camera['width'].to_bytes(8, byteorder='little'))
            f.write(camera['height'].to_bytes(8, byteorder='little'))
            
            # Write parameters
            params = camera['params']
            f.write(len(params).to_bytes(8, byteorder='little'))
            for param in params:
                f.write(struct.pack('d', param))


def save_images_bin(filepath: Path, images: Dict):
    """Save images in COLMAP binary format"""
    with open(filepath, 'wb') as f:
        # Write number of images
        num_images = len(images)
        f.write(num_images.to_bytes(8, byteorder='little'))
        
        for image_path, image in images.items():
            # Write image ID (use hash of path as ID)
            image_id = hash(image_path) % (2**31)  # 32-bit positive integer
            f.write(image_id.to_bytes(4, byteorder='little'))
            
            # Write camera ID
            f.write(image['camera_id'].to_bytes(4, byteorder='little'))
            
            # Write image name
            image_name = image['name'].encode('utf-8')
            f.write(len(image_name).to_bytes(8, byteorder='little'))
            f.write(image_name)
            
            # Write quaternion (rotation)
            qvec = image['qvec']
            for q in qvec:
                f.write(struct.pack('d', q))
            
            # Write translation vector
            tvec = image['tvec']
            for t in tvec:
                f.write(struct.pack('d', t))
            
            # Write number of points
            num_points = len(image['xys'])
            f.write(num_points.to_bytes(8, byteorder='little'))
            
            # Write 2D points
            xys = image['xys']
            for xy in xys:
                f.write(struct.pack('dd', xy[0], xy[1]))
            
            # Write point3D IDs
            point3d_ids = image['point3D_ids']
            for point3d_id in point3d_ids:
                f.write(point3d_id.to_bytes(4, byteorder='little'))


def save_points3d_bin(filepath: Path, points3d: Dict):
    """Save 3D points in COLMAP binary format"""
    with open(filepath, 'wb') as f:
        # Write number of points
        num_points = len(points3d)
        f.write(num_points.to_bytes(8, byteorder='little'))
        
        for point_id, point in points3d.items():
            # Write point ID
            f.write(point_id.to_bytes(8, byteorder='little'))
            
            # Write XYZ coordinates
            xyz = point['xyz']
            for coord in xyz:
                f.write(struct.pack('d', coord))
            
            # Write RGB color
            rgb = point['rgb']
            for color in rgb:
                f.write(color.to_bytes(1, byteorder='little'))
            
            # Write error
            f.write(struct.pack('d', point['error']))
            
            # Write track length
            track = point['track']
            f.write(len(track).to_bytes(8, byteorder='little'))
            
            # Write track elements
            for image_id, point2d_id in track:
                f.write(image_id.to_bytes(4, byteorder='little'))
                f.write(point2d_id.to_bytes(4, byteorder='little'))


def save_dense_depth_maps(output_dir: Path, depth_maps: Dict[str, np.ndarray]):
    """Save dense depth maps"""
    output_dir.mkdir(exist_ok=True)
    
    for img_path, depth_map in depth_maps.items():
        # Get image name
        img_name = Path(img_path).stem
        
        # Save as numpy array
        np.save(output_dir / f"{img_name}_depth.npy", depth_map)
        
        # Save as normalized image for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        import cv2
        cv2.imwrite(str(output_dir / f"{img_name}_depth.png"), depth_uint8)


def save_reconstruction_info(filepath: Path, cameras: Dict, images: Dict, points3d: Dict):
    """Save reconstruction information in JSON format"""
    info = {
        'num_cameras': len(cameras),
        'num_images': len(images),
        'num_points3d': len(points3d),
        'cameras': cameras,
        'images': {str(k): v for k, v in images.items()},
        'points3d': points3d
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
    
    with h5py.File(filepath, 'w') as f:
        for img_path, feat_data in features.items():
            grp = f.create_group(str(img_path))
            grp.create_dataset('keypoints', data=feat_data['keypoints'])
            grp.create_dataset('descriptors', data=feat_data['descriptors'])
            grp.create_dataset('scores', data=feat_data['scores'])
            grp.attrs['image_shape'] = feat_data['image_shape']


def load_features(filepath: Path) -> Dict[str, Any]:
    """Load features from H5 format"""
    import h5py
    
    features = {}
    with h5py.File(filepath, 'r') as f:
        for img_path in f.keys():
            grp = f[img_path]
            features[img_path] = {
                'keypoints': grp['keypoints'][:],
                'descriptors': grp['descriptors'][:],
                'scores': grp['scores'][:],
                'image_shape': tuple(grp.attrs['image_shape'])
            }
    
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