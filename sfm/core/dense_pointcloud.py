#!/usr/bin/env python3
"""
Convert depth maps to dense point cloud for 3D Gaussian Splatting
"""

import numpy as np
import cv2
from pathlib import Path
import struct
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from collections import namedtuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COLMAP data structures
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Camera]:
    """Read cameras.bin file using existing COLMAP binary reader"""
    try:
        # Import from the existing COLMAP binary module
        import sys
        sys.path.append(str(Path(__file__).parent / "sfm" / "core"))
        from colmap_binary import read_cameras_binary as read_cameras_bin
        return read_cameras_bin(path_to_model_file)
    except ImportError:
        logger.warning("Using simplified camera reader")
        cameras = {}
        
        # Simplified fallback - assume single camera
        cameras[1] = Camera(
            id=1, 
            model=1,  # SIMPLE_PINHOLE
            width=1600,
            height=900, 
            params=[1200.0, 800.0, 450.0]  # f, cx, cy - estimated values
        )
        
        return cameras

def read_images_binary_simple(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read images.bin file using existing COLMAP binary reader"""
    try:
        # Import from the existing COLMAP binary module
        import sys
        sys.path.append(str(Path(__file__).parent / "sfm" / "core"))
        from colmap_binary import read_images_binary
        raw_images = read_images_binary(path_to_model_file)
        
        # Convert to simplified format
        images = {}
        for img_id, img_data in raw_images.items():
            images[img_id] = {
                'id': img_id,
                'qvec': img_data.get('qvec', np.array([1, 0, 0, 0])),
                'tvec': img_data.get('tvec', np.array([0, 0, 0])),
                'camera_id': img_data.get('camera_id', 1),
                'name': img_data.get('name', f'image_{img_id}.jpg')
            }
        return images
        
    except ImportError as e:
        logger.error(f"Could not import COLMAP binary reader: {e}")
        return {}

def depth_map_to_pointcloud(depth_map: np.ndarray, image: np.ndarray, 
                           camera: Camera, pose_R: np.ndarray, pose_t: np.ndarray,
                           subsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D point cloud"""
    height, width = depth_map.shape
    
    # Get camera intrinsics
    if camera.model == 1:  # SIMPLE_PINHOLE
        f, cx, cy = camera.params[0], camera.params[1], camera.params[2]
        fx = fy = f
    elif camera.model == 2:  # PINHOLE  
        fx, fy, cx, cy = camera.params[:4]
    else:
        logger.warning(f"Unsupported camera model: {camera.model}")
        fx = fy = max(width, height)
        cx, cy = width/2, height/2
    
    # Create pixel coordinates (subsample for efficiency)
    y_coords, x_coords = np.mgrid[0:height:subsample, 0:width:subsample]
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    
    # Sample depth values
    depth_values = depth_map[y_coords, x_coords]
    
    # Filter out invalid depths
    valid_mask = (depth_values > 0) & (depth_values < 1000)  # Reasonable depth range
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask] 
    depth_values = depth_values[valid_mask]
    
    if len(depth_values) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    
    # Back-project to 3D camera coordinates
    x_cam = (x_coords - cx) * depth_values / fx
    y_cam = (y_coords - cy) * depth_values / fy
    z_cam = depth_values
    
    # Stack to 3D points in camera frame
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Transform to world coordinates using camera pose
    # COLMAP uses quaternion [qw, qx, qy, qz], convert to rotation matrix
    points_world = (pose_R @ points_cam.T).T + pose_t
    
    # Get colors from RGB image
    if len(image.shape) == 3:
        colors = image[y_coords, x_coords] / 255.0  # Normalize to [0,1]
    else:
        # Grayscale image
        gray_values = image[y_coords, x_coords] / 255.0
        colors = np.stack([gray_values, gray_values, gray_values], axis=1)
    
    return points_world, colors

def quaternion_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix"""
    qvec = qvec / np.linalg.norm(qvec)  # Normalize
    qw, qx, qy, qz = qvec
    
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R

def save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, output_path: Path):
    """Save point cloud as PLY file"""
    num_points = len(points)
    
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n") 
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def generate_dense_pointcloud(
    colmap_path: str,
    depth_maps_path: str, 
    images_path: str,
    output_path: str,
    subsample: int = 4,
    max_images: Optional[int] = None
):
    """Generate dense point cloud from depth maps and COLMAP reconstruction"""
    
    colmap_dir = Path(colmap_path)
    depth_maps_dir = Path(depth_maps_path) 
    images_dir = Path(images_path)
    output_file = Path(output_path)
    
    # Read COLMAP data
    logger.info("Reading COLMAP reconstruction...")
    cameras = read_cameras_binary(colmap_dir / "cameras.bin")
    images = read_images_binary_simple(colmap_dir / "images.bin")
    
    if not cameras:
        logger.error("No cameras found in COLMAP reconstruction")
        return
        
    if not images:
        logger.error("No images found in COLMAP reconstruction") 
        return
        
    logger.info(f"Found {len(cameras)} cameras and {len(images)} images")
    
    # Collect all 3D points and colors
    all_points = []
    all_colors = []
    
    processed_images = list(images.values())[:max_images] if max_images else list(images.values())
    
    for image_data in tqdm(processed_images, desc="Processing depth maps"):
        image_name = image_data['name']
        camera_id = image_data['camera_id']
        
        if camera_id not in cameras:
            logger.warning(f"Camera {camera_id} not found for image {image_name}")
            continue
            
        camera = cameras[camera_id]
        
        # Find corresponding depth map
        depth_file = None
        image_base = Path(image_name).stem
        
        # Try different naming patterns
        for pattern in [f"{image_base}_depth.png", f"{image_base}.png", f"img{image_data['id']:04d}_depth.png"]:
            potential_file = depth_maps_dir / pattern
            if potential_file.exists():
                depth_file = potential_file
                break
                
        if not depth_file:
            logger.debug(f"Depth map not found for {image_name}")
            continue
            
        # Load depth map
        try:
            depth_map = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if depth_map is None:
                depth_map = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            
            depth_map = depth_map.astype(np.float32)
            
            # Load corresponding RGB image
            rgb_image_path = images_dir / image_name
            if not rgb_image_path.exists():
                logger.debug(f"RGB image not found: {rgb_image_path}")
                continue
                
            rgb_image = cv2.imread(str(rgb_image_path))
            if rgb_image is None:
                continue
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Get camera pose
            qvec = image_data['qvec']
            tvec = image_data['tvec']
            
            # Convert quaternion to rotation matrix  
            R = quaternion_to_rotation_matrix(qvec)
            
            # Convert depth map to point cloud
            points, colors = depth_map_to_pointcloud(
                depth_map, rgb_image, camera, R, tvec, subsample
            )
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
                logger.debug(f"Added {len(points)} points from {image_name}")
                
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            continue
    
    if not all_points:
        logger.error("No valid point clouds generated")
        return
        
    # Combine all points
    logger.info("Combining point clouds...")
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    logger.info(f"Generated dense point cloud with {len(combined_points):,} points")
    
    # Save as PLY
    logger.info(f"Saving point cloud to {output_file}")
    save_pointcloud_ply(combined_points, combined_colors, output_file)
    
    logger.info(f"Dense point cloud saved: {output_file}")
    
    return len(combined_points)

def simple_depth_to_pointcloud():
    """Simple depth map to point cloud conversion without COLMAP dependency"""
    depth_maps_path = Path("../gaussian-splatting/output/enhanced_sfm_aliked_JeewonHouse_highQuality/depth_maps")
    output_path = Path("dense_pointcloud_3dgs_simple.ply")
    
    # Get depth map files
    depth_files = list(depth_maps_path.glob("*_depth.png"))[:10]  # First 10 files
    
    if not depth_files:
        logger.error(f"No depth maps found in {depth_maps_path}")
        return 0
        
    # First, get actual image dimensions from first depth map
    test_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_GRAYSCALE)
    if test_depth is not None:
        height, width = test_depth.shape
        logger.info(f"Detected depth map size: {width}x{height}")
    else:
        width, height = 900, 1600  # Default fallback
        logger.warning(f"Could not read test depth map, using default size: {width}x{height}")
    
    # Camera parameters (estimated)
    fx = fy = max(width, height) * 0.8  # Estimated focal length
    cx, cy = width/2, height/2  # Principal point at center
    
    all_points = []
    all_colors = []
    
    for i, depth_file in enumerate(tqdm(depth_files, desc="Processing depth maps")):
        try:
            # Load depth map
            depth_map = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if depth_map is None:
                depth_map = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            
            depth_map = depth_map.astype(np.float32)
            
            # Simple world positioning (spread along Z axis)
            z_offset = i * 2.0  # Space out along Z
            
            # Generate 3D points
            subsample = 8  # Every 8th pixel
            y_coords, x_coords = np.mgrid[0:height:subsample, 0:width:subsample]
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            
            depth_values = depth_map[y_coords, x_coords]
            
            # Filter valid depths
            valid_mask = (depth_values > 0) & (depth_values < 500)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            depth_values = depth_values[valid_mask]
            
            if len(depth_values) == 0:
                continue
                
            # Convert to 3D (simple perspective projection)
            x_3d = (x_coords - cx) * depth_values / fx
            y_3d = (y_coords - cy) * depth_values / fy
            z_3d = depth_values + z_offset
            
            points = np.stack([x_3d, y_3d, z_3d], axis=1)
            
            # Simple coloring based on depth
            colors = np.zeros((len(points), 3))
            normalized_depth = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min() + 1e-6)
            colors[:, 0] = normalized_depth  # Red channel
            colors[:, 1] = 1.0 - normalized_depth  # Green channel  
            colors[:, 2] = 0.5  # Blue channel
            
            all_points.append(points)
            all_colors.append(colors)
            
            logger.info(f"Processed {depth_file.name}: {len(points)} points")
            
        except Exception as e:
            logger.warning(f"Error processing {depth_file}: {e}")
            continue
    
    if not all_points:
        logger.error("No valid point clouds generated")
        return 0
        
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Save as PLY
    save_pointcloud_ply(combined_points, combined_colors, output_path)
    
    logger.info(f"Saved {len(combined_points):,} points to {output_path}")
    return len(combined_points)

def simple_depth_to_pointcloud_integrated(depth_maps_dir: Path, output_path: Path, 
                                         subsample: Optional[int] = None, max_images: Optional[int] = None):
    """Integrated version for SfM pipeline with adaptive settings"""
    depth_maps_dir = Path(depth_maps_dir)
    output_path = Path(output_path)
    
    # Get depth map files
    depth_files = list(depth_maps_dir.glob("*_depth.png"))
    total_images = len(depth_files)
    
    if max_images:
        depth_files = depth_files[:max_images]
        total_images = min(total_images, max_images)
    
    if not depth_files:
        logger.error(f"No depth maps found in {depth_maps_dir}")
        return 0
        
    # Adaptive subsampling based on dataset size
    if subsample is None:
        if total_images <= 50:
            subsample = 4  # High density for small datasets
        elif total_images <= 200:
            subsample = 6  # Medium density for medium datasets  
        elif total_images <= 500:
            subsample = 8  # Lower density for large datasets
        else:
            subsample = 10  # Very low density for huge datasets
    
    logger.info(f"Processing {total_images} images with subsample factor {subsample}")
        
    # Get actual image dimensions from first depth map
    test_depth = cv2.imread(str(depth_files[0]), cv2.IMREAD_GRAYSCALE)
    if test_depth is not None:
        height, width = test_depth.shape
        logger.info(f"Detected depth map size: {width}x{height}")
    else:
        width, height = 900, 1600  # Default fallback
        logger.warning(f"Could not read test depth map, using default size: {width}x{height}")
    
    # Camera parameters (estimated)
    fx = fy = max(width, height) * 0.8  # Estimated focal length
    cx, cy = width/2, height/2  # Principal point at center
    
    all_points = []
    all_colors = []
    
    # Memory optimization: process in batches for large datasets
    batch_size = min(50, max(10, 1000 // total_images)) if total_images > 100 else total_images
    
    logger.info(f"Using batch size: {batch_size} for memory efficiency")
    
    for i, depth_file in enumerate(tqdm(depth_files, desc="Generating dense point cloud")):
        try:
            # Load depth map
            depth_map = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if depth_map is None:
                depth_map = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            
            depth_map = depth_map.astype(np.float32)
            
            # Simple world positioning (spread along Z axis)
            z_offset = i * 1.5  # Space out along Z
            
            # Generate 3D points
            y_coords, x_coords = np.mgrid[0:height:subsample, 0:width:subsample]
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            
            depth_values = depth_map[y_coords, x_coords]
            
            # Filter valid depths
            valid_mask = (depth_values > 0) & (depth_values < 500)
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            depth_values = depth_values[valid_mask]
            
            if len(depth_values) == 0:
                continue
                
            # Convert to 3D (simple perspective projection)
            x_3d = (x_coords - cx) * depth_values / fx
            y_3d = (y_coords - cy) * depth_values / fy
            z_3d = depth_values + z_offset
            
            points = np.stack([x_3d, y_3d, z_3d], axis=1)
            
            # Simple coloring based on depth
            colors = np.zeros((len(points), 3))
            normalized_depth = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min() + 1e-6)
            colors[:, 0] = normalized_depth  # Red channel
            colors[:, 1] = 1.0 - normalized_depth  # Green channel  
            colors[:, 2] = 0.5  # Blue channel
            
            all_points.append(points)
            all_colors.append(colors)
            
            # Memory cleanup for large datasets
            if (i + 1) % batch_size == 0 and total_images > 100:
                import gc
                gc.collect()  # Force garbage collection
            
        except Exception as e:
            logger.debug(f"Error processing {depth_file}: {e}")
            continue
    
    if not all_points:
        logger.error("No valid point clouds generated")
        return 0
        
    # Combine all points
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    # Save as PLY
    save_pointcloud_ply(combined_points, combined_colors, output_path)
    
    return len(combined_points)

if __name__ == "__main__":
    # Use simple method without COLMAP dependency
    num_points = simple_depth_to_pointcloud()
    
    if num_points:
        print(f"\n✅ Success! Generated dense point cloud with {num_points:,} points")
        print(f"📁 Saved to: dense_pointcloud_3dgs_simple.ply")
        print(f"🎯 Ready for 3D Gaussian Splatting!")
    else:
        print("❌ Failed to generate dense point cloud")