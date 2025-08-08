"""
Direct COLMAP binary execution without pycolmap to avoid CUDA conflicts
"""

import os
import subprocess
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
import cv2
from collections import namedtuple

logger = logging.getLogger(__name__)

# COLMAP data structures
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def filter_matches_with_magsac(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any]) -> Dict[Tuple[str, str], Any]:
    """Filter matches using cv2.USAC_MAGSAC geometric verification"""
    logger.info("Filtering matches with cv2 USAC_MAGSAC...")
    
    filtered_matches = {}
    
    for (img1_path, img2_path), match_data in tqdm(matches.items(), desc="MAGSAC filtering"):
        try:
            # Get keypoints for both images
            kpts1 = features[img1_path]['keypoints']
            kpts2 = features[img2_path]['keypoints']
            
            # Get matched keypoint indices
            matches0 = match_data['matches0']
            matches1 = match_data['matches1']
            
            # Filter out invalid matches (-1)
            valid_mask = (matches0 >= 0) & (matches1 >= 0)
            if not valid_mask.any():
                continue
                
            valid_matches0 = matches0[valid_mask]
            valid_matches1 = matches1[valid_mask]
            
            # Get matched keypoints
            matched_kpts1 = kpts1[valid_matches0]
            matched_kpts2 = kpts2[valid_matches1]
            
            if len(matched_kpts1) < 8:  # Need at least 8 points for fundamental matrix
                continue
            
            # Run MAGSAC
            F_matrix, inlier_mask = cv2.findFundamentalMat(
                matched_kpts1.astype(np.float32),
                matched_kpts2.astype(np.float32),
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
                confidence=0.999,
                maxIters=10000
            )
            
            if F_matrix is None or inlier_mask is None:
                continue
            
            # Keep only inlier matches
            inlier_mask = inlier_mask.ravel().astype(bool)
            if inlier_mask.sum() < 4:  # Need minimum matches
                continue
            
            # Update match data with filtered matches
            filtered_matches0 = valid_matches0[inlier_mask]
            filtered_matches1 = valid_matches1[inlier_mask]
            
            # Create new match data
            filtered_matches[(img1_path, img2_path)] = {
                'matches0': filtered_matches0,
                'matches1': filtered_matches1,
                'mscores0': match_data['mscores0'][valid_mask][inlier_mask] if 'mscores0' in match_data else np.ones(len(filtered_matches0)),
                'mscores1': match_data['mscores1'][valid_mask][inlier_mask] if 'mscores1' in match_data else np.ones(len(filtered_matches1))
            }
            
            logger.debug(f"Match {Path(img1_path).name} - {Path(img2_path).name}: {len(valid_matches0)} -> {len(filtered_matches0)} matches")
            
        except Exception as e:
            #logger.debug(f"MAGSAC filtering failed for {Path(img1_path).name} - {Path(img2_path).name}: {e}")
            continue
    
    logger.info(f"MAGSAC filtering: {len(matches)} -> {len(filtered_matches)} pairs")
    return filtered_matches


def create_colmap_database(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any], 
                          database_path: Path) -> Dict[str, int]:
    """Create COLMAP database with features and matches"""
    logger.info("Creating COLMAP database...")
    
    # Remove existing database
    if database_path.exists():
        database_path.unlink()
    
    # Create database connection
    conn = sqlite3.connect(database_path)
    
    # Create COLMAP tables
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL);
            
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL);
            
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
            
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB);
            
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB,
            qvec BLOB,
            tvec BLOB);
    ''')
    
    # Add camera (assume single camera)
    first_feature = next(iter(features.values()))
    height, width = first_feature['image_shape']
    focal_length = 1.2 * max(width, height)  # hloc heuristic
    
    camera_params = np.array([focal_length, width / 2.0, height / 2.0], dtype=np.float64)
    params_blob = camera_params.tobytes()
    
    conn.execute(
        "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
        (1, 0, width, height, params_blob, 1)  # model 0 = SIMPLE_PINHOLE
    )
    
    # Add images and get image IDs
    image_ids = {}
    for i, img_path in enumerate(features.keys(), 1):
        image_name = Path(img_path).name
        conn.execute(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            (i, image_name, 1)
        )
        image_ids[image_name] = i
    
    # Add keypoints
    logger.info("Adding keypoints to database...")
    for img_path, feat_data in tqdm(features.items(), desc="Adding keypoints"):
        image_name = Path(img_path).name
        image_id = image_ids[image_name]
        keypoints = feat_data['keypoints'] + 0.5  # COLMAP origin
        keypoints = keypoints.astype(np.float32)
        keypoints_blob = keypoints.tobytes()
        
        conn.execute(
            "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob)
        )
    
    # Add matches
    logger.info("Adding matches to database...")
    for (img1_path, img2_path), match_data in tqdm(matches.items(), desc="Adding matches"):
        name0 = Path(img1_path).name
        name1 = Path(img2_path).name
        
        if name0 in image_ids and name1 in image_ids:
            id0, id1 = image_ids[name0], image_ids[name1]
            
            # Create pair ID
            if id0 > id1:
                id0, id1 = id1, id0
            pair_id = id0 * 2147483647 + id1
            
            # Extract matches
            matches_array = np.column_stack([
                match_data['matches0'],
                match_data['matches1']
            ]).astype(np.uint32)
            
            matches_blob = matches_array.tobytes()
            
            conn.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (pair_id, matches_array.shape[0], matches_array.shape[1], matches_blob)
            )
            
            # Also add to two_view_geometries (required for COLMAP incremental mapping)
            # Use dummy geometry for now - COLMAP will verify
            dummy_F = np.eye(3, dtype=np.float64).tobytes()
            dummy_E = np.eye(3, dtype=np.float64).tobytes()
            dummy_H = np.eye(3, dtype=np.float64).tobytes()
            dummy_qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64).tobytes()
            dummy_tvec = np.array([0.0, 0.0, 0.0], dtype=np.float64).tobytes()
            
            conn.execute(
                "INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair_id, matches_array.shape[0], matches_array.shape[1], matches_blob, 2, dummy_F, dummy_E, dummy_H, dummy_qvec, dummy_tvec)
            )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database created with {len(image_ids)} images and {len(matches)} match pairs")
    
    # Verify database contents
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    match_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM keypoints")
    keypoint_count = cursor.fetchone()[0]
    
    conn.close()
    
    logger.info(f"Database verification: {image_count} images, {match_count} matches, {keypoint_count} keypoints")
    
    return image_ids


def run_colmap_binary(database_path: Path, image_dir: Path, output_path: Path) -> bool:
    """Run COLMAP using binary executable"""
    
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True)
    
    # Step 2: Incremental mapping (hloc-style minimal options)
    logger.info("Running COLMAP incremental mapping...")
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_path),
        "--Mapper.num_threads", "16"
    ]
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        logger.info(f"COLMAP mapper return code: {result.returncode}")
        logger.info(f"COLMAP stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"COLMAP stderr: {result.stderr}")
            
        if result.returncode == 0:
            logger.info("COLMAP incremental mapping completed successfully")
            return True
        else:
            logger.error(f"COLMAP mapping failed with return code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("COLMAP mapping timed out")
        return False
    except Exception as e:
        logger.error(f"COLMAP mapping error: {e}")
        return False


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read cameras.bin file"""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = struct.unpack("<Q", fid.read(8))[0]
            params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
            cameras[camera_id] = {
                'model_id': model_id,
                'model': 'PINHOLE',  # Simplified
                'width': width,
                'height': height,
                'params': list(params)
            }
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read images.bin file"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        
        # Add bounds checking to prevent overflow
        if num_reg_images > 100000:  # Reasonable upper limit
            logger.warning(f"Unusually large number of registered images: {num_reg_images}, truncating to 100000")
            num_reg_images = 100000
        
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<iidddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            # Read image name
            image_name = ""
            current_char = struct.unpack("<c", fid.read(1))[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("<c", fid.read(1))[0]
            
            # Read 2D points with overflow protection
            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            
            # Add bounds checking to prevent overflow
            if num_points2D > 1000000:  # Reasonable upper limit
                logger.warning(f"Unusually large number of 2D points ({num_points2D}) for image {image_name}, truncating to 1000000")
                num_points2D = 1000000
            
            if num_points2D > 0:
                try:
                    # Use int64 to prevent overflow in struct calculations
                    total_elements = int(num_points2D) * 3
                    total_bytes = int(num_points2D) * 24
                    
                    x_y_id_s = struct.unpack(f"<{total_elements}d", fid.read(total_bytes))
                    xys = np.column_stack([np.array(x_y_id_s[0::3]), np.array(x_y_id_s[1::3])])
                    point3D_ids = np.array(x_y_id_s[2::3], dtype=np.int64)
                except (struct.error, OverflowError, MemoryError) as e:
                    logger.warning(f"Error reading 2D points for image {image_name}: {e}, skipping")
                    xys = np.array([]).reshape(0, 2)
                    point3D_ids = np.array([], dtype=np.int64)
            else:
                xys = np.array([]).reshape(0, 2)
                point3D_ids = np.array([], dtype=np.int64)
            
            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            }
    return images


def read_points3d_binary(path_to_model_file: Path) -> Dict[int, Dict]:
    """Read points3D.bin file"""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        
        # Add bounds checking to prevent overflow
        if num_points > 10000000:  # Reasonable upper limit
            logger.warning(f"Unusually large number of 3D points: {num_points}, truncating to 10000000")
            num_points = 10000000
        
        for _ in range(num_points):
            binary_point_line_properties = struct.unpack("<QdddBBBd", fid.read(43))
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7], dtype=np.int64)
            error = binary_point_line_properties[7]
            
            # Read track with overflow protection
            track_length = struct.unpack("<Q", fid.read(8))[0]
            
            # Add bounds checking to prevent overflow
            if track_length > 100000:  # Reasonable upper limit for track length
                logger.warning(f"Unusually large track length ({track_length}) for point {point3D_id}, truncating to 100000")
                track_length = 100000
            
            if track_length > 0:
                try:
                    # Use int64 to prevent overflow in struct calculations
                    total_elements = int(track_length) * 2
                    total_bytes = int(track_length) * 8
                    
                    track_elems = struct.unpack(f"<{total_elements}i", fid.read(total_bytes))
                    image_ids = np.array(track_elems[0::2], dtype=np.int64)
                    point2D_idxs = np.array(track_elems[1::2], dtype=np.int64)
                except (struct.error, OverflowError, MemoryError) as e:
                    logger.warning(f"Error reading track for point {point3D_id}: {e}, skipping")
                    image_ids = np.array([], dtype=np.int64)
                    point2D_idxs = np.array([], dtype=np.int64)
            else:
                image_ids = np.array([], dtype=np.int64)
                point2D_idxs = np.array([], dtype=np.int64)
            
            points3D[point3D_id] = {
                'xyz': xyz,
                'rgb': rgb,
                'error': error,
                'track': list(zip(image_ids, point2D_idxs))
            }
    return points3D


def read_colmap_binary_results(sparse_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Read COLMAP binary results and return proper dictionaries"""
    
    # Look for reconstruction directory
    reconstruction_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
    if not reconstruction_dirs:
        logger.warning("No reconstruction found")
        return {}, {}, {}
    
    # Use first (or largest) reconstruction
    recon_dir = reconstruction_dirs[0]
    
    logger.info(f"Reading COLMAP results from {recon_dir}")
    
    # Check for required output files
    cameras_file = recon_dir / "cameras.bin"
    images_file = recon_dir / "images.bin"  
    points_file = recon_dir / "points3D.bin"
    
    if cameras_file.exists() and images_file.exists() and points_file.exists():
        # Get file sizes for verification
        cameras_size = cameras_file.stat().st_size
        images_size = images_file.stat().st_size
        points_size = points_file.stat().st_size
        
        logger.info(f"COLMAP reconstruction successful:")
        logger.info(f"  - cameras.bin: {cameras_size} bytes")
        logger.info(f"  - images.bin: {images_size} bytes") 
        logger.info(f"  - points3D.bin: {points_size} bytes")
        
        # Read actual COLMAP binary files with comprehensive error handling
        try:
            cameras = read_cameras_binary(cameras_file)
            logger.debug(f"Successfully read {len(cameras)} cameras")
        except Exception as e:
            logger.warning(f"Error reading cameras.bin: {e}")
            cameras = {}
            
        try:
            images = read_images_binary(images_file)
            logger.debug(f"Successfully read {len(images)} images")
        except Exception as e:
            logger.warning(f"Error reading images.bin: {e}")
            images = {}
            
        try:
            points3d = read_points3d_binary(points_file)
            logger.debug(f"Successfully read {len(points3d)} points")
        except Exception as e:
            logger.warning(f"Error reading points3D.bin: {e}")
            points3d = {}
        
        # Log comprehensive reconstruction statistics like hloc
        num_cameras = len(cameras)
        num_images = len(images)
        num_points = len(points3d)
        
        if num_cameras > 0 or num_images > 0 or num_points > 0:
            # Calculate statistics
            registered_images = sum(1 for img_data in images.values() if len(img_data.get('point3D_ids', [])) > 0)
            observations = sum(len(point_data.get('track', [])) for point_data in points3d.values())
            mean_track_length = observations / num_points if num_points > 0 else 0.0
            mean_observations_per_image = observations / registered_images if registered_images > 0 else 0.0
            
            # Calculate reprojection errors if available
            errors = [point_data.get('error', 0.0) for point_data in points3d.values()]
            mean_reprojection_error = np.mean(errors) if errors else 0.0
            
            logger.info("=" * 50)
            logger.info("RECONSTRUCTION STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Cameras: {num_cameras}")
            logger.info(f"Images: {num_images}")
            logger.info(f"Registered images: {registered_images}")
            logger.info(f"Points: {num_points}")
            logger.info(f"Observations: {observations}")
            logger.info(f"Mean track length: {mean_track_length:.2f}")
            logger.info(f"Mean observations per image: {mean_observations_per_image:.2f}")
            logger.info(f"Mean reprojection error: {mean_reprojection_error:.4f} px")
            logger.info("=" * 50)
        else:
            logger.warning("No reconstruction data could be read from binary files")
        
        return points3d, cameras, images
    else:
        logger.error("COLMAP reconstruction files not found")
        missing = []
        if not cameras_file.exists(): missing.append("cameras.bin")
        if not images_file.exists(): missing.append("images.bin")
        if not points_file.exists(): missing.append("points3D.bin")
        logger.error(f"Missing files: {missing}")
        return {}, {}, {}


def colmap_binary_reconstruction(features: Dict[str, Any], matches: Dict[Tuple[str, str], Any],
                                output_path: Path, image_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Run complete COLMAP reconstruction using binary executable"""
    
    database_path = output_path / "database.db"
    
    # First filter matches with cv2 MAGSAC
    filtered_matches = filter_matches_with_magsac(features, matches)
    
    if not filtered_matches:
        logger.error("No matches passed MAGSAC filtering")
        return {}, {}, {}
    
    # Create database with filtered matches
    image_ids = create_colmap_database(features, filtered_matches, database_path)
    
    # Run COLMAP binary
    success = run_colmap_binary(database_path, image_dir, output_path)
    
    if success:
        # Read results
        return read_colmap_binary_results(output_path / "sparse")
    else:
        logger.error("COLMAP reconstruction failed")
        return {}, {}, {}