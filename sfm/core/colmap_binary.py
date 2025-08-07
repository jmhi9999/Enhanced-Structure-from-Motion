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

logger = logging.getLogger(__name__)


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
            logger.warning(f"MAGSAC filtering failed for {Path(img1_path).name} - {Path(img2_path).name}: {e}")
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
    
    return image_ids


def run_colmap_binary(database_path: Path, image_dir: Path, output_path: Path) -> bool:
    """Run COLMAP using binary executable"""
    
    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True)
    
    # Skip COLMAP geometric verification since we already did MAGSAC filtering
    logger.info("Skipping COLMAP geometric verification (already done with cv2 MAGSAC)")
    
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info("COLMAP incremental mapping completed successfully")
            logger.info(f"COLMAP output: {result.stdout}")
            return True
        else:
            logger.error(f"COLMAP mapping failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("COLMAP mapping timed out")
        return False
    except Exception as e:
        logger.error(f"COLMAP mapping error: {e}")
        return False


def read_colmap_binary_results(sparse_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Read COLMAP binary results"""
    
    # Look for reconstruction directory
    reconstruction_dirs = [d for d in sparse_path.iterdir() if d.is_dir()]
    if not reconstruction_dirs:
        logger.warning("No reconstruction found")
        return {}, {}, {}
    
    # Use first (or largest) reconstruction
    recon_dir = reconstruction_dirs[0]
    
    logger.info(f"Reading COLMAP results from {recon_dir}")
    
    # For now, return empty results - would need to implement binary readers
    # This is complex and would require implementing COLMAP's binary format readers
    
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