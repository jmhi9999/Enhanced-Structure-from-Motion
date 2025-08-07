"""
COLMAP-based reconstruction using pycolmap for robustness
Based on hloc approach for reliable SfM
"""

import os
# Fix CUDA library path issue before importing pycolmap
if 'CUDA_HOME' not in os.environ:
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.2'
if '/usr/local/cuda-12.2/lib64' not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pycolmap
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class COLMAPDatabase:
    """COLMAP database interface for features and matches"""
    
    def __init__(self, database_path: Path):
        self.database_path = database_path
        
    @classmethod
    def connect(cls, database_path: Path):
        return cls(database_path)
        
    def create_tables(self):
        """Create COLMAP database tables"""
        conn = sqlite3.connect(self.database_path)
        
        # Create tables as per COLMAP schema
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
                prior_tz REAL,
                CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
                
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
                
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
        '''.format(*([2**31] * 4)))
        
        conn.commit()
        conn.close()
        
    def add_camera(self, model: int, width: int, height: int, params: np.ndarray, 
                   camera_id: Optional[int] = None, prior_focal_length: bool = True):
        """Add camera to database"""
        conn = sqlite3.connect(self.database_path)
        params_blob = params.astype(np.float64).tobytes()
        
        if camera_id is None:
            conn.execute(
                "INSERT INTO cameras(model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
                (model, width, height, params_blob, int(prior_focal_length))
            )
        else:
            conn.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, model, width, height, params_blob, int(prior_focal_length))
            )
        conn.commit()
        conn.close()
        
    def add_image(self, name: str, camera_id: int, image_id: Optional[int] = None):
        """Add image to database"""
        conn = sqlite3.connect(self.database_path)
        
        if image_id is None:
            conn.execute(
                "INSERT INTO images(name, camera_id) VALUES (?, ?)",
                (name, camera_id)
            )
        else:
            conn.execute(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                (image_id, name, camera_id)
            )
        conn.commit()
        conn.close()
        
    def add_keypoints(self, image_id: int, keypoints: np.ndarray):
        """Add keypoints to database"""
        conn = sqlite3.connect(self.database_path)
        
        keypoints = keypoints.astype(np.float32)
        keypoints_blob = keypoints.tobytes()
        
        conn.execute(
            "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob)
        )
        conn.commit()
        conn.close()
        
    def add_matches(self, image_id0: int, image_id1: int, matches: np.ndarray):
        """Add matches to database"""
        conn = sqlite3.connect(self.database_path)
        
        pair_id = self._image_ids_to_pair_id(image_id0, image_id1)
        matches = matches.astype(np.uint32)
        matches_blob = matches.tobytes()
        
        conn.execute(
            "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], matches_blob)
        )
        conn.commit()
        conn.close()
        
    def add_two_view_geometry(self, image_id0: int, image_id1: int, matches: np.ndarray,
                             F: np.ndarray = None, E: np.ndarray = None, H: np.ndarray = None,
                             qvec: np.ndarray = None, tvec: np.ndarray = None, config: int = 2):
        """Add two view geometry to database"""
        conn = sqlite3.connect(self.database_path)
        
        pair_id = self._image_ids_to_pair_id(image_id0, image_id1)
        matches = matches.astype(np.uint32)
        matches_blob = matches.tobytes()
        
        # Convert arrays to blobs
        F_blob = F.astype(np.float64).tobytes() if F is not None else None
        E_blob = E.astype(np.float64).tobytes() if E is not None else None
        H_blob = H.astype(np.float64).tobytes() if H is not None else None
        qvec_blob = qvec.astype(np.float64).tobytes() if qvec is not None else None
        tvec_blob = tvec.astype(np.float64).tobytes() if tvec is not None else None
        
        conn.execute(
            "INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], matches_blob, config, F_blob, E_blob, H_blob, qvec_blob, tvec_blob)
        )
        conn.commit()
        conn.close()
        
    def execute(self, query: str, params: Tuple = ()):
        """Execute query and return cursor"""
        conn = sqlite3.connect(self.database_path)
        return conn.execute(query, params)
        
    def commit(self):
        """Commit changes"""
        pass  # Auto-commit in individual operations
        
    def close(self):
        """Close database"""
        pass  # Auto-close in individual operations
        
    @staticmethod
    def _image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
        """Convert image IDs to pair ID"""
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * 2147483647 + image_id2


class COLMAPReconstruction:
    """COLMAP-based SfM reconstruction using pycolmap"""
    
    def __init__(self, output_path: Path, device: str = "cpu"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_path / "database.db"
        self.device = device
        
    def create_database(self, features: Dict[str, Any], matches: Dict[Tuple[str, str], Any]) -> Dict[str, int]:
        """Create COLMAP database with features and matches"""
        logger.info("Creating COLMAP database...")
        
        # Create empty database
        if self.database_path.exists():
            self.database_path.unlink()
            
        db = COLMAPDatabase.connect(self.database_path)
        db.create_tables()
        
        # Add camera (assume single camera for now)
        first_feature = next(iter(features.values()))
        height, width = first_feature['image_shape']
        focal_length = 1.2 * max(width, height)  # hloc heuristic
        
        camera_params = np.array([focal_length, width / 2.0, height / 2.0])
        db.add_camera(
            model=0,  # SIMPLE_PINHOLE
            width=width,
            height=height,
            params=camera_params,
            camera_id=1
        )
        
        # Add images and get image IDs
        image_ids = {}
        for i, img_path in enumerate(features.keys(), 1):
            image_name = Path(img_path).name
            db.add_image(image_name, camera_id=1, image_id=i)
            image_ids[image_name] = i
            
        # Add keypoints
        logger.info("Adding keypoints to database...")
        for img_path, feat_data in tqdm(features.items(), desc="Adding keypoints"):
            image_name = Path(img_path).name
            image_id = image_ids[image_name]
            keypoints = feat_data['keypoints'] + 0.5  # COLMAP origin
            db.add_keypoints(image_id, keypoints)
            
        # Add matches
        logger.info("Adding matches to database...")
        for (img1_path, img2_path), match_data in tqdm(matches.items(), desc="Adding matches"):
            name0 = Path(img1_path).name
            name1 = Path(img2_path).name
            
            if name0 in image_ids and name1 in image_ids:
                id0, id1 = image_ids[name0], image_ids[name1]
                
                # Extract matches
                matches_array = np.column_stack([
                    match_data['matches0'],
                    match_data['matches1']
                ])
                
                db.add_matches(id0, id1, matches_array)
                
        return image_ids
        
    def run_geometric_verification(self, image_ids: Dict[str, int]):
        """Run geometric verification using COLMAP"""
        logger.info("Running geometric verification...")
        
        # Use COLMAP's two-view geometry estimation
        options = {
            'min_num_inliers': 15,
            'ransac_confidence': 0.999,
            'ransac_max_num_trials': 10000,
            'min_inlier_ratio': 0.1,
            'multiple_models': False,
            'guided_matching': False,
        }
        
        with pycolmap.ostream():
            pycolmap.verify_matches(
                self.database_path,
                options=options
            )
            
    def run_incremental_mapping(self, image_dir: Path) -> Dict:
        """Run COLMAP incremental mapping"""
        logger.info("Running COLMAP incremental mapping...")
        
        sfm_path = self.output_path / "sparse"
        sfm_path.mkdir(exist_ok=True)
        
        options = {
            'min_num_matches': 15,
            'ignore_watermarks': False,
            'multiple_models': False,
            'max_num_models': 50,
            'max_model_overlap': 20,
            'min_model_size': 10,
            'init_image_id1': -1,
            'init_image_id2': -1,
            'init_num_trials': 200,
            'extract_colors': True,
            'num_threads': -1,
            'min_focal_length_ratio': 0.1,
            'max_focal_length_ratio': 10.0,
            'max_extra_param': 1.0,
            'ba_refine_focal_length': True,
            'ba_refine_principal_point': False,
            'ba_refine_extra_params': True,
            'ba_min_num_residuals_for_multi_threading': 50000,
            'ba_local_num_images': 6,
            'ba_local_max_num_iterations': 25,
            'ba_global_images_ratio': 1.1,
            'ba_global_points_ratio': 1.1,
            'ba_global_images_freq': 500,
            'ba_global_points_freq': 250000,
            'ba_global_max_num_iterations': 50,
            'ba_local_max_refinements': 2,
            'ba_local_max_refinement_change': 0.001,
            'ba_global_max_refinements': 5,
            'ba_global_max_refinement_change': 0.0005,
        }
        
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                self.database_path,
                image_dir,
                sfm_path,
                options=options
            )
            
        return reconstructions
        
    def reconstruct(self, features: Dict[str, Any], matches: Dict[Tuple[str, str], Any], image_dir: Path) -> Tuple[Dict, Dict, Dict]:
        """Run complete COLMAP-based reconstruction"""
        # Create database
        image_ids = self.create_database(features, matches)
        
        # Run geometric verification
        self.run_geometric_verification(image_ids)
        
        # Run incremental mapping
        reconstructions = self.run_incremental_mapping(image_dir)
        
        if not reconstructions:
            logger.warning("No reconstructions found")
            return {}, {}, {}
            
        # Get largest reconstruction
        largest_reconstruction = max(reconstructions.values(), key=lambda x: len(x.images))
        
        logger.info(f"Reconstruction complete: {len(largest_reconstruction.images)} images, {len(largest_reconstruction.points3D)} points")
        
        # Save COLMAP standard binary files (images.bin, cameras.bin, points3D.bin)
        sparse_path = self.output_path / "sparse" / "0"
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Writing COLMAP binary files...")
        largest_reconstruction.write_binary(str(sparse_path))
        
        # Also save as text files for easier reading
        text_path = self.output_path / "sparse" / "0_text"
        text_path.mkdir(parents=True, exist_ok=True)
        largest_reconstruction.write_text(str(text_path))
        
        logger.info(f"COLMAP files saved to {sparse_path}")
        logger.info(f"Binary files: images.bin, cameras.bin, points3D.bin")
        logger.info(f"Text files saved to {text_path}")
        
        # Convert to our format
        points3d = {}
        for point_id, point in largest_reconstruction.points3D.items():
            points3d[point_id] = {
                'xyz': point.xyz,
                'rgb': point.color,
                'error': point.error,
                'track': [(img_id, pt_id) for img_id, pt_id in point.track.elements()]
            }
            
        cameras = {}
        for camera_id, camera in largest_reconstruction.cameras.items():
            cameras[camera_id] = {
                'model': camera.model_name,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params
            }
            
        images = {}
        for image_id, image in largest_reconstruction.images.items():
            images[image.name] = {
                'camera_id': image.camera_id,
                'qvec': image.qvec,
                'tvec': image.tvec,
                'name': image.name,
                'point3D_ids': []  # Would need to extract from matches
            }
            
        return points3d, cameras, images