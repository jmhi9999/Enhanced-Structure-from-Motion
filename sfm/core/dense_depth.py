"""
Dense depth estimation from sparse SfM points with monocular depth models
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import scipy.interpolate as interpolate
from pathlib import Path
import requests
from PIL import Image
import torchvision.transforms as transforms
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Import transformers for depth estimation - completely optional
try:
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
    DEPTH_MODEL_AVAILABLE = True
except (ImportError, TypeError, AttributeError) as e:
    DEPTH_MODEL_AVAILABLE = False
    DPTFeatureExtractor = None
    DPTForDepthEstimation = None
    # Silently handle the import error to avoid breaking the whole library


class DenseDepthEstimator:
    """Dense depth estimation from sparse SfM points with monocular depth models"""
    
    def __init__(self, device: torch.device, depth_model: str = 'dpt-large', high_quality: bool = True):
        self.device = device
        self.depth_model_name = depth_model
        self.high_quality = high_quality
        self.depth_model = None
        self.feature_extractor = None
        
        # Initialize depth estimation model
        self._initialize_depth_model()
        
    def _initialize_depth_model(self):
        """Initialize monocular depth estimation model"""
        if not DEPTH_MODEL_AVAILABLE:
            logger.warning("DPT depth model not available, using fallback")
            return
        
        try:
            # Load DPT model for depth estimation
            model_name = "Intel/dpt-large" if self.depth_model_name == 'dpt-large' else "Intel/dpt-hybrid-midas"
            logger.info(f"Loading DPT model: {model_name}")
            
            self.feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
            
            # Move to device
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
            logger.info(f"Successfully loaded DPT depth model on {self.device}")
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU memory insufficient for DPT model, falling back to CPU")
            if self.device.type == "cuda":
                self.device = torch.device("cpu")
                try:
                    self.depth_model = DPTForDepthEstimation.from_pretrained(model_name)
                    self.depth_model.to(self.device)
                    self.depth_model.eval()
                    logger.info("DPT model loaded on CPU")
                except Exception as e:
                    logger.error(f"Failed to load DPT model on CPU: {e}")
                    self.depth_model = None
                    
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            self.depth_model = None
    
    def estimate_dense_depth(self, sparse_points: Dict, cameras: Dict, 
                           images: Dict, features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Estimate dense depth maps from sparse SfM points with monocular depth"""
        
        print("Estimating dense depth maps with monocular depth estimation...")
        dense_depth_maps = {}
        
        for img_key, img_data in tqdm(images.items(), desc="Generating dense depth maps"):
            try:
                # Handle both path-based and ID-based keys for robustness
                if isinstance(img_key, (int, np.integer)):
                    # If key is numeric ID, use image name from data
                    img_path = img_data.get('name', f'image_{img_key}')
                else:
                    # If key is already a path, use it directly
                    img_path = img_key
                    
                depth_map = self._estimate_depth_for_image(
                    img_path, img_data, cameras, sparse_points, features
                )
                dense_depth_maps[img_path] = depth_map
            except Exception as e:
                print(f"Warning: Failed to estimate depth for {img_key}: {e}")
                continue
        
        return dense_depth_maps
    
    def _estimate_depth_for_image(self, img_path: str, img_data: Dict, 
                                cameras: Dict, sparse_points: Dict,
                                features: Dict[str, Any]) -> np.ndarray:
        """Estimate dense depth map for a single image"""
        
        # Get camera parameters
        camera_id = img_data.get('camera_id')
        if camera_id is None:
            # If camera_id is missing, try to use the first available camera or default to 1
            camera_id = next(iter(cameras.keys())) if cameras else 1
            logger.warning(f"Missing camera_id for {img_path}, using camera_id={camera_id}")
        camera = cameras[camera_id]
        
        # Get image dimensions
        height, width = camera['height'], camera['width']
        
        # Get camera pose
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(qvec)
        
        # Create camera matrix
        K = np.array(camera['params'][:3]).reshape(3, 3)
        
        # Get sparse depth values
        sparse_depths = self._get_sparse_depths(
            img_path, img_data, sparse_points, K, R, tvec
        )
        
        # Get monocular depth estimation
        monocular_depth = self._get_monocular_depth(img_path)
        
        # Combine sparse and monocular depth
        depth_map = self._combine_depth_estimates(
            sparse_depths, monocular_depth, width, height
        )
        
        # Apply geometry completion for texture-poor regions
        depth_map = self._apply_geometry_completion(depth_map, img_path)
        
        return depth_map
    
    def _get_monocular_depth(self, img_path: str) -> np.ndarray:
        """Get monocular depth estimation using DPT model"""
        if self.depth_model is None:
            # Fallback: return None for monocular depth
            return None
        
        try:
            # Load image with memory optimization
            image = Image.open(img_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get depth prediction with memory management
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy and free GPU memory immediately
            depth_map = predicted_depth.squeeze().cpu().numpy()
            del predicted_depth, outputs, inputs
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            
            # Normalize depth to reasonable range (0-100 meters)
            depth_map = self._normalize_depth_map(depth_map)
            
            return depth_map
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"GPU memory insufficient for depth estimation on {img_path}")
            torch.cuda.empty_cache()
            return None
            
        except Exception as e:
            logger.error(f"Monocular depth estimation failed for {img_path}: {e}")
            return None
    
    def _normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map to reasonable range"""
        # Remove outliers
        depth_map = np.clip(depth_map, np.percentile(depth_map, 1), np.percentile(depth_map, 99))
        
        # Normalize to 0-100 meter range
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 100.0
        
        return depth_map
    
    def _combine_depth_estimates(self, sparse_depths: List[Tuple[float, float, float]], 
                               monocular_depth: Optional[np.ndarray],
                               width: int, height: int) -> np.ndarray:
        """Combine sparse SfM depths with monocular depth estimation"""
        
        if len(sparse_depths) < 3 and monocular_depth is None:
            # Not enough data
            return np.zeros((height, width))
        
        # Initialize depth map
        depth_map = np.zeros((height, width))
        
        if monocular_depth is not None:
            # Use monocular depth as base
            depth_map = cv2.resize(monocular_depth, (width, height))
        
        # Interpolate sparse depths
        if len(sparse_depths) >= 3:
            sparse_depth_map = self._interpolate_depth_map(sparse_depths, width, height)
            
            # Combine with monocular depth
            if monocular_depth is not None:
                # Weighted combination
                sparse_weight = 0.7
                monocular_weight = 0.3
                
                # Normalize sparse depth to same range
                sparse_depth_map = self._normalize_depth_map(sparse_depth_map)
                
                # Combine
                depth_map = (sparse_weight * sparse_depth_map + 
                           monocular_weight * depth_map)
            else:
                depth_map = sparse_depth_map
        
        return depth_map
    
    def _get_sparse_depths(self, img_path: str, img_data: Dict, 
                          sparse_points: Dict, K: np.ndarray, 
                          R: np.ndarray, tvec: np.ndarray) -> List[Tuple[float, float, float]]:
        """Get sparse depth values for the image"""
        
        sparse_depths = []
        
        # Get 2D keypoints and their 3D point IDs
        keypoints = img_data['xys']
        point3d_ids = img_data['point3D_ids']
        
        for i, (kp, point3d_id) in enumerate(zip(keypoints, point3d_ids)):
            if point3d_id != -1 and point3d_id in sparse_points:
                # Get 3D point
                point3d = sparse_points[point3d_id]['xyz']
                
                # Project 3D point to 2D
                point3d_homo = np.append(point3d, 1.0)
                point2d_proj = K @ (R @ point3d_homo[:3] + tvec)
                point2d_proj = point2d_proj[:2] / point2d_proj[2]
                
                # Calculate depth
                depth = np.linalg.norm(R @ point3d + tvec)
                
                # Add to sparse depths
                sparse_depths.append((point2d_proj[0], point2d_proj[1], depth))
        
        return sparse_depths
    
    def _interpolate_depth_map(self, sparse_depths: List[Tuple[float, float, float]], 
                              width: int, height: int) -> np.ndarray:
        """Interpolate sparse depths to dense depth map"""
        
        if len(sparse_depths) < 3:
            # Not enough points for interpolation
            return np.zeros((height, width))
        
        # Extract coordinates and depths
        coords = np.array([(d[0], d[1]) for d in sparse_depths])
        depths = np.array([d[2] for d in sparse_depths])
        
        # Create grid for interpolation
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Use scipy's griddata for interpolation
        try:
            depth_map = interpolate.griddata(
                coords, depths, (x_grid, y_grid), 
                method='linear', fill_value=0.0
            )
        except:
            # Fallback to nearest neighbor if linear fails
            depth_map = interpolate.griddata(
                coords, depths, (x_grid, y_grid), 
                method='nearest', fill_value=0.0
            )
        
        return depth_map
    
    def _apply_geometry_completion(self, depth_map: np.ndarray, img_path: str) -> np.ndarray:
        """Apply geometry completion for texture-poor regions"""
        
        # Fill holes using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = (depth_map > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Interpolate remaining holes
        depth_map_filled = depth_map.copy()
        zero_mask = (depth_map == 0)
        
        if np.any(zero_mask):
            # Use distance transform to fill holes
            from scipy import ndimage
            depth_map_filled = ndimage.distance_transform_edt(
                depth_map_filled, return_distances=False, return_indices=True
            )
        
        # Apply bilateral filter for smoothness
        depth_map_smooth = cv2.bilateralFilter(
            depth_map_filled.astype(np.float32), 9, 75, 75
        )
        
        return depth_map_smooth
    
    def _quaternion_to_rotation_matrix(self, qvec: List[float]) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        w, x, y, z = qvec
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        return R
    
    def save_depth_maps(self, depth_maps: Dict[str, np.ndarray], output_dir: str):
        """Save depth maps to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for img_path, depth_map in depth_maps.items():
            # Normalize depth map for visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            
            # Save as image
            img_name = Path(img_path).stem
            output_file = output_path / f"{img_name}_depth.png"
            cv2.imwrite(str(output_file), depth_uint8)
            
            # Save raw depth data
            output_file_raw = output_path / f"{img_name}_depth.npy"
            np.save(str(output_file_raw), depth_map)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get depth estimation statistics"""
        return {
            "depth_model_available": self.depth_model is not None,
            "device": str(self.device),
            "model_type": "DPT-Large" if self.depth_model else "Fallback"
        } 