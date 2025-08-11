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
    # Log the actual import error to help with debugging
    import logging
    logging.getLogger(__name__).warning(
        f"Could not import DPT model components from transformers: {e}",
        exc_info=True
    )


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
        if camera_id is None or camera_id not in cameras:
            # If camera_id is missing or invalid, use the first available camera
            if cameras:
                camera_id = next(iter(cameras.keys()))
                logger.warning(f"Missing or invalid camera_id for {img_path}, using camera_id={camera_id}")
            else:
                # Skip depth estimation for images without cameras (not registered in COLMAP)
                logger.warning(f"No cameras available and missing camera_id for {img_path}, skipping depth estimation")
                return np.zeros((480, 640))  # Return dummy depth map
        camera = cameras[camera_id]
        
        # Get image dimensions
        height, width = camera['height'], camera['width']
        
        # Get camera pose
        qvec = img_data['qvec']
        tvec = img_data['tvec']
        
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_rotation_matrix(qvec)
        
        # Create camera matrix
        params = camera['params']
        if len(params) >= 4:  # PINHOLE model: fx, fy, cx, cy
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif len(params) >= 3:  # SIMPLE_PINHOLE model: f, cx, cy
            f, cx, cy = params[:3]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        else:
            logger.warning(f"Invalid camera parameters: {params}, using default")
            f = max(width, height) * 1.2
            K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]])
        
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
        """Get monocular depth estimation using DPT model with robust error handling"""
        if self.depth_model is None or not DEPTH_MODEL_AVAILABLE:
            logger.debug(f"DPT model not available, skipping monocular depth for {Path(img_path).name}")
            return None
        
        try:
            # Check if image file exists
            if not Path(img_path).exists():
                logger.warning(f"Image file not found: {img_path}")
                return None
            
            # Load and validate image
            try:
                image = Image.open(img_path).convert('RGB')
                if image.size[0] == 0 or image.size[1] == 0:
                    logger.warning(f"Invalid image dimensions for {img_path}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                return None
            
            # Resize image if too large to prevent memory issues
            max_size = 1024  # Maximum dimension
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {Path(img_path).name} to {new_size}")
            
            # Prepare inputs with error handling
            try:
                inputs = self.feature_extractor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                logger.error(f"Feature extraction failed for {img_path}: {e}")
                return None
            
            # Get depth prediction with memory management
            try:
                with torch.no_grad():
                    outputs = self.depth_model(**inputs)
                    if hasattr(outputs, 'predicted_depth'):
                        predicted_depth = outputs.predicted_depth
                    else:
                        # Fallback for different model outputs
                        predicted_depth = outputs.prediction if hasattr(outputs, 'prediction') else outputs[0]
                
                # Convert to numpy and validate
                depth_map = predicted_depth.squeeze().cpu().numpy()
                
                # Clean up GPU memory immediately
                del predicted_depth, outputs, inputs
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Validate depth map
                if depth_map.size == 0 or not np.isfinite(depth_map).any():
                    logger.warning(f"Invalid depth map generated for {img_path}")
                    return None
                
                # Normalize depth to reasonable range (0-50 meters)
                depth_map = self._normalize_depth_map(depth_map)
                
                logger.debug(f"Generated monocular depth for {Path(img_path).name}: {depth_map.min():.2f}-{depth_map.max():.2f}m")
                return depth_map
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"GPU memory insufficient for depth estimation on {Path(img_path).name}")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                return None
            
            except RuntimeError as e:
                if "CUDA" in str(e):
                    logger.error(f"CUDA error during depth estimation for {Path(img_path).name}: {e}")
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"Runtime error during depth estimation for {Path(img_path).name}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Monocular depth estimation failed for {Path(img_path).name}: {e}")
            return None
    
    def _normalize_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Normalize depth map to reasonable range with robust handling"""
        try:
            # Handle edge cases
            if depth_map.size == 0:
                return depth_map
            
            # Remove NaN and infinite values
            depth_map_clean = np.nan_to_num(depth_map, nan=0.0, posinf=50.0, neginf=0.0)
            
            # Remove outliers using percentiles
            valid_depths = depth_map_clean[depth_map_clean > 0]
            
            if len(valid_depths) == 0:
                logger.warning("All depth values are zero or invalid")
                return depth_map_clean
            
            # Use percentile-based clipping to remove outliers
            p1, p99 = np.percentile(valid_depths, [1, 99])
            depth_clipped = np.clip(depth_map_clean, 0, p99)
            
            # Normalize to 0-50 meter range (more realistic for indoor/outdoor scenes)
            depth_min, depth_max = depth_clipped.min(), depth_clipped.max()
            
            if depth_max > depth_min and depth_max > 0:
                # Linear normalization to 0-50m range
                depth_normalized = (depth_clipped - depth_min) / (depth_max - depth_min) * 50.0
            else:
                # Fallback: keep original values if normalization fails
                depth_normalized = depth_clipped
            
            # Ensure reasonable depth values
            depth_normalized = np.clip(depth_normalized, 0.1, 200.0)  # 10cm to 200m range
            
            return depth_normalized
            
        except Exception as e:
            logger.warning(f"Depth normalization failed: {e}, returning original")
            return depth_map
    
    def _combine_depth_estimates(self, sparse_depths: List[Tuple[float, float, float]], 
                               monocular_depth: Optional[np.ndarray],
                               width: int, height: int) -> np.ndarray:
        """Combine sparse SfM depths with monocular depth estimation"""
        
        logger.debug(f"Combining depth estimates: {len(sparse_depths)} sparse points, monocular: {monocular_depth is not None}")
        
        # Initialize depth map
        depth_map = np.zeros((height, width))
        
        # Start with monocular depth if available
        if monocular_depth is not None and monocular_depth.size > 0:
            try:
                # Resize monocular depth to match target dimensions
                if monocular_depth.shape != (height, width):
                    depth_map = cv2.resize(monocular_depth, (width, height), interpolation=cv2.INTER_LINEAR)
                else:
                    depth_map = monocular_depth.copy()
                logger.debug(f"Using monocular depth as base: {depth_map.min():.2f} - {depth_map.max():.2f}")
            except Exception as e:
                logger.warning(f"Error using monocular depth: {e}")
                depth_map = np.zeros((height, width))
        
        # Add sparse depth constraints
        if len(sparse_depths) >= 3:
            try:
                sparse_depth_map = self._interpolate_depth_map(sparse_depths, width, height)
                
                if sparse_depth_map.max() > 0:
                    logger.debug(f"Sparse depth range: {sparse_depth_map.min():.2f} - {sparse_depth_map.max():.2f}")
                    
                    if depth_map.max() > 0:  # Combine with existing monocular depth
                        # Scale matching for better fusion
                        sparse_valid = sparse_depth_map > 0
                        monocular_valid = depth_map > 0
                        
                        if np.any(sparse_valid & monocular_valid):
                            # Find scale factor between sparse and monocular depth
                            overlap_sparse = sparse_depth_map[sparse_valid & monocular_valid]
                            overlap_monocular = depth_map[sparse_valid & monocular_valid]
                            
                            if len(overlap_sparse) > 0:
                                scale_factor = np.median(overlap_sparse) / (np.median(overlap_monocular) + 1e-8)
                                scale_factor = np.clip(scale_factor, 0.1, 10.0)  # Reasonable range
                                logger.debug(f"Applying scale factor: {scale_factor:.3f}")
                                depth_map *= scale_factor
                        
                        # Weighted combination
                        sparse_weight = 0.8  # Trust sparse more
                        monocular_weight = 0.2
                        
                        # Create combined depth map
                        combined = depth_map.copy()
                        combined[sparse_valid] = (sparse_weight * sparse_depth_map[sparse_valid] + 
                                               monocular_weight * depth_map[sparse_valid])
                        depth_map = combined
                    else:
                        # Use sparse depth only
                        depth_map = sparse_depth_map
                        logger.debug("Using sparse depth only")
                        
            except Exception as e:
                logger.warning(f"Error combining sparse depths: {e}")
        
        # Final validation
        if depth_map.max() == 0:
            logger.warning("Final depth map is all zeros, creating fallback")
            # Create a simple depth gradient as last resort
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            depth_map = ((y_coords / height) * 0.3 + 0.7) * 5.0  # 3.5-5m range
        
        logger.debug(f"Final depth map range: {depth_map.min():.2f} - {depth_map.max():.2f}")
        return depth_map
    
    def _get_sparse_depths(self, img_path: str, img_data: Dict, 
                          sparse_points: Dict, K: np.ndarray, 
                          R: np.ndarray, tvec: np.ndarray) -> List[Tuple[float, float, float]]:
        """Get sparse depth values for the image with corrected projection"""
        
        sparse_depths = []
        
        # Get 2D keypoints and their 3D point IDs
        keypoints = img_data.get('xys', [])
        point3d_ids = img_data.get('point3D_ids', [])
        
        if len(keypoints) == 0 or len(point3d_ids) == 0:
            logger.warning(f"No keypoints or point3D_ids found for {img_path}")
            return sparse_depths
        
        valid_count = 0
        for i, (kp, point3d_id) in enumerate(zip(keypoints, point3d_ids)):
            if point3d_id != -1 and point3d_id in sparse_points:
                try:
                    # Get 3D point in world coordinates
                    point3d_world = np.array(sparse_points[point3d_id]['xyz'])
                    
                    # Transform to camera coordinates: P_cam = R * P_world + t
                    point3d_cam = R @ point3d_world + tvec
                    
                    # Check if point is in front of camera
                    if point3d_cam[2] <= 0:
                        continue
                    
                    # Project to image coordinates
                    point2d_proj = K @ point3d_cam
                    if abs(point2d_proj[2]) < 1e-8:  # Avoid division by zero
                        continue
                        
                    point2d_proj = point2d_proj[:2] / point2d_proj[2]
                    
                    # Calculate depth (Z coordinate in camera frame)
                    depth = point3d_cam[2]
                    
                    # Sanity check: depth should be positive and reasonable
                    if depth > 0 and depth < 1000:  # Max 1km depth
                        sparse_depths.append((point2d_proj[0], point2d_proj[1], depth))
                        valid_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing point {point3d_id}: {e}")
                    continue
        
        logger.debug(f"Generated {valid_count} valid sparse depths for {Path(img_path).name}")
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
        """Apply geometry completion for texture-poor regions with proper hole filling"""
        
        try:
            if depth_map.max() == 0:
                logger.debug(f"Skipping geometry completion for all-zero depth map: {Path(img_path).name}")
                return depth_map
            
            depth_map_filled = depth_map.copy()
            
            # Create mask of valid (non-zero) pixels
            valid_mask = (depth_map > 0).astype(np.uint8)
            
            # Fill small holes using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            valid_mask_closed = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find pixels that need interpolation
            hole_mask = (valid_mask_closed > 0) & (depth_map == 0)
            
            if np.any(hole_mask):
                # Use inpainting to fill holes
                depth_map_uint8 = (depth_map / depth_map.max() * 255).astype(np.uint8)
                inpaint_mask = hole_mask.astype(np.uint8)
                
                # OpenCV inpainting for hole filling
                inpainted = cv2.inpaint(depth_map_uint8, inpaint_mask, 3, cv2.INPAINT_TELEA)
                
                # Convert back to depth values
                depth_map_filled = (inpainted / 255.0) * depth_map.max()
                
                # Preserve original valid depths
                depth_map_filled[valid_mask > 0] = depth_map[valid_mask > 0]
            
            # Apply gentle smoothing to reduce noise
            if depth_map_filled.max() > 0:
                depth_map_smooth = cv2.bilateralFilter(
                    depth_map_filled.astype(np.float32), 5, 50, 50
                )
            else:
                depth_map_smooth = depth_map_filled
            
            return depth_map_smooth
            
        except Exception as e:
            logger.warning(f"Geometry completion failed for {Path(img_path).name}: {e}")
            return depth_map
    
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
        """Save depth maps to files with proper value handling"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(depth_maps)} depth maps to {output_path}")
        
        for img_path, depth_map in depth_maps.items():
            try:
                # Check if depth map has valid values
                if depth_map is None or depth_map.size == 0:
                    logger.warning(f"Empty depth map for {img_path}, skipping")
                    continue
                
                # Remove NaN and inf values
                depth_map_clean = np.nan_to_num(depth_map, nan=0.0, posinf=100.0, neginf=0.0)
                
                # Check if depth map has any non-zero values
                non_zero_count = np.count_nonzero(depth_map_clean)
                total_count = depth_map_clean.size
                logger.debug(f"Depth map for {Path(img_path).name}: {non_zero_count}/{total_count} non-zero pixels")
                
                if non_zero_count == 0:
                    logger.warning(f"Depth map for {img_path} is all zeros, creating dummy depth")
                    # Create a simple gradient depth map as fallback
                    h, w = depth_map_clean.shape
                    y_coords, x_coords = np.mgrid[0:h, 0:w]
                    depth_map_clean = (y_coords / h + x_coords / w) * 10.0  # 0-20m range
                
                # Get depth statistics
                min_depth = np.min(depth_map_clean[depth_map_clean > 0]) if non_zero_count > 0 else 0.0
                max_depth = np.max(depth_map_clean)
                mean_depth = np.mean(depth_map_clean[depth_map_clean > 0]) if non_zero_count > 0 else 0.0
                
                logger.debug(f"Depth stats for {Path(img_path).name}: min={min_depth:.2f}, max={max_depth:.2f}, mean={mean_depth:.2f}")
                
                # Normalize depth map for visualization (0-255)
                if max_depth > min_depth and max_depth > 0:
                    # Use percentile-based normalization for better visualization
                    p1, p99 = np.percentile(depth_map_clean[depth_map_clean > 0], [1, 99]) if non_zero_count > 0 else (0, max_depth)
                    depth_clipped = np.clip(depth_map_clean, p1, p99)
                    depth_normalized = (depth_clipped - p1) / (p99 - p1 + 1e-8)
                else:
                    depth_normalized = np.ones_like(depth_map_clean) * 0.5  # Gray fallback
                
                # Convert to uint8 for PNG saving
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                
                # Apply colormap for better visualization
                depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
                
                # Save as image
                img_name = Path(img_path).stem
                output_file = output_path / f"{img_name}_depth.png"
                success = cv2.imwrite(str(output_file), depth_colored)
                
                # Also save grayscale version
                output_file_gray = output_path / f"{img_name}_depth_gray.png"
                cv2.imwrite(str(output_file_gray), depth_uint8)
                
                if success:
                    logger.debug(f"Saved depth map: {output_file}")
                else:
                    logger.warning(f"Failed to save depth map: {output_file}")
                
                # Save raw depth data
                output_file_raw = output_path / f"{img_name}_depth.npy"
                np.save(str(output_file_raw), depth_map_clean)
                
            except Exception as e:
                logger.error(f"Error saving depth map for {img_path}: {e}")
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get depth estimation statistics"""
        return {
            "depth_model_available": self.depth_model is not None,
            "device": str(self.device),
            "model_type": "DPT-Large" if self.depth_model else "Fallback"
        } 