"""
Feature extractors for SfM pipeline
Supports SuperPoint, ALIKED, and DISK
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import cv2
from pathlib import Path
import logging

# Don't import LightGlue at module level - use lazy imports


logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None):
        self.device = device
        self.config = config or {}
        self.model = None
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """Setup the feature extraction model"""
        pass
    
    @abstractmethod
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract features from images"""
        pass
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for feature extraction"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def clear_memory(self):
        """Clear GPU memory used by the feature extractor"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"{self.__class__.__name__} memory cleared")
        except Exception as e:
            logger.warning(f"Error clearing {self.__class__.__name__} memory: {e}")


class SuperPointExtractor(BaseFeatureExtractor):
    """SuperPoint feature extractor"""
    
    def _setup_model(self):
        """Setup SuperPoint model"""
        try:
            # Import only SuperPoint, avoid SIFT module that needs pycolmap
            import lightglue
            from lightglue.superpoint import SuperPoint
            
            # SuperPoint configuration matching the provided format
            superpoint_conf = {
                "nms_radius": self.config.get('nms_radius', 4),
                "keypoint_threshold": self.config.get('keypoint_threshold', 0.005),  
                "max_keypoints": self.config.get('max_keypoints', 4096),  
                "remove_borders": self.config.get('remove_borders', 4),
                "fix_sampling": self.config.get('fix_sampling', False),
            }
            
            self.model = SuperPoint(
                nms_radius=superpoint_conf["nms_radius"],
                keypoint_threshold=superpoint_conf["keypoint_threshold"],
                max_keypoints=superpoint_conf["max_keypoints"],
                remove_borders=superpoint_conf["remove_borders"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue SuperPoint not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract SuperPoint features from images"""
        features = {}
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img_data in batch:
                image = img_data['image']
                image_path = img_data['path']
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Extract features
                with torch.no_grad():
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                features[image_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': image.shape[:2]
                }
                
                # Clean up intermediate tensors
                del image_tensor, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return features


class ALIKEDExtractor(BaseFeatureExtractor):
    """ALIKED feature extractor"""
    
    def _setup_model(self):
        """Setup ALIKED model"""
        try:
            from lightglue.aliked import ALIKED
            
            # ALIKED configuration matching the provided format
            aliked_conf = {
                "model_name": self.config.get('model_name', 'aliked-n16'),
                "max_num_keypoints": self.config.get('max_num_keypoints', 4096),  # Set reasonable max limit
                "detection_threshold": self.config.get('detection_threshold', 0.1),  # Lower threshold for more keypoints
                "nms_radius": self.config.get('nms_radius', 2),
            }
            
            self.model = ALIKED(
                model_name=aliked_conf["model_name"],
                max_num_keypoints=aliked_conf["max_num_keypoints"],
                detection_threshold=aliked_conf["detection_threshold"],
                nms_radius=aliked_conf["nms_radius"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue ALIKED not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract ALIKED features from images"""
        features = {}
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img_data in batch:
                image = img_data['image']
                image_path = img_data['path']
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Extract features
                with torch.no_grad():
                    # ALIKED expects a dictionary with 'image' key
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                features[image_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': image.shape[:2]
                }
                
                # Clean up intermediate tensors
                del image_tensor, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return features


class DISKExtractor(BaseFeatureExtractor):
    """DISK feature extractor"""
    
    def _setup_model(self):
        """Setup DISK model"""
        try:
            from lightglue.disk import DISK
            
            # DISK configuration matching the provided format
            disk_conf = {
                "weights": self.config.get('weights', 'depth'),
                "max_keypoints": self.config.get('max_keypoints', None),
                "nms_window_size": self.config.get('nms_window_size', 5),
                "detection_threshold": self.config.get('detection_threshold', 0.0),
                "pad_if_not_divisible": self.config.get('pad_if_not_divisible', True),
            }
            
            self.model = DISK(
                weights=disk_conf["weights"],
                max_keypoints=disk_conf["max_keypoints"],
                nms_window_size=disk_conf["nms_window_size"],
                detection_threshold=disk_conf["detection_threshold"],
                pad_if_not_divisible=disk_conf["pad_if_not_divisible"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue DISK not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract DISK features from images"""
        features = {}
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img_data in batch:
                image = img_data['image']
                image_path = img_data['path']
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Extract features
                with torch.no_grad():
                    # DISK expects a dictionary with 'image' key
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                features[image_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': image.shape[:2]
                }
                
                # Clean up intermediate tensors
                del image_tensor, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return features


class FeatureExtractorFactory:
    """Factory for creating feature extractors"""
    
    _extractors = {
        'superpoint': SuperPointExtractor,
        'aliked': ALIKEDExtractor,
        'disk': DISKExtractor
    }
    
    @classmethod
    def create(cls, extractor_type: str, device: torch.device, config: Dict[str, Any] = None) -> BaseFeatureExtractor:
        """Create a feature extractor instance"""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(cls._extractors.keys())}")
        
        return cls._extractors[extractor_type](device, config)
    
    @classmethod
    def get_supported_extractors(cls) -> List[str]:
        """Get list of supported extractor types"""
        return list(cls._extractors.keys())


class FeatureExtractor:
    """Main feature extractor class that wraps the factory"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.extractor_type = config.get('feature_extractor', 'superpoint')
        self.max_keypoints = config.get('max_keypoints', 4096)
        self.max_image_size = config.get('max_image_size', 1600)
        
        # Create the actual extractor
        self.extractor = FeatureExtractorFactory.create(self.extractor_type, self.device, self.config)
    
    def extract(self, image_paths: List[str]) -> Dict[str, Any]:
        """Extract features from images"""
        # Load and preprocess images
        images = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is not None:
                # Resize if needed
                if max(image.shape[:2]) > self.max_image_size:
                    scale = self.max_image_size / max(image.shape[:2])
                    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                    image = cv2.resize(image, new_size)
                
                images.append({
                    'image': image,
                    'path': path
                })
        
        # Extract features
        features = self.extractor.extract_features(images)
        
        return features 