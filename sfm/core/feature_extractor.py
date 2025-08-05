"""
Feature extractors for SfM pipeline
Supports SuperPoint, ALIKED, and DISK
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import cv2
from pathlib import Path

try:
    from lightglue import SuperPoint, ALIKED, DISK
except ImportError:
    print("Warning: LightGlue not installed. Using fallback implementations.")


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    def __init__(self, device: torch.device):
        self.device = device
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
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)


class SuperPointExtractor(BaseFeatureExtractor):
    """SuperPoint feature extractor"""
    
    def _setup_model(self):
        try:
            self.model = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        except NameError:
            # Fallback implementation
            self.model = self._create_superpoint_fallback()
    
    def _create_superpoint_fallback(self):
        """Fallback SuperPoint implementation"""
        # Simplified SuperPoint implementation
        class SimpleSuperPoint(nn.Module):
            def __init__(self):
                super().__init__()
                # Simple CNN for feature detection
                self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 256, 3, padding=1)
                
            def forward(self, x):
                if x.shape[1] == 3:
                    x = torch.mean(x, dim=1, keepdim=True)  # Convert to grayscale
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.conv3(x)
                return x
        
        return SimpleSuperPoint().eval().to(self.device)
    
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
                    pred = self.model(image_tensor)
                
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
        
        return features


class ALIKEDExtractor(BaseFeatureExtractor):
    """ALIKED feature extractor"""
    
    def _setup_model(self):
        try:
            self.model = ALIKED(max_num_keypoints=2048).eval().to(self.device)
        except NameError:
            # Fallback implementation
            self.model = self._create_aliked_fallback()
    
    def _create_aliked_fallback(self):
        """Fallback ALIKED implementation"""
        # Simplified ALIKED implementation
        class SimpleALIKED(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                
            def forward(self, x):
                if x.shape[1] == 3:
                    x = torch.mean(x, dim=1, keepdim=True)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.conv3(x)
                return x
        
        return SimpleALIKED().eval().to(self.device)
    
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
                    pred = self.model(image_tensor)
                
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
        
        return features


class DISKExtractor(BaseFeatureExtractor):
    """DISK feature extractor"""
    
    def _setup_model(self):
        try:
            self.model = DISK(max_num_keypoints=2048).eval().to(self.device)
        except NameError:
            # Fallback implementation
            self.model = self._create_disk_fallback()
    
    def _create_disk_fallback(self):
        """Fallback DISK implementation"""
        # Simplified DISK implementation
        class SimpleDISK(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                
            def forward(self, x):
                if x.shape[1] == 3:
                    x = torch.mean(x, dim=1, keepdim=True)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.conv3(x)
                return x
        
        return SimpleDISK().eval().to(self.device)
    
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
                    pred = self.model(image_tensor)
                
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
        
        return features


class FeatureExtractorFactory:
    """Factory for creating feature extractors"""
    
    _extractors = {
        'superpoint': SuperPointExtractor,
        'aliked': ALIKEDExtractor,
        'disk': DISKExtractor
    }
    
    @classmethod
    def create(cls, extractor_type: str, device: torch.device) -> BaseFeatureExtractor:
        """Create a feature extractor instance"""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        return cls._extractors[extractor_type](device)
    
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
        self.max_keypoints = config.get('max_keypoints', 2048)
        self.max_image_size = config.get('max_image_size', 1600)
        
        # Create the actual extractor
        self.extractor = FeatureExtractorFactory.create(self.extractor_type, self.device)
    
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