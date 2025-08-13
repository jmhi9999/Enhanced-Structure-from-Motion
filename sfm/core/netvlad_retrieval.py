"""
NetVLAD-based image retrieval for enhanced pair selection
Provides global image descriptors for better place recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import pickle
import hashlib

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    
    def __init__(self, num_clusters=64, dim=512, alpha=100.0):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)
        
        self._init_params()
    
    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )
    
    def forward(self, x):
        N, C = x.shape[:2]
        
        # Soft assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        # Calculate residuals to each centroid
        x_flatten = x.view(N, C, -1)
        
        # Broadcast and calculate residuals
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                  self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        
        # Weight residuals by soft assignment
        weighted_residual = residual * soft_assign.unsqueeze(2)
        
        # Sum over spatial locations
        vlad = weighted_residual.sum(dim=-1)
        
        # Intra-normalization
        vlad = F.normalize(vlad, p=2, dim=2)
        
        # Flatten and L2 normalize
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad


class NetVLADRetrieval:
    """NetVLAD-based image retrieval system"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None, output_path: Optional[str] = None):
        self.device = device
        self.config = config or {}
        self.output_path = Path(output_path) if output_path else Path(".")
        
        # NetVLAD parameters
        self.num_clusters = self.config.get('netvlad_clusters', 64)
        self.feature_dim = self.config.get('feature_dim', 512)
        
        # Initialize NetVLAD
        self.netvlad = NetVLAD(
            num_clusters=self.num_clusters,
            dim=self.feature_dim
        ).to(device)
        
        # Feature extractor (using ResNet backbone)
        self.backbone = self._init_backbone()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Storage
        self.image_descriptors = {}
        self.descriptor_matrix = None
        self.image_paths = []
        
        # FAISS index for fast retrieval
        self.use_faiss = FAISS_AVAILABLE and torch.cuda.is_available()
        self.faiss_index = None
        
        logger.info(f"NetVLAD retrieval initialized with {self.num_clusters} clusters")
    
    def _init_backbone(self):
        """Initialize feature extraction backbone"""
        try:
            import torchvision.models as models
            
            # Use ResNet50 as backbone
            backbone = models.resnet50(pretrained=True)
            
            # Remove final layers (keep up to conv features)
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone = backbone.to(self.device)
            backbone.eval()
            
            return backbone
            
        except ImportError:
            logger.warning("torchvision not available, using simple CNN backbone")
            return self._create_simple_backbone()
    
    def _create_simple_backbone(self):
        """Create simple CNN backbone as fallback"""
        backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
        ).to(self.device)
        return backbone
    
    def extract_global_descriptor(self, image_path: str) -> torch.Tensor:
        """Extract NetVLAD global descriptor from image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Extract CNN features
                cnn_features = self.backbone(image_tensor)
                
                # Apply NetVLAD
                global_descriptor = self.netvlad(cnn_features)
                
            return global_descriptor.squeeze(0).cpu()
            
        except Exception as e:
            logger.warning(f"Failed to extract NetVLAD descriptor for {image_path}: {e}")
            return torch.zeros(self.num_clusters * self.feature_dim)
    
    def build_database(self, image_paths: List[str], force_rebuild: bool = False):
        """Build NetVLAD descriptor database"""
        cache_path = self.output_path / "netvlad_cache.pkl"
        
        # Check cache
        if not force_rebuild and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    if set(cached_data['image_paths']) == set(image_paths):
                        self.image_descriptors = cached_data['descriptors']
                        self.image_paths = cached_data['image_paths']
                        self._build_retrieval_index()
                        logger.info("Loaded cached NetVLAD descriptors")
                        return
            except Exception as e:
                logger.warning(f"Failed to load NetVLAD cache: {e}")
        
        logger.info(f"Building NetVLAD database for {len(image_paths)} images...")
        
        self.image_descriptors = {}
        descriptors = []
        valid_paths = []
        
        for image_path in tqdm(image_paths, desc="Extracting NetVLAD descriptors"):
            descriptor = self.extract_global_descriptor(image_path)
            if descriptor is not None and not torch.isnan(descriptor).any():
                self.image_descriptors[image_path] = descriptor
                descriptors.append(descriptor.numpy())
                valid_paths.append(image_path)
        
        self.image_paths = valid_paths
        
        if descriptors:
            self.descriptor_matrix = np.vstack(descriptors)
            self._build_retrieval_index()
            
            # Cache results
            self._cache_descriptors(cache_path, valid_paths)
            
            logger.info(f"Built NetVLAD database with {len(descriptors)} descriptors")
        else:
            logger.warning("No valid NetVLAD descriptors extracted")
    
    def _build_retrieval_index(self):
        """Build FAISS index for fast retrieval"""
        if not self.use_faiss or self.descriptor_matrix is None:
            return
        
        try:
            d = self.descriptor_matrix.shape[1]
            
            # Use cosine similarity (inner product on normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(d)
            
            # Normalize descriptors for cosine similarity
            normalized_descriptors = self.descriptor_matrix.copy()
            faiss.normalize_L2(normalized_descriptors)
            
            self.faiss_index.add(normalized_descriptors.astype(np.float32))
            
            logger.info("Built FAISS index for NetVLAD retrieval")
            
        except Exception as e:
            logger.warning(f"Failed to build FAISS index: {e}")
            self.faiss_index = None
    
    def query_similar_images(self, query_image_path: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Query similar images using NetVLAD descriptors"""
        if query_image_path not in self.image_descriptors:
            # Extract descriptor for new image
            query_descriptor = self.extract_global_descriptor(query_image_path)
            if query_descriptor is None:
                return []
        else:
            query_descriptor = self.image_descriptors[query_image_path]
        
        query_np = query_descriptor.numpy().reshape(1, -1)
        
        # Use FAISS if available
        if self.faiss_index is not None and self.descriptor_matrix is not None:
            return self._faiss_search(query_np, top_k)
        else:
            return self._cosine_search(query_np, top_k)
    
    def _faiss_search(self, query_np: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """FAISS-based similarity search"""
        try:
            # Normalize query
            faiss.normalize_L2(query_np)
            
            # Search
            scores, indices = self.faiss_index.search(query_np.astype(np.float32), min(top_k, len(self.image_paths)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.image_paths):
                    results.append((self.image_paths[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.warning(f"FAISS search failed: {e}")
            return self._cosine_search(query_np, top_k)
    
    def _cosine_search(self, query_np: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Cosine similarity search fallback"""
        if self.descriptor_matrix is None or not SKLEARN_AVAILABLE:
            return []
        
        try:
            similarities = cosine_similarity(query_np, self.descriptor_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.image_paths):
                    results.append((self.image_paths[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.warning(f"Cosine search failed: {e}")
            return []
    
    def get_image_pairs_for_matching(self, image_paths: List[str], 
                                   max_pairs_per_image: int = 20,
                                   min_similarity: float = 0.3) -> List[Tuple[str, str]]:
        """Get image pairs using NetVLAD similarity"""
        if not self.image_descriptors:
            self.build_database(image_paths)
        
        pairs = set()
        
        for image_path in tqdm(image_paths, desc="NetVLAD pair selection"):
            similar_images = self.query_similar_images(image_path, max_pairs_per_image * 2)
            
            # Filter by similarity threshold
            valid_pairs = [(img, sim) for img, sim in similar_images 
                          if img != image_path and sim >= min_similarity]
            
            # Take top pairs
            for similar_img, _ in valid_pairs[:max_pairs_per_image]:
                pair = tuple(sorted([image_path, similar_img]))
                pairs.add(pair)
        
        logger.info(f"NetVLAD generated {len(pairs)} image pairs")
        return list(pairs)
    
    def _cache_descriptors(self, cache_path: Path, image_paths: List[str]):
        """Cache descriptors to disk"""
        try:
            cache_data = {
                'descriptors': self.image_descriptors,
                'image_paths': image_paths
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached NetVLAD descriptors to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache NetVLAD descriptors: {e}")