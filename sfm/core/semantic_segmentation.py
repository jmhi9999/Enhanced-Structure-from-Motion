"""
Semantic segmentation module for Enhanced SfM Pipeline
Provides semantic masks for image understanding and filtering
"""

import logging
import numpy as np
import torch
import gc
from PIL import Image
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from tqdm import tqdm

# Conditional imports for transformers
try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoImageProcessor = None
    AutoModelForSemanticSegmentation = None
    F = None

logger = logging.getLogger(__name__)


class SemanticSegmenter:
    """
    Semantic segmentation using HuggingFace transformers models
    Optimized for SegFormer models (nvidia/segformer-b0-finetuned-ade-512-512)
    """
    
    def __init__(self, 
                 model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                 device: Optional[torch.device] = None):
        """
        Initialize semantic segmenter
        
        Args:
            model_name: HuggingFace model name for semantic segmentation
            device: Torch device to use for inference
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers not available. Install with: pip install transformers>=4.20.0"
            )
        
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading semantic segmentation model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load processor and model
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get label information
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            self.num_labels = len(self.id2label)
            
            logger.info(f"Model loaded successfully with {self.num_labels} classes")
            
        except Exception as e:
            logger.error(f"Failed to load semantic segmentation model: {e}")
            raise
    
    def segment_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Segment a single image
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            
        Returns:
            Semantic mask as numpy array with class indices
        """
        # Convert input to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get predictions
        # Resize logits to original image size
        original_size = pil_image.size[::-1]  # (height, width)
        logits = F.interpolate(
            logits,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )
        
        # Get class predictions
        predicted_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
        
        return predicted_mask
    
    def segment_images_batch(self, 
                           image_paths: List[Union[str, Path]], 
                           batch_size: int = 4) -> Dict[str, np.ndarray]:
        """
        Segment multiple images in batches
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping image paths to semantic masks
        """
        results = {}
        
        logger.info(f"Segmenting {len(image_paths)} images with batch size {batch_size}")
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Semantic Segmentation"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_original_sizes = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                try:
                    pil_image = Image.open(img_path).convert("RGB")
                    batch_images.append(pil_image)
                    batch_original_sizes.append(pil_image.size[::-1])  # (height, width)
                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Process batch
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                
                # Process each image in the batch
                for j, (img_path, original_size) in enumerate(zip(batch_paths, batch_original_sizes)):
                    if j >= len(batch_images):
                        continue
                        
                    # Extract logits for this image
                    img_logits = logits[j:j+1]
                    
                    # Resize to original size
                    img_logits = F.interpolate(
                        img_logits,
                        size=original_size,
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # Get predictions
                    predicted_mask = img_logits.argmax(dim=1).squeeze().cpu().numpy()
                    results[str(img_path)] = predicted_mask
                    
                    # Clean up intermediate tensors
                    del img_logits
                
                # Clean up batch tensors
                del logits, outputs
                if 'inputs' in locals():
                    del inputs
                
                # Force garbage collection and clear CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"Failed to process batch starting at {batch_paths[0]}: {e}")
                # Fallback to individual processing
                for img_path in batch_paths:
                    try:
                        mask = self.segment_image(img_path)
                        results[str(img_path)] = mask
                    except Exception as e2:
                        logger.warning(f"Failed to process individual image {img_path}: {e2}")
        
        logger.info(f"Successfully segmented {len(results)}/{len(image_paths)} images")
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def save_masks(self, masks: Dict[str, np.ndarray], output_dir: str):
        """
        Save semantic masks to files
        
        Args:
            masks: Dictionary of masks from segment_images_batch
            output_dir: Output directory for mask files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(masks)} semantic masks to {output_dir}")
        
        for img_path, mask in tqdm(masks.items(), desc="Saving masks"):
            # Create output filename
            img_name = Path(img_path).name
            mask_path = output_path / f"{img_name}.png"
            
            # Convert mask to uint8 and save
            mask_uint8 = mask.astype(np.uint8)
            mask_image = Image.fromarray(mask_uint8)
            mask_image.save(mask_path)
    
    def get_label_info(self) -> Dict:
        """
        Get information about semantic labels
        
        Returns:
            Dictionary with label information
        """
        return {
            'num_labels': self.num_labels,
            'id2label': self.id2label,
            'label2id': self.label2id,
            'model_name': self.model_name
        }
    
    def clear_memory(self):
        """Clear GPU memory used by the segmentation model"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Semantic segmentation model memory cleared")
        except Exception as e:
            logger.warning(f"Error clearing semantic segmentation memory: {e}")
    
    def get_semantic_statistics(self, masks: Dict[str, np.ndarray]) -> Dict:
        """
        Get statistics about semantic segmentation results
        
        Args:
            masks: Dictionary of semantic masks
            
        Returns:
            Statistics dictionary
        """
        if not masks:
            return {}
        
        # Collect all unique labels across all masks
        all_labels = set()
        label_counts = {}
        
        for img_path, mask in masks.items():
            unique_labels = np.unique(mask)
            all_labels.update(unique_labels)
            
            for label in unique_labels:
                count = np.sum(mask == label)
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += count
        
        # Convert to readable format
        readable_counts = {}
        for label_id, count in label_counts.items():
            if label_id in self.id2label:
                label_name = self.id2label[label_id]
                readable_counts[label_name] = count
            else:
                readable_counts[f"unknown_{label_id}"] = count
        
        return {
            'total_images': len(masks),
            'unique_labels': len(all_labels),
            'label_counts': readable_counts,
            'most_common_label': max(readable_counts.items(), key=lambda x: x[1]) if readable_counts else None
        }


def create_semantic_segmenter(model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                            device: Optional[torch.device] = None) -> SemanticSegmenter:
    """
    Factory function to create a semantic segmenter
    
    Args:
        model_name: HuggingFace model name
        device: Torch device
        
    Returns:
        SemanticSegmenter instance
    """
    return SemanticSegmenter(model_name=model_name, device=device)


def segment_images(image_paths: List[Union[str, Path]], 
                  model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
                  batch_size: int = 4,
                  device: Optional[torch.device] = None) -> Dict[str, np.ndarray]:
    """
    Convenience function to segment multiple images
    
    Args:
        image_paths: List of image paths
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        device: Torch device
        
    Returns:
        Dictionary mapping image paths to semantic masks
    """
    segmenter = SemanticSegmenter(model_name=model_name, device=device)
    try:
        results = segmenter.segment_images_batch(image_paths, batch_size=batch_size)
        return results
    finally:
        # Always clean up memory
        segmenter.clear_memory()
        del segmenter