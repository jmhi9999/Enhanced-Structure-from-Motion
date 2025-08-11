
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class SemanticSegmenter:
    """
    A class to handle semantic segmentation of images using a pre-trained SegFormer model.
    """
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device="cuda"):
        """
        Initializes the SemanticSegmenter.

        Args:
            model_name (str): The name of the pre-trained model to use from Hugging Face.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SemanticSegmenter on device: {self.device}")
        
        try:
            logger.info(f"Loading semantic segmentation model: {model_name}")
            self.processor = SegformerImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
            logger.info(f"Successfully loaded model '{model_name}'")
        except Exception as e:
            # Try fallback models if the primary model fails
            fallback_models = [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "nvidia/mit-b0",
                "facebook/detr-resnet-50-panoptic"  # Last resort fallback
            ]
            
            for fallback_model in fallback_models:
                if fallback_model != model_name:  # Don't retry the same model
                    try:
                        logger.warning(f"Trying fallback model: {fallback_model}")
                        self.processor = SegformerImageProcessor.from_pretrained(fallback_model)
                        self.model = SegformerForSemanticSegmentation.from_pretrained(fallback_model).to(self.device).eval()
                        logger.info(f"Successfully loaded fallback model '{fallback_model}'")
                        return
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                        continue
            
            logger.error(f"Failed to load semantic segmentation model and all fallbacks: {e}")
            raise Exception("No semantic segmentation model could be loaded")

    def segment_image(self, image_path: str) -> np.ndarray:
        """
        Performs semantic segmentation on a single image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            np.ndarray: A 2D numpy array representing the semantic map.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits.cpu()
            # Upsample logits to the original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
            return pred_seg
        except Exception as e:
            logger.error(f"Failed to segment image {image_path}: {e}")
            return None

    def segment_images_batch(self, image_paths: list[str], batch_size: int = 4) -> dict[str, np.ndarray]:
        """
        Performs semantic segmentation on a batch of images.

        Args:
            image_paths (list[str]): A list of paths to the input images.
            batch_size (int): The number of images to process in a single batch.

        Returns:
            dict[str, np.ndarray]: A dictionary mapping image paths to their semantic maps.
        """
        results = {}
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Segmenting {len(image_paths)} images ({total_batches} batches)"):
            batch_paths = image_paths[i:i+batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            try:
                # Process images with their original sizes
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                logits = outputs.logits.cpu()

                # Upsample and find argmax for each image in the batch
                for j, image in enumerate(images):
                    upsampled_logits = torch.nn.functional.interpolate(
                        logits[j].unsqueeze(0),
                        size=image.size[::-1], # (height, width)
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
                    results[batch_paths[j]] = pred_seg
            except Exception as e:
                logger.error(f"Failed to process batch starting with {batch_paths[0]}: {e}")
                for path in batch_paths:
                    results[path] = None # Mark as failed

        # Log final statistics
        successful = sum(1 for v in results.values() if v is not None)
        failed = len(results) - successful
        logger.info(f"Semantic segmentation completed: {successful} successful, {failed} failed out of {len(image_paths)} images")
        
        return results

    def save_masks(self, masks: dict[str, np.ndarray], output_dir: str):
        """
        Saves semantic masks as PNG files.

        Args:
            masks (dict[str, np.ndarray]): Dictionary mapping image paths to semantic masks.
            output_dir (str): The directory to save the masks in.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for image_path, mask_array in tqdm(masks.items(), desc="Saving semantic masks"):
            if mask_array is not None:
                mask_img = Image.fromarray(mask_array)
                # Use the original image filename for the mask
                mask_filename = Path(image_path).name
                mask_img.save(output_path / f"{mask_filename}.png")

    def get_label_info(self) -> dict:
        """
        Returns information about the labels used by the model.

        Returns:
            dict: A dictionary containing label mappings.
        """
        return self.model.config.id2label
