"""
Adaptive parameter system for robust SfM pipeline
Automatically adjusts parameters based on input characteristics and performance feedback
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ImageCharacteristics:
    """Characteristics of input images that affect SfM performance"""
    avg_resolution: Tuple[int, int]
    texture_score: float  # 0-1, higher = more textured
    brightness_variance: float
    blur_score: float  # 0-1, higher = more blurred
    num_images: int
    keypoint_density: float  # avg keypoints per image
    descriptor_variance: float  # variance in descriptor space


@dataclass
class ParameterSet:
    """Set of parameters for SfM pipeline components"""
    # Feature matching
    confidence_threshold: float = 0.1
    min_matches: int = 8
    
    # Geometric verification (MAGSAC)
    magsac_threshold: float = 3.0
    magsac_confidence: float = 0.99
    magsac_max_iters: int = 1000
    
    # Advanced verification
    fundamental_threshold: float = 1.0
    essential_threshold: float = 1.0
    
    # Semantic filtering
    semantic_consistency_threshold: float = 0.4
    semantic_min_matches: int = 6
    
    # Bundle adjustment
    ba_max_iterations: int = 200
    ba_convergence_tolerance: float = 1e-6
    
    # Quality control
    min_triangulation_angle: float = 2.0  # degrees
    max_reprojection_error: float = 4.0   # pixels
    min_track_length: int = 2


class AdaptiveParameterManager:
    """
    Manages adaptive parameter selection based on image characteristics
    and pipeline performance feedback
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.performance_history = []
        self.parameter_history = []
        
        # Default parameter sets for different scenarios
        self.parameter_presets = {
            'conservative': ParameterSet(
                confidence_threshold=0.2,
                min_matches=15,
                magsac_threshold=1.0,
                magsac_confidence=0.999,
                magsac_max_iters=2000,
                fundamental_threshold=0.75,
                essential_threshold=0.75,
                semantic_consistency_threshold=0.6,
                semantic_min_matches=10,
                ba_max_iterations=300,
                min_triangulation_angle=3.0,
                max_reprojection_error=2.0,
                min_track_length=3
            ),
            'balanced': ParameterSet(
                confidence_threshold=0.1,
                min_matches=8,
                magsac_threshold=3.0,
                magsac_confidence=0.99,
                magsac_max_iters=1000,
                fundamental_threshold=1.0,
                essential_threshold=1.0,
                semantic_consistency_threshold=0.4,
                semantic_min_matches=6,
                ba_max_iterations=200,
                min_triangulation_angle=2.0,
                max_reprojection_error=4.0,
                min_track_length=2
            ),
            'permissive': ParameterSet(
                confidence_threshold=0.05,
                min_matches=4,
                magsac_threshold=5.0,
                magsac_confidence=0.9,
                magsac_max_iters=500,
                fundamental_threshold=2.0,
                essential_threshold=2.0,
                semantic_consistency_threshold=0.2,
                semantic_min_matches=4,
                ba_max_iterations=100,
                min_triangulation_angle=1.0,
                max_reprojection_error=8.0,
                min_track_length=2
            ),
            'experimental': ParameterSet(
                confidence_threshold=0.01,
                min_matches=3,
                magsac_threshold=8.0,
                magsac_confidence=0.85,
                magsac_max_iters=300,
                fundamental_threshold=3.0,
                essential_threshold=3.0,
                semantic_consistency_threshold=0.1,
                semantic_min_matches=3,
                ba_max_iterations=50,
                min_triangulation_angle=0.5,
                max_reprojection_error=12.0,
                min_track_length=2
            )
        }
        
        # Load historical performance data if available
        self.load_performance_history()
    
    def analyze_image_characteristics(self, features: Dict[str, Any], 
                                    image_paths: List[str] = None) -> ImageCharacteristics:
        """Analyze characteristics of input images"""
        logger.info("Analyzing image characteristics for adaptive parameters...")
        
        # Calculate image resolution
        resolutions = []
        keypoint_counts = []
        descriptor_variances = []
        
        for img_path, feat_data in features.items():
            if 'image_shape' in feat_data:
                resolutions.append(feat_data['image_shape'])
            
            if 'keypoints' in feat_data:
                keypoint_counts.append(len(feat_data['keypoints']))
            
            if 'descriptors' in feat_data:
                desc_var = np.var(feat_data['descriptors'])
                descriptor_variances.append(desc_var)
        
        # Calculate averages
        if resolutions:
            avg_resolution = tuple(np.mean(resolutions, axis=0).astype(int))
        else:
            avg_resolution = (1024, 768)  # default
        
        avg_keypoints = np.mean(keypoint_counts) if keypoint_counts else 1000
        avg_desc_variance = np.mean(descriptor_variances) if descriptor_variances else 1.0
        
        # Estimate texture score based on keypoint density
        image_area = avg_resolution[0] * avg_resolution[1]
        keypoint_density = avg_keypoints / image_area
        texture_score = min(1.0, keypoint_density * 1000000)  # normalize
        
        # Simple heuristics for other scores (can be improved with actual image analysis)
        brightness_variance = 0.5  # placeholder
        blur_score = max(0.0, min(1.0, 1.0 - texture_score))  # inverse of texture
        
        characteristics = ImageCharacteristics(
            avg_resolution=avg_resolution,
            texture_score=texture_score,
            brightness_variance=brightness_variance,
            blur_score=blur_score,
            num_images=len(features),
            keypoint_density=keypoint_density,
            descriptor_variance=avg_desc_variance
        )
        
        logger.info(f"Image characteristics: {characteristics}")
        return characteristics
    
    def select_initial_parameters(self, characteristics: ImageCharacteristics) -> ParameterSet:
        """Select initial parameters based on image characteristics"""
        
        # Decision tree based on characteristics
        if characteristics.num_images < 10:
            # Small dataset - be more permissive
            if characteristics.texture_score < 0.3:
                preset = 'experimental'
            else:
                preset = 'permissive'
        elif characteristics.num_images > 100:
            # Large dataset - be more conservative
            if characteristics.texture_score > 0.7:
                preset = 'conservative'
            else:
                preset = 'balanced'
        else:
            # Medium dataset
            if characteristics.blur_score > 0.5:
                preset = 'permissive'
            elif characteristics.texture_score > 0.6:
                preset = 'conservative'
            else:
                preset = 'balanced'
        
        params = self.parameter_presets[preset]
        logger.info(f"Selected initial parameter preset: {preset}")
        logger.info(f"MAGSAC threshold: {params.magsac_threshold}, min_matches: {params.min_matches}")
        
        return params
    
    def adapt_parameters_from_feedback(self, current_params: ParameterSet,
                                     feedback: Dict[str, Any]) -> ParameterSet:
        """Adapt parameters based on pipeline performance feedback"""
        
        new_params = ParameterSet(**current_params.__dict__)
        
        # Extract feedback metrics
        success_rate = feedback.get('success_rate', 0.0)
        avg_matches = feedback.get('avg_matches_per_pair', 0)
        reconstruction_quality = feedback.get('reconstruction_quality', 0.0)
        processing_time = feedback.get('processing_time', 0.0)
        
        logger.info(f"Adapting parameters: success_rate={success_rate:.2f}, "
                   f"avg_matches={avg_matches:.1f}, quality={reconstruction_quality:.2f}")
        
        # Adaptation rules
        if success_rate < 0.3:
            # Very low success rate - make parameters more permissive
            new_params.magsac_threshold = min(8.0, current_params.magsac_threshold * 1.5)
            new_params.min_matches = max(3, current_params.min_matches - 2)
            new_params.confidence_threshold = max(0.01, current_params.confidence_threshold * 0.7)
            new_params.magsac_confidence = max(0.85, current_params.magsac_confidence - 0.05)
            logger.info("Low success rate detected - making parameters more permissive")
            
        elif success_rate > 0.8 and reconstruction_quality < 0.5:
            # High success but low quality - make parameters more conservative
            new_params.magsac_threshold = max(1.0, current_params.magsac_threshold * 0.8)
            new_params.min_matches = min(20, current_params.min_matches + 3)
            new_params.confidence_threshold = min(0.3, current_params.confidence_threshold * 1.2)
            new_params.magsac_confidence = min(0.999, current_params.magsac_confidence + 0.02)
            logger.info("High success but low quality - making parameters more conservative")
            
        elif processing_time > 300:  # 5 minutes
            # Too slow - reduce iterations and relax some parameters
            new_params.magsac_max_iters = max(200, current_params.magsac_max_iters - 200)
            new_params.ba_max_iterations = max(50, current_params.ba_max_iterations - 50)
            logger.info("Processing too slow - reducing iterations")
        
        # Clamp all parameters to reasonable ranges
        new_params = self._clamp_parameters(new_params)
        
        return new_params
    
    def _clamp_parameters(self, params: ParameterSet) -> ParameterSet:
        """Clamp parameters to reasonable ranges"""
        params.confidence_threshold = max(0.001, min(0.5, params.confidence_threshold))
        params.min_matches = max(3, min(30, params.min_matches))
        params.magsac_threshold = max(0.5, min(10.0, params.magsac_threshold))
        params.magsac_confidence = max(0.8, min(0.9999, params.magsac_confidence))
        params.magsac_max_iters = max(100, min(5000, params.magsac_max_iters))
        params.fundamental_threshold = max(0.5, min(5.0, params.fundamental_threshold))
        params.essential_threshold = max(0.5, min(5.0, params.essential_threshold))
        params.semantic_consistency_threshold = max(0.0, min(1.0, params.semantic_consistency_threshold))
        params.semantic_min_matches = max(3, min(20, params.semantic_min_matches))
        params.ba_max_iterations = max(50, min(500, params.ba_max_iterations))
        params.min_triangulation_angle = max(0.1, min(10.0, params.min_triangulation_angle))
        params.max_reprojection_error = max(1.0, min(20.0, params.max_reprojection_error))
        params.min_track_length = max(2, min(10, params.min_track_length))
        
        return params
    
    def get_fallback_parameters(self, current_params: ParameterSet,
                              attempt: int = 1) -> ParameterSet:
        """Get increasingly permissive fallback parameters"""
        
        fallback_params = ParameterSet(**current_params.__dict__)
        
        # Apply increasingly permissive changes based on attempt number
        for i in range(attempt):
            fallback_params.magsac_threshold *= 1.5
            fallback_params.min_matches = max(3, fallback_params.min_matches - 2)
            fallback_params.confidence_threshold *= 0.7
            fallback_params.magsac_confidence = max(0.85, fallback_params.magsac_confidence - 0.05)
            fallback_params.fundamental_threshold *= 1.3
            fallback_params.essential_threshold *= 1.3
        
        fallback_params = self._clamp_parameters(fallback_params)
        
        logger.info(f"Fallback attempt {attempt}: threshold={fallback_params.magsac_threshold:.1f}, "
                   f"min_matches={fallback_params.min_matches}")
        
        return fallback_params
    
    def record_performance(self, params: ParameterSet, feedback: Dict[str, Any]):
        """Record performance for learning"""
        performance_record = {
            'timestamp': time.time(),
            'parameters': params.__dict__,
            'feedback': feedback
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Save to disk
        self.save_performance_history()
    
    def load_performance_history(self):
        """Load historical performance data"""
        try:
            history_file = Path('adaptive_parameters_history.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance records")
        except Exception as e:
            logger.warning(f"Could not load performance history: {e}")
            self.performance_history = []
    
    def save_performance_history(self):
        """Save performance history to disk"""
        try:
            history_file = Path('adaptive_parameters_history.json')
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save performance history: {e}")
    
    def get_best_parameters_for_characteristics(self, 
                                              characteristics: ImageCharacteristics) -> Optional[ParameterSet]:
        """Find best parameters from history for similar characteristics"""
        
        if not self.performance_history:
            return None
        
        # Simple similarity matching (can be improved)
        best_params = None
        best_score = -1
        
        for record in self.performance_history:
            feedback = record['feedback']
            success_rate = feedback.get('success_rate', 0.0)
            quality = feedback.get('reconstruction_quality', 0.0)
            
            # Combined score
            score = (success_rate * 0.7 + quality * 0.3)
            
            if score > best_score:
                best_score = score
                best_params = ParameterSet(**record['parameters'])
        
        if best_params and best_score > 0.5:
            logger.info(f"Found good historical parameters with score {best_score:.2f}")
            return best_params
        
        return None


def create_adaptive_config(characteristics: ImageCharacteristics) -> Dict[str, Any]:
    """Create adaptive configuration for the entire pipeline"""
    
    manager = AdaptiveParameterManager()
    params = manager.select_initial_parameters(characteristics)
    
    # Convert to pipeline configuration
    config = {
        # Feature matching
        'confidence_threshold': params.confidence_threshold,
        'min_matches': params.min_matches,
        
        # Geometric verification
        'magsac_threshold': params.magsac_threshold,
        'magsac_confidence': params.magsac_confidence,
        'magsac_max_iters': params.magsac_max_iters,
        'fundamental_threshold': params.fundamental_threshold,
        'essential_threshold': params.essential_threshold,
        
        # Semantic filtering
        'semantic_consistency_threshold': params.semantic_consistency_threshold,
        'semantic_min_matches': params.semantic_min_matches,
        
        # Bundle adjustment
        'ba_max_iterations': params.ba_max_iterations,
        'ba_convergence_tolerance': params.ba_convergence_tolerance,
        
        # Quality control
        'min_triangulation_angle': params.min_triangulation_angle,
        'max_reprojection_error': params.max_reprojection_error,
        'min_track_length': params.min_track_length,
        
        # Enable adaptive mode
        'adaptive_mode': True,
        'parameter_manager': manager
    }
    
    return config