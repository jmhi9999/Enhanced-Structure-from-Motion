"""
Robust SfM pipeline with automatic fallback mechanisms and quality assessment
Designed to be universally good across different input types
"""

import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import traceback

from .adaptive_parameters import AdaptiveParameterManager, ImageCharacteristics, ParameterSet, create_adaptive_config
from .geometric_verification import GeometricVerification, RANSACMethod
from .feature_matcher import EnhancedLightGlueMatcher

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution with quality metrics"""
    success: bool
    sparse_points: Optional[Any] = None
    cameras: Optional[Any] = None
    images: Optional[Any] = None
    features: Optional[Dict] = None
    matches: Optional[Dict] = None
    
    # Quality metrics
    num_reconstructed_images: int = 0
    num_3d_points: int = 0
    avg_reprojection_error: float = 0.0
    reconstruction_coverage: float = 0.0  # % of images successfully reconstructed
    processing_time: float = 0.0
    
    # Feedback for learning
    success_rate: float = 0.0
    avg_matches_per_pair: float = 0.0
    reconstruction_quality: float = 0.0
    
    error_message: str = ""


class QualityAssessor:
    """Assesses the quality of reconstruction results"""
    
    @staticmethod
    def assess_matches(matches: Dict) -> Dict[str, float]:
        """Assess quality of feature matches"""
        if not matches:
            return {'match_success_rate': 0.0, 'avg_matches_per_pair': 0.0}
        
        total_pairs_attempted = len(matches)
        successful_pairs = sum(1 for m in matches.values() if len(m.get('matches0', [])) > 0)
        match_counts = [len(m.get('matches0', [])) for m in matches.values()]
        
        success_rate = successful_pairs / total_pairs_attempted if total_pairs_attempted > 0 else 0.0
        avg_matches = np.mean(match_counts) if match_counts else 0.0
        
        return {
            'match_success_rate': success_rate,
            'avg_matches_per_pair': avg_matches,
            'total_pairs': total_pairs_attempted,
            'successful_pairs': successful_pairs
        }
    
    @staticmethod
    def assess_reconstruction(sparse_points, cameras, images, input_image_count: int) -> Dict[str, float]:
        """Assess quality of 3D reconstruction"""
        if not sparse_points or not cameras or not images:
            return {
                'reconstruction_coverage': 0.0,
                'points_per_image': 0.0,
                'avg_track_length': 0.0,
                'reconstruction_quality': 0.0
            }
        
        num_points = len(sparse_points)
        num_cameras = len(cameras)
        num_images = len(images)
        
        # Coverage: how many images were successfully reconstructed
        coverage = num_images / input_image_count if input_image_count > 0 else 0.0
        
        # Points per image
        points_per_image = num_points / num_images if num_images > 0 else 0.0
        
        # Estimate track length (simplified)
        avg_track_length = 2.5  # placeholder - would need actual track analysis
        
        # Overall reconstruction quality score
        quality_score = min(1.0, coverage * 0.5 + min(1.0, points_per_image / 1000) * 0.3 + min(1.0, avg_track_length / 3) * 0.2)
        
        return {
            'reconstruction_coverage': coverage,
            'points_per_image': points_per_image,
            'avg_track_length': avg_track_length,
            'reconstruction_quality': quality_score,
            'num_3d_points': num_points,
            'num_cameras': num_cameras,
            'num_reconstructed_images': num_images
        }


class RobustSfMPipeline:
    """
    Robust SfM pipeline that adapts to different input characteristics
    and automatically falls back to alternative strategies when needed
    """
    
    def __init__(self, device, config: Dict[str, Any] = None):
        self.device = device
        self.config = config or {}
        self.parameter_manager = AdaptiveParameterManager(config)
        self.quality_assessor = QualityAssessor()
        
        # Pipeline components (will be initialized adaptively)
        self.current_params = None
        self.geometric_verifier = None
        self.feature_matcher = None
        
        # Execution history
        self.execution_history = []
        
        logger.info("Robust SfM pipeline initialized")
    
    def execute(self, features: Dict[str, Any], image_paths: List[str], 
                output_path: Path, semantic_masks: Dict = None) -> PipelineResult:
        """
        Execute the robust SfM pipeline with adaptive parameters and fallbacks
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("EXECUTING ROBUST SFM PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Analyze input characteristics
            characteristics = self.parameter_manager.analyze_image_characteristics(features, image_paths)
            
            # Step 2: Select initial parameters
            self.current_params = self.parameter_manager.select_initial_parameters(characteristics)
            
            # Step 3: Try main pipeline with adaptive fallbacks
            result = self._execute_with_fallbacks(features, image_paths, output_path, semantic_masks)
            
            # Step 4: Record performance for learning
            if result.success:
                feedback = {
                    'success_rate': result.success_rate,
                    'avg_matches_per_pair': result.avg_matches_per_pair,
                    'reconstruction_quality': result.reconstruction_quality,
                    'processing_time': result.processing_time
                }
                self.parameter_manager.record_performance(self.current_params, feedback)
            
            return result
            
        except Exception as e:
            logger.error(f"Critical pipeline failure: {e}")
            logger.error(traceback.format_exc())
            
            result = PipelineResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )
            return result
    
    def _execute_with_fallbacks(self, features: Dict[str, Any], image_paths: List[str],
                               output_path: Path, semantic_masks: Dict = None,
                               max_attempts: int = 4) -> PipelineResult:
        """Execute pipeline with automatic fallback mechanisms"""
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Pipeline attempt {attempt + 1}/{max_attempts}")
                
                # Get parameters for this attempt
                if attempt == 0:
                    params = self.current_params
                else:
                    params = self.parameter_manager.get_fallback_parameters(self.current_params, attempt)
                    logger.info(f"Using fallback parameters: MAGSAC threshold={params.magsac_threshold:.1f}, "
                               f"min_matches={params.min_matches}")
                
                # Execute single attempt
                result = self._execute_single_attempt(features, image_paths, output_path, 
                                                    semantic_masks, params)
                
                if result.success:
                    logger.info(f"✅ Pipeline succeeded on attempt {attempt + 1}")
                    return result
                else:
                    last_error = result.error_message
                    logger.warning(f"❌ Attempt {attempt + 1} failed: {result.error_message}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Attempt {attempt + 1} crashed: {e}")
                continue
        
        # All attempts failed
        logger.error(f"🚫 All {max_attempts} attempts failed. Last error: {last_error}")
        
        return PipelineResult(
            success=False,
            error_message=f"All {max_attempts} attempts failed. Last error: {last_error}"
        )
    
    def _execute_single_attempt(self, features: Dict[str, Any], image_paths: List[str],
                               output_path: Path, semantic_masks: Dict = None,
                               params: ParameterSet = None) -> PipelineResult:
        """Execute a single pipeline attempt with given parameters"""
        
        start_time = time.time()
        params = params or self.current_params
        
        try:
            # Step 1: Feature matching with adaptive parameters
            logger.info("Step 1: Adaptive feature matching...")
            matches = self._robust_feature_matching(features, semantic_masks, params)
            
            if not matches:
                return PipelineResult(
                    success=False,
                    error_message="No feature matches found"
                )
            
            # Assess match quality
            match_quality = self.quality_assessor.assess_matches(matches)
            logger.info(f"Match quality: {match_quality['match_success_rate']:.1%} success rate, "
                       f"{match_quality['avg_matches_per_pair']:.1f} avg matches/pair")
            
            # Step 2: Geometric verification with multiple fallbacks
            logger.info("Step 2: Robust geometric verification...")
            verified_matches = self._robust_geometric_verification(matches, features, params)
            
            if not verified_matches:
                return PipelineResult(
                    success=False,
                    error_message="No matches passed geometric verification"
                )
            
            verified_quality = self.quality_assessor.assess_matches(verified_matches)
            logger.info(f"Verified matches: {len(verified_matches)} pairs, "
                       f"{verified_quality['avg_matches_per_pair']:.1f} avg matches/pair")
            
            # Step 3: COLMAP reconstruction with fallbacks
            logger.info("Step 3: Robust COLMAP reconstruction...")
            sparse_points, cameras, images = self._robust_colmap_reconstruction(
                features, verified_matches, output_path, image_paths, params
            )
            
            if not sparse_points or not cameras or not images:
                return PipelineResult(
                    success=False,
                    error_message="COLMAP reconstruction failed"
                )
            
            # Step 4: Quality assessment
            reconstruction_quality = self.quality_assessor.assess_reconstruction(
                sparse_points, cameras, images, len(image_paths)
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Reconstruction quality: {reconstruction_quality['reconstruction_quality']:.2f}")
            logger.info(f"Coverage: {reconstruction_quality['reconstruction_coverage']:.1%}")
            logger.info(f"3D points: {reconstruction_quality['num_3d_points']}")
            
            # Create successful result
            result = PipelineResult(
                success=True,
                sparse_points=sparse_points,
                cameras=cameras,
                images=images,
                features=features,
                matches=verified_matches,
                processing_time=processing_time,
                **reconstruction_quality,
                success_rate=verified_quality['match_success_rate'],
                avg_matches_per_pair=verified_quality['avg_matches_per_pair']
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline step failed: {e}"
            logger.error(error_msg)
            return PipelineResult(
                success=False,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def _robust_feature_matching(self, features: Dict[str, Any], semantic_masks: Dict = None,
                                params: ParameterSet = None) -> Dict:
        """Robust feature matching with multiple strategies"""
        
        # Configure matcher with adaptive parameters
        matcher_config = {
            'confidence_threshold': params.confidence_threshold,
            'min_matches': params.min_matches,
            'use_semantic_filtering': semantic_masks is not None,
            'semantic_consistency_threshold': params.semantic_consistency_threshold,
            'semantic_min_matches': params.semantic_min_matches,
            'use_brute_force': True,  # Prefer brute force for robustness
            'use_vocabulary_tree': len(features) > 100,  # Use vocab tree for large datasets
        }
        
        try:
            # Primary matching strategy
            matcher = EnhancedLightGlueMatcher(
                device=self.device,
                feature_type=self.config.get('feature_type', 'superpoint'),
                config=matcher_config
            )
            
            matches = matcher.match_features(features, semantic_masks)
            
            # Clean up matcher memory
            if hasattr(matcher, 'clear_memory'):
                matcher.clear_memory()
            
            return matches
            
        except Exception as e:
            logger.warning(f"Primary matching failed: {e}")
            
            # Fallback: try without semantic filtering
            if semantic_masks is not None:
                logger.info("Trying fallback without semantic filtering...")
                try:
                    matcher_config['use_semantic_filtering'] = False
                    matcher = EnhancedLightGlueMatcher(
                        device=self.device,
                        feature_type=self.config.get('feature_type', 'superpoint'),
                        config=matcher_config
                    )
                    
                    matches = matcher.match_features(features)
                    
                    if hasattr(matcher, 'clear_memory'):
                        matcher.clear_memory()
                    
                    return matches
                    
                except Exception as e2:
                    logger.error(f"Fallback matching also failed: {e2}")
            
            return {}
    
    def _robust_geometric_verification(self, matches: Dict, features: Dict,
                                     params: ParameterSet = None) -> Dict:
        """Robust geometric verification with multiple RANSAC strategies"""
        
        strategies = [
            # Strategy 1: Adaptive MAGSAC
            {
                'method': RANSACMethod.OPENCV_MAGSAC,
                'threshold': params.magsac_threshold,
                'confidence': params.magsac_confidence,
                'max_iterations': params.magsac_max_iters,
                'min_matches': params.min_matches
            },
            # Strategy 2: More permissive MAGSAC
            {
                'method': RANSACMethod.OPENCV_MAGSAC,
                'threshold': params.magsac_threshold * 1.5,
                'confidence': max(0.9, params.magsac_confidence - 0.05),
                'max_iterations': params.magsac_max_iters,
                'min_matches': max(6, params.min_matches - 2)
            },
            # Strategy 3: Very permissive MAGSAC
            {
                'method': RANSACMethod.OPENCV_MAGSAC,
                'threshold': params.magsac_threshold * 2.0,
                'confidence': 0.9,
                'max_iterations': max(500, params.magsac_max_iters // 2),
                'min_matches': max(4, params.min_matches - 4)
            }
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Geometric verification strategy {i+1}: "
                           f"threshold={strategy['threshold']:.1f}, min_matches={strategy['min_matches']}")
                
                verifier = GeometricVerification(
                    method=strategy['method'],
                    threshold=strategy['threshold'],
                    confidence=strategy['confidence'],
                    max_iterations=strategy['max_iterations'],
                    min_matches=strategy['min_matches']
                )
                
                verified_matches = verifier.verify(matches, features)
                
                if verified_matches and len(verified_matches) > 0:
                    success_rate = len(verified_matches) / len(matches) if matches else 0
                    logger.info(f"✅ Strategy {i+1} succeeded: {len(verified_matches)} pairs "
                               f"({success_rate:.1%} success rate)")
                    return verified_matches
                else:
                    logger.warning(f"❌ Strategy {i+1} failed: no verified matches")
                    
            except Exception as e:
                logger.warning(f"❌ Strategy {i+1} crashed: {e}")
                continue
        
        logger.error("All geometric verification strategies failed")
        return {}
    
    def _robust_colmap_reconstruction(self, features: Dict, matches: Dict, output_path: Path,
                                    image_paths: List[str], params: ParameterSet = None) -> Tuple:
        """Robust COLMAP reconstruction with fallback strategies"""
        
        try:
            # Primary strategy: COLMAP binary
            from .colmap_binary import colmap_binary_reconstruction
            
            first_image_path = Path(next(iter(features.keys())))
            image_dir = first_image_path.parent
            
            sparse_points, cameras, images = colmap_binary_reconstruction(
                features=features,
                matches=matches,
                output_path=output_path,
                image_dir=image_dir
            )
            
            if sparse_points and cameras and images:
                logger.info("✅ COLMAP binary reconstruction succeeded")
                return sparse_points, cameras, images
            
        except Exception as e:
            logger.warning(f"COLMAP binary reconstruction failed: {e}")
        
        try:
            # Fallback strategy: COLMAP wrapper
            from .colmap_wrapper import safe_colmap_reconstruction
            
            sparse_points, cameras, images = safe_colmap_reconstruction(
                features=features,
                matches=matches,
                output_path=output_path,
                image_dir=Path(next(iter(features.keys()))).parent,
                device=self.device
            )
            
            if sparse_points and cameras and images:
                logger.info("✅ COLMAP wrapper reconstruction succeeded")
                return sparse_points, cameras, images
                
        except Exception as e:
            logger.warning(f"COLMAP wrapper reconstruction failed: {e}")
        
        logger.error("All COLMAP reconstruction strategies failed")
        return None, None, None


def run_robust_pipeline(features: Dict[str, Any], image_paths: List[str],
                       output_path: Path, device, config: Dict[str, Any] = None,
                       semantic_masks: Dict = None) -> PipelineResult:
    """
    Convenience function to run the robust SfM pipeline
    """
    pipeline = RobustSfMPipeline(device, config)
    return pipeline.execute(features, image_paths, output_path, semantic_masks)