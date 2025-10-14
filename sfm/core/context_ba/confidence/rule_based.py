"""
Rule-based confidence computation

Computes confidence scores using hand-crafted heuristics without any training.

Camera confidence factors:
1. Covisibility: How many other cameras share points
2. Match Quality: Average matching score from feature matcher
3. Feature Density: Number of keypoints detected
4. Spatial Uniformity: How evenly keypoints are distributed
5. Multi-hop Connectivity: Indirect connectivity (friends-of-friends)
6. Geometric Consistency: Inlier ratio from geometric verification

Point confidence factors:
1. Track Length: Number of cameras observing this point
2. Reprojection Error: Reprojection accuracy
3. Triangulation Angle: Angle between observing cameras
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

from .base import ConfidenceCalculator

logger = logging.getLogger(__name__)


class RuleBasedConfidence(ConfidenceCalculator):
    """
    Rule-based confidence computation without training

    Uses weighted combination of hand-crafted features.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: ContextBAConfig or None (uses default weights)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract weights from config
        if config is not None and hasattr(config, 'weights'):
            w = config.weights
            self.camera_weights = {
                'covisibility': w.covisibility,
                'match_quality': w.match_quality,
                'feature_density': w.feature_density,
                'spatial_uniformity': w.spatial_uniformity,
                'multi_hop_connectivity': w.multi_hop_connectivity,
                'geometric_consistency': w.geometric_consistency,
            }
            self.point_weights = {
                'track_length': w.track_length,
                'reprojection_error': w.reprojection_error,
                'triangulation_angle': w.triangulation_angle,
            }
        else:
            # Default weights
            self.camera_weights = {
                'covisibility': 0.25,
                'match_quality': 0.20,
                'feature_density': 0.15,
                'spatial_uniformity': 0.15,
                'multi_hop_connectivity': 0.15,
                'geometric_consistency': 0.10,
            }
            self.point_weights = {
                'track_length': 0.50,
                'reprojection_error': 0.30,
                'triangulation_angle': 0.20,
            }

    def compute_camera_confidence(
        self,
        camera_id: int,
        graph: Any,  # SceneGraph
        features: Dict[str, Any],
    ) -> float:
        """
        Compute camera confidence from rule-based features

        Returns:
            Confidence score in [0, 1]
        """
        camera = graph.cameras.get(camera_id)
        if camera is None:
            return 0.0

        # Extract features
        features_dict = self._extract_camera_features(camera, graph)

        # Weighted combination
        confidence = (
            self.camera_weights['covisibility'] * features_dict['covisibility'] +
            self.camera_weights['match_quality'] * features_dict['match_quality'] +
            self.camera_weights['feature_density'] * features_dict['feature_density'] +
            self.camera_weights['spatial_uniformity'] * features_dict['spatial_uniformity'] +
            self.camera_weights['multi_hop_connectivity'] * features_dict['multi_hop_connectivity'] +
            self.camera_weights['geometric_consistency'] * features_dict['geometric_consistency']
        )

        return np.clip(confidence, 0.0, 1.0)

    def _extract_camera_features(
        self,
        camera: Any,  # CameraNode
        graph: Any,  # SceneGraph
    ) -> Dict[str, float]:
        """
        Extract normalized features for a camera

        Returns:
            Dictionary of features, each in [0, 1]
        """
        # 1. Covisibility score
        total_covis = camera.total_covisibility()
        max_possible_covis = graph.num_cameras() * 1000  # Assume max 1000 shared points per pair
        covisibility_score = min(total_covis / max_possible_covis, 1.0)

        # 2. Match quality score (already in [0, 1] from LightGlue)
        match_quality_score = camera.avg_match_score()

        # 3. Feature density score
        # Normalize by typical SuperPoint output (~2048 keypoints)
        feature_density_score = min(camera.num_keypoints / 2048.0, 1.0)

        # 4. Spatial uniformity score (already in [0, 1])
        uniformity_score = camera.spatial_uniformity()

        # 5. Multi-hop connectivity score
        # Number of 2-hop neighbors normalized by total possible
        two_hop_neighbors = graph.get_neighbors(camera.image_id, max_hops=2)
        max_possible = graph.num_cameras() - 1  # Exclude self
        connectivity_score = len(two_hop_neighbors) / max_possible if max_possible > 0 else 0.0

        # 6. Geometric consistency score (inlier ratio, already in [0, 1])
        geometric_score = camera.avg_inlier_ratio()

        return {
            'covisibility': covisibility_score,
            'match_quality': match_quality_score,
            'feature_density': feature_density_score,
            'spatial_uniformity': uniformity_score,
            'multi_hop_connectivity': connectivity_score,
            'geometric_consistency': geometric_score,
        }

    def compute_point_confidence(
        self,
        point_id: int,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute 3D point confidence from rule-based features

        Args:
            point_data: {'xyz': ndarray, 'rgb': ndarray, 'error': float,
                        'image_ids': list, 'point2D_idxs': list}

        Returns:
            Confidence score in [0, 1]
        """
        # Extract features
        features_dict = self._extract_point_features(point_data, cameras, images)

        # Weighted combination
        confidence = (
            self.point_weights['track_length'] * features_dict['track_length'] +
            self.point_weights['reprojection_error'] * features_dict['reprojection_error'] +
            self.point_weights['triangulation_angle'] * features_dict['triangulation_angle']
        )

        return np.clip(confidence, 0.0, 1.0)

    def _extract_point_features(
        self,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> Dict[str, float]:
        """
        Extract normalized features for a 3D point

        Returns:
            Dictionary of features, each in [0, 1]
        """
        # 1. Track length score
        # Number of cameras observing this point
        track_length = len(point_data['image_ids'])
        # Normalize by typical max track length (~20 cameras)
        track_length_score = min(track_length / 20.0, 1.0)

        # 2. Reprojection error score
        # Lower error = higher confidence
        error = point_data['error']
        # Error typically in [0, 10] pixels, use exponential decay
        error_score = np.exp(-error / 2.0)  # Decay with scale=2px

        # 3. Triangulation angle score
        # Compute average angle between camera rays
        angle_score = self._compute_triangulation_angle_score(
            point_data, cameras, images
        )

        return {
            'track_length': track_length_score,
            'reprojection_error': error_score,
            'triangulation_angle': angle_score,
        }

    def _compute_triangulation_angle_score(
        self,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute triangulation angle quality score

        Better triangulation = larger angles between viewing rays

        Returns:
            Score in [0, 1], higher = better angles
        """
        image_ids = point_data['image_ids']
        if len(image_ids) < 2:
            return 0.0

        point_xyz = point_data['xyz']  # (3,)

        # Compute camera centers
        camera_centers = []
        for img_id in image_ids:
            image = images.get(img_id)
            if image is None:
                continue

            # Camera center in world coordinates: C = -R^T * t
            qvec = image['qvec']  # (4,) quaternion
            tvec = image['tvec']  # (3,)

            # Convert quaternion to rotation matrix
            R = self._qvec_to_rotmat(qvec)
            C = -R.T @ tvec

            camera_centers.append(C)

        if len(camera_centers) < 2:
            return 0.0

        # Compute pairwise angles
        angles = []
        camera_centers = np.array(camera_centers)  # (N, 3)

        for i in range(len(camera_centers)):
            for j in range(i + 1, len(camera_centers)):
                # Vectors from point to cameras
                v1 = camera_centers[i] - point_xyz
                v2 = camera_centers[j] - point_xyz

                # Angle between vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))

                angles.append(angle_deg)

        if not angles:
            return 0.0

        # Average angle
        avg_angle = np.mean(angles)

        # Normalize: optimal angle ~30-60 degrees
        # Score = 1 at 45 degrees, decays outside [15, 75] range
        if 15 <= avg_angle <= 75:
            score = 1.0 - abs(avg_angle - 45.0) / 30.0
        else:
            score = max(0.0, 1.0 - abs(avg_angle - 45.0) / 45.0)

        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix

        Args:
            qvec: Quaternion (w, x, y, z), shape (4,)

        Returns:
            Rotation matrix, shape (3, 3)
        """
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
            [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def extract_feature_vector(
        self,
        camera: Any,  # CameraNode
        graph: Any,  # SceneGraph
    ) -> np.ndarray:
        """
        Extract feature vector for hybrid learning

        Returns:
            Feature vector, shape (6,)
        """
        features_dict = self._extract_camera_features(camera, graph)

        return np.array([
            features_dict['covisibility'],
            features_dict['match_quality'],
            features_dict['feature_density'],
            features_dict['spatial_uniformity'],
            features_dict['multi_hop_connectivity'],
            features_dict['geometric_consistency'],
        ], dtype=np.float32)
