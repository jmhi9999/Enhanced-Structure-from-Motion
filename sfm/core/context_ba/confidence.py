"""
Confidence computation for Context-Aware Bundle Adjustment

Computes confidence scores for cameras and 3D points based on:
- Scene graph structure (covisibility, connectivity)
- Feature quality (density, uniformity, match scores)
- Geometric consistency (inlier ratios, triangulation angles)
- Track statistics (track length with saturation)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .scene_graph import SceneGraph, CameraNode

logger = logging.getLogger(__name__)

# Check if PyTorch available for hybrid confidence (optional)
try:
    import torch
    import torch.nn as nn
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    torch = None
    nn = None


class ConfidenceCalculator(ABC):
    """Abstract base class for confidence computation"""

    @abstractmethod
    def compute_all_camera_confidences(
        self,
        scene_graph: SceneGraph,
        features: Dict[str, Any],
    ) -> np.ndarray:
        """Compute confidence scores for all cameras"""
        pass

    @abstractmethod
    def compute_all_point_confidences(
        self,
        points3d: Dict[int, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> Dict[int, float]:
        """Compute confidence scores for all 3D points"""
        pass


class RuleBasedConfidence(ConfidenceCalculator):
    """
    Rule-based confidence computation (no training required)

    Camera confidence factors:
        - Covisibility (25%)
        - Match quality (20%)
        - Feature density (15%)
        - Spatial uniformity (15%)
        - 2-hop connectivity (15%)
        - Geometric consistency/inlier ratio (10%)

    Point confidence factors:
        - Track length with saturation (50%)
        - Reprojection error (30%)
        - Triangulation angle (20%)
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def compute_all_camera_confidences(
        self,
        scene_graph: SceneGraph,
        features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute confidence scores for all cameras in scene graph

        Returns:
            Array of shape (num_cameras,) with confidence scores ∈ [0, 1]
        """
        num_cameras = scene_graph.num_cameras()
        if num_cameras == 0:
            return np.array([])

        confidences = np.zeros(num_cameras)

        for cam_id, camera_node in scene_graph.cameras.items():
            confidence = self._compute_camera_confidence(camera_node, scene_graph)
            confidences[cam_id] = confidence

        return confidences

    def _compute_camera_confidence(
        self,
        camera_node: CameraNode,
        scene_graph: SceneGraph,
    ) -> float:
        """
        Compute confidence for a single camera

        Formula:
            confidence = 0.25×covisibility + 0.20×match_quality +
                        0.15×feature_density + 0.15×uniformity +
                        0.15×two_hop_connectivity + 0.10×inlier_ratio
        """
        # 1. Covisibility score (25%)
        total_covisibility = camera_node.total_covisibility()
        max_covisibility = 500  # Typical max for well-connected camera
        covisibility_score = min(total_covisibility / max_covisibility, 1.0)

        # 2. Match quality score (20%)
        match_quality_score = camera_node.avg_match_score()

        # 3. Feature density score (15%)
        num_keypoints = camera_node.num_keypoints
        max_keypoints = 2000  # Typical max
        feature_density_score = min(num_keypoints / max_keypoints, 1.0)

        # 4. Spatial uniformity score (15%)
        # Already implemented in scene_graph.py:66-103 using grid entropy!
        uniformity_score = camera_node.spatial_uniformity()

        # 5. Two-hop connectivity score (15%)
        neighbors = scene_graph.get_neighbors(camera_node.image_id, max_hops=2)
        two_hop_connectivity = len(neighbors)
        max_two_hop = 50  # Typical max for well-connected scene
        two_hop_score = min(two_hop_connectivity / max_two_hop, 1.0)

        # 6. Geometric consistency score (10%)
        inlier_ratio_score = camera_node.avg_inlier_ratio()

        # Weighted combination
        confidence = (
            0.25 * covisibility_score +
            0.20 * match_quality_score +
            0.15 * feature_density_score +
            0.15 * uniformity_score +
            0.15 * two_hop_score +
            0.10 * inlier_ratio_score
        )

        return np.clip(confidence, 0.0, 1.0)

    def compute_all_point_confidences(
        self,
        points3d: Dict[int, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> Dict[int, float]:
        """
        Compute confidence scores for all 3D points

        Returns:
            Dictionary mapping point_id to confidence score ∈ [0, 1]
        """
        point_confidences = {}

        for point_id, point_data in points3d.items():
            confidence = self._compute_point_confidence(
                point_data, cameras, images
            )
            point_confidences[point_id] = confidence

        return point_confidences

    def _compute_point_confidence(
        self,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute confidence for a single 3D point

        Formula (with saturation):
            confidence = 0.50×track_score + 0.30×error_score + 0.20×angle_score

        where:
            track_score = min(sqrt(L_j) / sqrt(10), 1.0)  # SATURATED!
            L_j = track length (number of observations)
        """
        # 1. Track length score with saturation (50%)
        track_length = len(point_data.get('image_ids', []))

        # Hard threshold: reject points with < 3 observations
        if track_length < 3:
            return 0.0  # Unreliable point

        # Saturated track score (sqrt for diminishing returns)
        L_ref = 10.0  # Reference track length
        track_score = min(np.sqrt(track_length) / np.sqrt(L_ref), 1.0)

        # 2. Reprojection error score (30%)
        error = point_data.get('error', 1.0)  # Default to 1.0 if not available
        error_score = 1.0 / (1.0 + error)  # Inverse relationship

        # 3. Triangulation angle score (20%)
        # Compute average triangulation angle between observing cameras
        tri_angle = self._compute_triangulation_angle(point_data, cameras, images)
        tri_angle_deg = np.rad2deg(tri_angle)
        angle_score = np.clip(tri_angle_deg / 30.0, 0.0, 1.0)  # Saturate at 30°

        # Weighted combination
        confidence = (
            0.50 * track_score +
            0.30 * error_score +
            0.20 * angle_score
        )

        return np.clip(confidence, 0.0, 1.0)

    def _compute_triangulation_angle(
        self,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute average triangulation angle for a 3D point

        Returns:
            Average angle in radians
        """
        image_ids = point_data.get('image_ids', [])
        if len(image_ids) < 2:
            return 0.0

        point_xyz = point_data.get('xyz', np.zeros(3))
        if isinstance(point_xyz, list):
            point_xyz = np.array(point_xyz)

        angles = []
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                img_id_i = image_ids[i]
                img_id_j = image_ids[j]

                # Get camera centers
                image_i = images.get(img_id_i)
                image_j = images.get(img_id_j)

                if image_i is None or image_j is None:
                    continue

                # Camera center is -R^T @ t (for pose [R | t])
                tvec_i = np.array(image_i.get('tvec', [0, 0, 0]))
                tvec_j = np.array(image_j.get('tvec', [0, 0, 0]))

                qvec_i = np.array(image_i.get('qvec', [1, 0, 0, 0]))
                qvec_j = np.array(image_j.get('qvec', [1, 0, 0, 0]))

                R_i = self._quat_to_rotation_matrix(qvec_i)
                R_j = self._quat_to_rotation_matrix(qvec_j)

                center_i = -R_i.T @ tvec_i
                center_j = -R_j.T @ tvec_j

                # Rays from cameras to point
                ray_i = point_xyz - center_i
                ray_j = point_xyz - center_j

                # Normalize
                ray_i_norm = ray_i / (np.linalg.norm(ray_i) + 1e-10)
                ray_j_norm = ray_j / (np.linalg.norm(ray_j) + 1e-10)

                # Angle between rays
                cos_angle = np.clip(np.dot(ray_i_norm, ray_j_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(angle)

        if len(angles) == 0:
            return 0.0

        return float(np.mean(angles))

    @staticmethod
    def _quat_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix"""
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
            [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R


# Optional: Hybrid confidence with lightweight MLP
# (Only available if PyTorch is installed)
if HYBRID_AVAILABLE:
    class HybridConfidence(ConfidenceCalculator):
        """
        Hybrid confidence: Rule-based features + learned MLP combiner

        Architecture:
            6 rule-based features → 16-dim hidden → 1 confidence score
            Total parameters: 6×16 + 16 + 16×1 + 1 = 129 parameters

        Training:
            - Use COLMAP results as pseudo-GT
            - Low error cameras → label = 1.0
            - High error cameras → label = 0.0
            - Train on 50-100 scenes (few hours)
        """

        def __init__(self, config: Optional[Any] = None):
            self.config = config
            self.rule_based = RuleBasedConfidence(config)
            self.logger = logging.getLogger(__name__)

            # Tiny MLP combiner (only 129 parameters!)
            self.combiner = nn.Sequential(
                nn.Linear(6, 16),   # 6 rule-based features
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

            # TODO: Load pretrained weights if available
            # self.combiner.load_state_dict(torch.load('pretrained_combiner.pth'))

        def compute_all_camera_confidences(
            self,
            scene_graph: SceneGraph,
            features: Dict[str, Any],
        ) -> np.ndarray:
            """
            Compute camera confidences using learned combiner

            Returns:
                Array of shape (num_cameras,)
            """
            # Extract rule-based features for all cameras
            num_cameras = scene_graph.num_cameras()
            if num_cameras == 0:
                return np.array([])

            # Feature matrix: (num_cameras, 6)
            feature_matrix = np.zeros((num_cameras, 6))

            for cam_id, camera_node in scene_graph.cameras.items():
                features_cam = self._extract_camera_features(camera_node, scene_graph)
                feature_matrix[cam_id] = features_cam

            # Run through MLP
            with torch.no_grad():
                features_tensor = torch.from_numpy(feature_matrix).float()
                confidences_tensor = self.combiner(features_tensor).squeeze(-1)
                confidences = confidences_tensor.numpy()

            return confidences

        def _extract_camera_features(
            self,
            camera_node: CameraNode,
            scene_graph: SceneGraph,
        ) -> np.ndarray:
            """Extract 6 rule-based features for a camera"""
            # Same as RuleBasedConfidence but return raw features
            total_covisibility = camera_node.total_covisibility()
            covisibility_score = min(total_covisibility / 500.0, 1.0)

            match_quality_score = camera_node.avg_match_score()

            feature_density_score = min(camera_node.num_keypoints / 2000.0, 1.0)

            uniformity_score = camera_node.spatial_uniformity()

            neighbors = scene_graph.get_neighbors(camera_node.image_id, max_hops=2)
            two_hop_score = min(len(neighbors) / 50.0, 1.0)

            inlier_ratio_score = camera_node.avg_inlier_ratio()

            return np.array([
                covisibility_score,
                match_quality_score,
                feature_density_score,
                uniformity_score,
                two_hop_score,
                inlier_ratio_score
            ])

        def compute_all_point_confidences(
            self,
            points3d: Dict[int, Any],
            cameras: Dict[int, Any],
            images: Dict[int, Any],
        ) -> Dict[int, float]:
            """For now, use rule-based point confidence (could add MLP later)"""
            return self.rule_based.compute_all_point_confidences(
                points3d, cameras, images
            )

else:
    # Stub if PyTorch not available
    class HybridConfidence:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "HybridConfidence requires PyTorch. "
                "Install with: pip install torch"
            )
