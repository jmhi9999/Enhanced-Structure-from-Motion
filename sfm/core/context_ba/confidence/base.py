"""
Abstract base class for confidence calculators

Defines the interface that all confidence computation strategies must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ConfidenceCalculator(ABC):
    """
    Abstract base class for confidence computation

    All confidence calculators must implement:
    - compute_camera_confidence: Returns confidence score for a camera
    - compute_point_confidence: Returns confidence score for a 3D point
    """

    @abstractmethod
    def compute_camera_confidence(
        self,
        camera_id: int,
        graph: Any,  # SceneGraph
        features: Dict[str, Any],
    ) -> float:
        """
        Compute confidence score for a camera

        Args:
            camera_id: Camera ID in scene graph
            graph: SceneGraph instance
            features: Feature data dictionary

        Returns:
            Confidence score in [0, 1]
        """
        pass

    @abstractmethod
    def compute_point_confidence(
        self,
        point_id: int,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute confidence score for a 3D point

        Args:
            point_id: Point ID
            point_data: Point3D data (xyz, rgb, error, image_ids, point2D_idxs)
            cameras: Camera data
            images: Image data

        Returns:
            Confidence score in [0, 1]
        """
        pass

    def compute_all_camera_confidences(
        self,
        graph: Any,  # SceneGraph
        features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute confidence scores for all cameras

        Args:
            graph: SceneGraph instance
            features: Feature data dictionary

        Returns:
            Array of confidence scores, shape (num_cameras,)
        """
        confidences = np.zeros(graph.num_cameras())

        for cam_id in graph.cameras.keys():
            confidences[cam_id] = self.compute_camera_confidence(cam_id, graph, features)

        return confidences

    def compute_all_point_confidences(
        self,
        points3d: Dict[int, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> Dict[int, float]:
        """
        Compute confidence scores for all 3D points

        Args:
            points3d: Dictionary of Point3D data
            cameras: Camera data
            images: Image data

        Returns:
            Dictionary mapping point_id to confidence score
        """
        confidences = {}

        for point_id, point_data in points3d.items():
            confidences[point_id] = self.compute_point_confidence(
                point_id, point_data, cameras, images
            )

        return confidences
