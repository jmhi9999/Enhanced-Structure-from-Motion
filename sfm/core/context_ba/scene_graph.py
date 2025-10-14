"""
Scene graph construction and management

Builds a graph representation of the reconstruction problem with:
- Camera nodes: One per image
- Camera-camera edges: Weighted by covisibility (number of shared points)
- Node features: Pooled descriptors, statistics, match quality

This provides global scene structure for confidence computation.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CameraNode:
    """Camera node in scene graph"""

    image_path: str
    image_id: int

    # Feature statistics
    num_keypoints: int
    keypoint_positions: np.ndarray  # (N, 2) pixel coordinates

    # Descriptor features (pooled)
    pooled_descriptor: Optional[np.ndarray] = None  # (D,) feature vector

    # Connectivity
    neighbors: Set[int] = field(default_factory=set)  # Set of connected camera IDs
    covisibility: Dict[int, int] = field(default_factory=dict)  # {camera_id: num_shared_points}

    # Match quality statistics (per neighbor)
    match_scores: Dict[int, float] = field(default_factory=dict)  # {camera_id: avg_match_score}

    # Geometric verification statistics (per neighbor)
    inlier_ratios: Dict[int, float] = field(default_factory=dict)  # {camera_id: inlier_ratio}

    def degree(self) -> int:
        """Number of connected cameras"""
        return len(self.neighbors)

    def total_covisibility(self) -> int:
        """Total number of shared points across all neighbors"""
        return sum(self.covisibility.values())

    def avg_match_score(self) -> float:
        """Average match score across all neighbors"""
        if not self.match_scores:
            return 0.0
        return np.mean(list(self.match_scores.values()))

    def avg_inlier_ratio(self) -> float:
        """Average inlier ratio across all neighbors"""
        if not self.inlier_ratios:
            return 0.0
        return np.mean(list(self.inlier_ratios.values()))

    def spatial_uniformity(self) -> float:
        """
        Compute spatial distribution uniformity of keypoints

        Returns:
            Uniformity score in [0, 1], higher = more uniform
        """
        if len(self.keypoint_positions) < 4:
            return 0.0

        # Divide image into 4x4 grid
        grid_size = 4
        kpts = self.keypoint_positions

        # Normalize to [0, 1]
        kpts_norm = (kpts - kpts.min(axis=0)) / (kpts.max(axis=0) - kpts.min(axis=0) + 1e-6)

        # Compute grid cell occupancy
        grid_x = (kpts_norm[:, 0] * grid_size).astype(int)
        grid_y = (kpts_norm[:, 1] * grid_size).astype(int)
        grid_x = np.clip(grid_x, 0, grid_size - 1)
        grid_y = np.clip(grid_y, 0, grid_size - 1)

        # Count points per cell
        cell_counts = np.zeros((grid_size, grid_size))
        for x, y in zip(grid_x, grid_y):
            cell_counts[x, y] += 1

        # Compute uniformity as entropy normalized by max entropy
        occupied_cells = cell_counts[cell_counts > 0]
        if len(occupied_cells) == 0:
            return 0.0

        probs = occupied_cells / occupied_cells.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(grid_size * grid_size)

        return entropy / max_entropy


class SceneGraph:
    """
    Scene graph data structure

    Represents the reconstruction problem as a graph with cameras as nodes
    and covisibility relationships as edges.
    """

    def __init__(self):
        self.cameras: Dict[int, CameraNode] = {}  # {camera_id: CameraNode}
        self.image_to_id: Dict[str, int] = {}  # {image_path: camera_id}

    def add_camera(self, node: CameraNode) -> None:
        """Add camera node to graph"""
        self.cameras[node.image_id] = node
        self.image_to_id[node.image_path] = node.image_id

    def add_edge(self, cam1_id: int, cam2_id: int, weight: int) -> None:
        """Add bidirectional edge between cameras with covisibility weight"""
        if cam1_id not in self.cameras or cam2_id not in self.cameras:
            raise ValueError(f"Camera IDs {cam1_id} or {cam2_id} not in graph")

        self.cameras[cam1_id].neighbors.add(cam2_id)
        self.cameras[cam2_id].neighbors.add(cam1_id)
        self.cameras[cam1_id].covisibility[cam2_id] = weight
        self.cameras[cam2_id].covisibility[cam1_id] = weight

    def get_camera_by_path(self, image_path: str) -> Optional[CameraNode]:
        """Get camera node by image path"""
        cam_id = self.image_to_id.get(image_path)
        if cam_id is None:
            return None
        return self.cameras.get(cam_id)

    def get_neighbors(self, cam_id: int, max_hops: int = 1) -> Set[int]:
        """
        Get neighbors within max_hops distance

        Args:
            cam_id: Camera ID
            max_hops: Maximum graph distance (1 = direct neighbors, 2 = friends-of-friends)

        Returns:
            Set of camera IDs within max_hops
        """
        if cam_id not in self.cameras:
            return set()

        visited = {cam_id}
        current_frontier = {cam_id}

        for _ in range(max_hops):
            next_frontier = set()
            for node_id in current_frontier:
                neighbors = self.cameras[node_id].neighbors
                next_frontier.update(neighbors - visited)
            visited.update(next_frontier)
            current_frontier = next_frontier

        visited.remove(cam_id)  # Remove self
        return visited

    def num_cameras(self) -> int:
        """Number of camera nodes"""
        return len(self.cameras)

    def num_edges(self) -> int:
        """Number of edges (undirected)"""
        return sum(node.degree() for node in self.cameras.values()) // 2

    def __repr__(self) -> str:
        return f"SceneGraph(cameras={self.num_cameras()}, edges={self.num_edges()})"


class SceneGraphBuilder:
    """
    Build scene graph from features and matches

    Constructs the graph representation from SfM pipeline outputs.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: SceneGraphConfig or None (uses defaults)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def build(
        self,
        features: Dict[str, Any],
        matches: Dict[Tuple[str, str], Any]
    ) -> SceneGraph:
        """
        Build scene graph from features and matches

        Args:
            features: {image_path: {'keypoints': ndarray, 'descriptors': ndarray, ...}}
            matches: {(img1, img2): {'matches0': ndarray, 'matches1': ndarray, 'mscores0': ndarray, ...}}

        Returns:
            SceneGraph instance
        """
        self.logger.info("Building scene graph...")

        graph = SceneGraph()

        # Step 1: Create camera nodes
        for image_id, (image_path, feature_data) in enumerate(features.items()):
            node = self._create_camera_node(image_id, image_path, feature_data)
            graph.add_camera(node)

        self.logger.info(f"Created {graph.num_cameras()} camera nodes")

        # Step 2: Add edges from matches
        covisibility_counts = self._compute_covisibility(features, matches)
        num_edges = 0

        for (img1, img2), num_shared in covisibility_counts.items():
            # Apply minimum covisibility threshold
            min_covis = self.config.min_covisibility if self.config else 10
            if num_shared < min_covis:
                continue

            cam1 = graph.get_camera_by_path(img1)
            cam2 = graph.get_camera_by_path(img2)

            if cam1 is None or cam2 is None:
                continue

            graph.add_edge(cam1.image_id, cam2.image_id, num_shared)
            num_edges += 1

        self.logger.info(f"Added {num_edges} edges (covisibility >= {min_covis})")

        # Step 3: Add match quality statistics
        self._add_match_statistics(graph, matches)

        self.logger.info(f"Scene graph built: {graph}")
        return graph

    def _create_camera_node(
        self,
        image_id: int,
        image_path: str,
        feature_data: Dict[str, Any]
    ) -> CameraNode:
        """Create camera node from feature data"""
        keypoints = feature_data['keypoints']  # (N, 2)
        descriptors = feature_data.get('descriptors')  # (N, D) or None

        # Pool descriptors
        pooled_desc = None
        if descriptors is not None:
            pooling = self.config.pooling_method if self.config else "mean"
            if pooling == "mean":
                pooled_desc = descriptors.mean(axis=0)
            elif pooling == "max":
                pooled_desc = descriptors.max(axis=0)
            else:
                # Default to mean
                pooled_desc = descriptors.mean(axis=0)

        return CameraNode(
            image_path=image_path,
            image_id=image_id,
            num_keypoints=len(keypoints),
            keypoint_positions=keypoints,
            pooled_descriptor=pooled_desc
        )

    def _compute_covisibility(
        self,
        features: Dict[str, Any],
        matches: Dict[Tuple[str, str], Any]
    ) -> Dict[Tuple[str, str], int]:
        """
        Compute covisibility (number of shared matched points) between image pairs

        Returns:
            {(img1, img2): num_shared_matches}
        """
        covisibility = {}

        for (img1, img2), match_data in matches.items():
            matches0 = match_data.get('matches0', np.array([]))
            matches1 = match_data.get('matches1', np.array([]))

            # Count valid matches
            valid_mask = (matches0 >= 0) & (matches1 >= 0)
            num_shared = valid_mask.sum()

            if num_shared > 0:
                covisibility[(img1, img2)] = num_shared

        return covisibility

    def _add_match_statistics(
        self,
        graph: SceneGraph,
        matches: Dict[Tuple[str, str], Any]
    ) -> None:
        """Add match quality and inlier ratio statistics to camera nodes"""
        for (img1, img2), match_data in matches.items():
            cam1 = graph.get_camera_by_path(img1)
            cam2 = graph.get_camera_by_path(img2)

            if cam1 is None or cam2 is None:
                continue

            # Extract match scores (from LightGlue or other matcher)
            mscores0 = match_data.get('mscores0', np.array([]))
            mscores1 = match_data.get('mscores1', np.array([]))

            # Average match score
            if len(mscores0) > 0 and len(mscores1) > 0:
                avg_score = (mscores0.mean() + mscores1.mean()) / 2.0
            else:
                avg_score = 1.0  # Default if no scores available

            cam1.match_scores[cam2.image_id] = avg_score
            cam2.match_scores[cam1.image_id] = avg_score

            # Inlier ratio (if MAGSAC was applied, all matches are inliers)
            matches0 = match_data.get('matches0', np.array([]))
            matches1 = match_data.get('matches1', np.array([]))
            valid_mask = (matches0 >= 0) & (matches1 >= 0)
            total_matches = len(matches0)
            num_inliers = valid_mask.sum()

            inlier_ratio = num_inliers / total_matches if total_matches > 0 else 0.0

            cam1.inlier_ratios[cam2.image_id] = inlier_ratio
            cam2.inlier_ratios[cam1.image_id] = inlier_ratio
