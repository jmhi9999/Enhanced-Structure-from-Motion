"""
Unit tests for scene graph construction
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sfm.core.context_ba.scene_graph import (
    CameraNode,
    SceneGraph,
    SceneGraphBuilder,
)
from sfm.core.context_ba.config import SceneGraphConfig


class TestCameraNode:
    """Test CameraNode data structure"""

    def test_camera_node_creation(self):
        """Test basic camera node creation"""
        node = CameraNode(
            image_path="/path/to/img.jpg",
            image_id=0,
            num_keypoints=100,
            keypoint_positions=np.random.rand(100, 2) * 1000,
        )

        assert node.image_path == "/path/to/img.jpg"
        assert node.image_id == 0
        assert node.num_keypoints == 100
        assert node.degree() == 0

    def test_spatial_uniformity(self):
        """Test spatial uniformity computation"""
        # Uniform distribution (4x4 grid, 1 point per cell)
        kpts_uniform = np.array([
            [i*250, j*250] for i in range(4) for j in range(4)
        ])
        node_uniform = CameraNode(
            image_path="test.jpg",
            image_id=0,
            num_keypoints=16,
            keypoint_positions=kpts_uniform,
        )
        uniformity_uniform = node_uniform.spatial_uniformity()

        # Clustered distribution (all in one corner)
        kpts_clustered = np.random.rand(100, 2) * 100  # All in [0, 100]
        node_clustered = CameraNode(
            image_path="test.jpg",
            image_id=0,
            num_keypoints=100,
            keypoint_positions=kpts_clustered,
        )
        uniformity_clustered = node_clustered.spatial_uniformity()

        assert uniformity_uniform > uniformity_clustered
        assert 0 <= uniformity_uniform <= 1
        assert 0 <= uniformity_clustered <= 1

    def test_covisibility_tracking(self):
        """Test covisibility tracking"""
        node = CameraNode(
            image_path="test.jpg",
            image_id=0,
            num_keypoints=100,
            keypoint_positions=np.random.rand(100, 2),
        )

        node.neighbors.add(1)
        node.neighbors.add(2)
        node.covisibility[1] = 50
        node.covisibility[2] = 30

        assert node.degree() == 2
        assert node.total_covisibility() == 80


class TestSceneGraph:
    """Test SceneGraph data structure"""

    def test_graph_creation(self):
        """Test basic graph creation"""
        graph = SceneGraph()

        node1 = CameraNode("img1.jpg", 0, 100, np.random.rand(100, 2))
        node2 = CameraNode("img2.jpg", 1, 100, np.random.rand(100, 2))

        graph.add_camera(node1)
        graph.add_camera(node2)

        assert graph.num_cameras() == 2
        assert graph.num_edges() == 0

    def test_edge_addition(self):
        """Test edge addition"""
        graph = SceneGraph()

        node1 = CameraNode("img1.jpg", 0, 100, np.random.rand(100, 2))
        node2 = CameraNode("img2.jpg", 1, 100, np.random.rand(100, 2))

        graph.add_camera(node1)
        graph.add_camera(node2)
        graph.add_edge(0, 1, weight=50)

        assert graph.num_edges() == 1
        assert graph.cameras[0].degree() == 1
        assert graph.cameras[1].degree() == 1
        assert graph.cameras[0].covisibility[1] == 50

    def test_get_neighbors(self):
        """Test neighbor retrieval"""
        graph = SceneGraph()

        # Create chain: 0 - 1 - 2 - 3
        for i in range(4):
            node = CameraNode(f"img{i}.jpg", i, 100, np.random.rand(100, 2))
            graph.add_camera(node)

        graph.add_edge(0, 1, weight=10)
        graph.add_edge(1, 2, weight=10)
        graph.add_edge(2, 3, weight=10)

        # 1-hop neighbors of node 1
        neighbors_1hop = graph.get_neighbors(1, max_hops=1)
        assert neighbors_1hop == {0, 2}

        # 2-hop neighbors of node 1
        neighbors_2hop = graph.get_neighbors(1, max_hops=2)
        assert neighbors_2hop == {0, 2, 3}


class TestSceneGraphBuilder:
    """Test SceneGraphBuilder"""

    def create_dummy_features(self, num_images: int = 3):
        """Create dummy feature data"""
        features = {}
        for i in range(num_images):
            features[f"img{i}.jpg"] = {
                'keypoints': np.random.rand(100, 2) * 1000,
                'descriptors': np.random.rand(100, 256),
            }
        return features

    def create_dummy_matches(self, num_images: int = 3):
        """Create dummy match data"""
        matches = {}
        # Sequential matches: 0-1, 1-2
        for i in range(num_images - 1):
            img1 = f"img{i}.jpg"
            img2 = f"img{i+1}.jpg"

            num_matches = 50
            matches[(img1, img2)] = {
                'matches0': np.random.randint(0, 100, num_matches),
                'matches1': np.random.randint(0, 100, num_matches),
                'mscores0': np.random.rand(num_matches),
                'mscores1': np.random.rand(num_matches),
            }
        return matches

    def test_graph_building(self):
        """Test graph building from features and matches"""
        features = self.create_dummy_features(num_images=3)
        matches = self.create_dummy_matches(num_images=3)

        config = SceneGraphConfig(min_covisibility=10)
        builder = SceneGraphBuilder(config)

        graph = builder.build(features, matches)

        assert graph.num_cameras() == 3
        assert graph.num_edges() >= 0  # May be 0 if covisibility threshold not met

    def test_covisibility_computation(self):
        """Test covisibility computation"""
        features = self.create_dummy_features(num_images=2)
        matches = {
            ('img0.jpg', 'img1.jpg'): {
                'matches0': np.array([0, 1, 2]),
                'matches1': np.array([5, 6, 7]),
                'mscores0': np.array([0.9, 0.8, 0.7]),
                'mscores1': np.array([0.9, 0.8, 0.7]),
            }
        }

        config = SceneGraphConfig(min_covisibility=1)
        builder = SceneGraphBuilder(config)

        covisibility = builder._compute_covisibility(features, matches)

        assert ('img0.jpg', 'img1.jpg') in covisibility
        assert covisibility[('img0.jpg', 'img1.jpg')] == 3

    def test_match_statistics(self):
        """Test match statistics computation"""
        features = self.create_dummy_features(num_images=2)
        matches = {
            ('img0.jpg', 'img1.jpg'): {
                'matches0': np.array([0, 1, 2]),
                'matches1': np.array([5, 6, 7]),
                'mscores0': np.array([0.9, 0.9, 0.9]),
                'mscores1': np.array([0.8, 0.8, 0.8]),
            }
        }

        config = SceneGraphConfig(min_covisibility=1)
        builder = SceneGraphBuilder(config)

        graph = builder.build(features, matches)

        node0 = graph.cameras[0]
        assert 1 in node0.match_scores
        # Average of (0.9 + 0.8) / 2 = 0.85
        assert abs(node0.match_scores[1] - 0.85) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
