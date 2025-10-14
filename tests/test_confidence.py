"""
Unit tests for confidence computation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sfm.core.context_ba.confidence import RuleBasedConfidence, HYBRID_AVAILABLE
from sfm.core.context_ba.scene_graph import CameraNode, SceneGraph
from sfm.core.context_ba.config import ContextBAConfig


class TestRuleBasedConfidence:
    """Test rule-based confidence computation"""

    def create_test_graph(self):
        """Create a simple test graph"""
        graph = SceneGraph()

        # High-quality camera (many features, well-connected)
        node_high = CameraNode(
            image_path="high_quality.jpg",
            image_id=0,
            num_keypoints=2000,
            keypoint_positions=np.random.rand(2000, 2) * 1000,
        )
        node_high.neighbors = {1, 2, 3}
        node_high.covisibility = {1: 100, 2: 100, 3: 100}
        node_high.match_scores = {1: 0.9, 2: 0.9, 3: 0.9}
        node_high.inlier_ratios = {1: 0.95, 2: 0.95, 3: 0.95}

        # Low-quality camera (few features, poorly connected)
        node_low = CameraNode(
            image_path="low_quality.jpg",
            image_id=1,
            num_keypoints=100,
            keypoint_positions=np.random.rand(100, 2) * 1000,
        )
        node_low.neighbors = {0}
        node_low.covisibility = {0: 10}
        node_low.match_scores = {0: 0.3}
        node_low.inlier_ratios = {0: 0.5}

        # Medium-quality camera
        node_med = CameraNode(
            image_path="medium_quality.jpg",
            image_id=2,
            num_keypoints=1000,
            keypoint_positions=np.random.rand(1000, 2) * 1000,
        )
        node_med.neighbors = {0, 3}
        node_med.covisibility = {0: 50, 3: 50}
        node_med.match_scores = {0: 0.7, 3: 0.7}
        node_med.inlier_ratios = {0: 0.8, 3: 0.8}

        # Additional node for connectivity
        node_3 = CameraNode(
            image_path="node3.jpg",
            image_id=3,
            num_keypoints=1500,
            keypoint_positions=np.random.rand(1500, 2) * 1000,
        )
        node_3.neighbors = {0, 2}
        node_3.covisibility = {0: 80, 2: 80}
        node_3.match_scores = {0: 0.8, 2: 0.8}
        node_3.inlier_ratios = {0: 0.85, 2: 0.85}

        graph.add_camera(node_high)
        graph.add_camera(node_low)
        graph.add_camera(node_med)
        graph.add_camera(node_3)

        return graph

    def test_confidence_ordering(self):
        """Test that high-quality cameras get higher confidence"""
        graph = self.create_test_graph()
        config = ContextBAConfig()
        confidence_calc = RuleBasedConfidence(config)

        conf_high = confidence_calc.compute_camera_confidence(0, graph, {})
        conf_low = confidence_calc.compute_camera_confidence(1, graph, {})
        conf_med = confidence_calc.compute_camera_confidence(2, graph, {})

        assert conf_high > conf_med > conf_low
        assert 0 <= conf_low <= 1
        assert 0 <= conf_med <= 1
        assert 0 <= conf_high <= 1

    def test_feature_extraction(self):
        """Test feature vector extraction"""
        graph = self.create_test_graph()
        config = ContextBAConfig()
        confidence_calc = RuleBasedConfidence(config)

        camera = graph.cameras[0]
        features = confidence_calc._extract_camera_features(camera, graph)

        assert 'covisibility' in features
        assert 'match_quality' in features
        assert 'feature_density' in features
        assert 'spatial_uniformity' in features
        assert 'multi_hop_connectivity' in features
        assert 'geometric_consistency' in features

        for value in features.values():
            assert 0 <= value <= 1

    def test_point_confidence(self):
        """Test point confidence computation"""
        config = ContextBAConfig()
        confidence_calc = RuleBasedConfidence(config)

        # Good point (many observations, low error)
        point_good = {
            'xyz': np.array([0, 0, 5]),
            'rgb': np.array([128, 128, 128]),
            'error': 0.3,
            'image_ids': [0, 1, 2, 3, 4],  # 5 observations
            'point2D_idxs': [0, 0, 0, 0, 0],
        }

        # Bad point (few observations, high error)
        point_bad = {
            'xyz': np.array([0, 0, 5]),
            'rgb': np.array([128, 128, 128]),
            'error': 5.0,
            'image_ids': [0, 1],  # Only 2 observations
            'point2D_idxs': [0, 0],
        }

        cameras = {}
        images = {
            i: {
                'qvec': np.array([1, 0, 0, 0]),
                'tvec': np.array([i*0.1, 0, 0]),
                'camera_id': 0,
            }
            for i in range(6)
        }

        conf_good = confidence_calc.compute_point_confidence(0, point_good, cameras, images)
        conf_bad = confidence_calc.compute_point_confidence(1, point_bad, cameras, images)

        assert conf_good > conf_bad
        assert 0 <= conf_good <= 1
        assert 0 <= conf_bad <= 1

    def test_batch_computation(self):
        """Test batch confidence computation"""
        graph = self.create_test_graph()
        config = ContextBAConfig()
        confidence_calc = RuleBasedConfidence(config)

        confidences = confidence_calc.compute_all_camera_confidences(graph, {})

        assert len(confidences) == graph.num_cameras()
        assert all(0 <= c <= 1 for c in confidences)


@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="PyTorch not available")
class TestHybridConfidence:
    """Test hybrid confidence computation (requires PyTorch)"""

    def test_mlp_forward_pass(self):
        """Test MLP forward pass"""
        from sfm.core.context_ba.confidence import HybridConfidence

        config = ContextBAConfig(confidence_mode="hybrid")
        hybrid_calc = HybridConfidence(config)

        # Create dummy graph
        graph = SceneGraph()
        node = CameraNode(
            image_path="test.jpg",
            image_id=0,
            num_keypoints=1000,
            keypoint_positions=np.random.rand(1000, 2) * 1000,
        )
        node.neighbors = {1}
        node.covisibility = {1: 50}
        node.match_scores = {1: 0.8}
        node.inlier_ratios = {1: 0.9}

        graph.add_camera(node)

        # Compute confidence
        confidence = hybrid_calc.compute_camera_confidence(0, graph, {})

        assert 0 <= confidence <= 1

    def test_training(self):
        """Test MLP training"""
        from sfm.core.context_ba.confidence import HybridConfidence

        config = ContextBAConfig(confidence_mode="hybrid")
        hybrid_calc = HybridConfidence(config)

        # Dummy training data
        train_features = np.random.rand(100, 6).astype(np.float32)
        train_labels = np.random.rand(100).astype(np.float32)

        # Train for 5 epochs (quick test)
        config.hybrid_mlp.num_epochs = 5
        history = hybrid_calc.train(train_features, train_labels)

        assert 'train_loss' in history
        assert len(history['train_loss']) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
