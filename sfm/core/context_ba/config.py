"""
Configuration management for Context-Aware Bundle Adjustment

Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ConfidenceWeights:
    """Weights for rule-based confidence factors"""

    # Camera confidence weights (must sum to 1.0)
    covisibility: float = 0.25
    match_quality: float = 0.20
    feature_density: float = 0.15
    spatial_uniformity: float = 0.15
    multi_hop_connectivity: float = 0.15
    geometric_consistency: float = 0.10

    # Point confidence weights (must sum to 1.0)
    track_length: float = 0.50
    reprojection_error: float = 0.30
    triangulation_angle: float = 0.20

    def __post_init__(self):
        """Validate that weights sum to 1.0"""
        camera_sum = (
            self.covisibility + self.match_quality + self.feature_density +
            self.spatial_uniformity + self.multi_hop_connectivity + self.geometric_consistency
        )
        point_sum = self.track_length + self.reprojection_error + self.triangulation_angle

        if not (0.99 <= camera_sum <= 1.01):
            raise ValueError(f"Camera confidence weights must sum to 1.0, got {camera_sum:.3f}")
        if not (0.99 <= point_sum <= 1.01):
            raise ValueError(f"Point confidence weights must sum to 1.0, got {point_sum:.3f}")


@dataclass
class SceneGraphConfig:
    """Configuration for scene graph construction"""

    # Feature pooling method for camera nodes
    pooling_method: str = "mean"  # "mean", "max", or "attention"

    # Covisibility threshold (minimum shared points to create edge)
    min_covisibility: int = 10

    # Include point nodes in graph (optional, increases memory)
    include_point_nodes: bool = False

    # Maximum number of hops for connectivity analysis
    max_hop_distance: int = 2


@dataclass
class OptimizerConfig:
    """Configuration for bundle adjustment optimizer"""

    # Optimization method: "scipy" or "ceres" (if available)
    method: str = "scipy"

    # Maximum iterations
    max_iterations: int = 100

    # Convergence tolerance
    ftol: float = 1e-6
    xtol: float = 1e-6

    # Loss function: "linear", "soft_l1", "huber", "cauchy", "arctan"
    loss: str = "soft_l1"

    # Verbose output
    verbose: int = 2


@dataclass
class HybridMLPConfig:
    """Configuration for hybrid confidence MLP"""

    # Hidden layer size
    hidden_dim: int = 16

    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50

    # Model checkpoint path
    checkpoint_path: Optional[Path] = None

    # Device: "cpu", "cuda", or "auto"
    device: str = "auto"


@dataclass
class ContextBAConfig:
    """Main configuration for Context-Aware Bundle Adjustment"""

    # Confidence calculation method: "rule_based" or "hybrid"
    confidence_mode: str = "rule_based"

    # Sub-configurations
    weights: ConfidenceWeights = field(default_factory=ConfidenceWeights)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    hybrid_mlp: Optional[HybridMLPConfig] = None

    # Minimum confidence threshold (cameras/points below this are excluded)
    min_confidence_threshold: float = 0.1

    # Enable confidence weighting in BA (if False, uses uniform weights)
    enable_confidence_weighting: bool = True

    # Logging level: "DEBUG", "INFO", "WARNING", "ERROR"
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration"""
        if self.confidence_mode not in ["rule_based", "hybrid"]:
            raise ValueError(f"Invalid confidence_mode: {self.confidence_mode}")

        if self.confidence_mode == "hybrid" and self.hybrid_mlp is None:
            self.hybrid_mlp = HybridMLPConfig()

        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            raise ValueError(f"min_confidence_threshold must be in [0, 1], got {self.min_confidence_threshold}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ContextBAConfig":
        """Create config from dictionary (for CLI/JSON loading)"""
        # Nested configs
        weights = ConfidenceWeights(**config_dict.pop("weights", {}))
        scene_graph = SceneGraphConfig(**config_dict.pop("scene_graph", {}))
        optimizer = OptimizerConfig(**config_dict.pop("optimizer", {}))

        hybrid_mlp = None
        if "hybrid_mlp" in config_dict:
            hybrid_mlp = HybridMLPConfig(**config_dict.pop("hybrid_mlp"))

        return cls(
            weights=weights,
            scene_graph=scene_graph,
            optimizer=optimizer,
            hybrid_mlp=hybrid_mlp,
            **config_dict
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary"""
        return {
            "confidence_mode": self.confidence_mode,
            "weights": self.weights.__dict__,
            "scene_graph": self.scene_graph.__dict__,
            "optimizer": self.optimizer.__dict__,
            "hybrid_mlp": self.hybrid_mlp.__dict__ if self.hybrid_mlp else None,
            "min_confidence_threshold": self.min_confidence_threshold,
            "enable_confidence_weighting": self.enable_confidence_weighting,
            "log_level": self.log_level,
        }
