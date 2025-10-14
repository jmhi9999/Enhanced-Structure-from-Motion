"""
Hybrid confidence computation

Combines rule-based features with a lightweight learned MLP combiner.

Architecture:
- Input: 6 rule-based features (from RuleBasedConfidence)
- Hidden layer: 16 units + ReLU
- Output: 1 confidence score + Sigmoid
- Total parameters: 6*16 + 16 + 16*1 + 1 = 129 parameters

Training:
- Self-supervised: Uses COLMAP reprojection errors as pseudo-GT
- Low error cameras → label = 1.0
- High error cameras → label = 0.0
- Binary cross-entropy loss
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    Dataset = None
    DataLoader = None

from .base import ConfidenceCalculator
from .rule_based import RuleBasedConfidence

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class ConfidenceMLP(nn.Module):
        """
        Lightweight MLP for confidence prediction

        Input: 6 rule-based features
        Hidden: 16 units + ReLU
        Output: 1 confidence score + Sigmoid
        Total: 129 parameters
        """

        def __init__(self, input_dim: int = 6, hidden_dim: int = 16):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Input features, shape (batch_size, 6)

            Returns:
                Confidence scores, shape (batch_size, 1)
            """
            return self.network(x)


    class ConfidenceDataset(Dataset):
        """Dataset for training confidence MLP"""

        def __init__(self, features: np.ndarray, labels: np.ndarray):
            """
            Args:
                features: Rule-based features, shape (N, 6)
                labels: Confidence labels, shape (N,)
            """
            self.features = torch.FloatTensor(features)
            self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (N, 1)

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, idx: int):
            return self.features[idx], self.labels[idx]


class HybridConfidence(ConfidenceCalculator):
    """
    Hybrid confidence calculator

    Combines rule-based features with learned MLP combiner.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Args:
            config: ContextBAConfig or None
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for HybridConfidence. "
                "Install with: pip install torch"
            )

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Rule-based feature extractor
        self.rule_calculator = RuleBasedConfidence(config)

        # MLP model
        hidden_dim = 16
        if config and config.hybrid_mlp:
            hidden_dim = config.hybrid_mlp.hidden_dim

        self.mlp = ConfidenceMLP(input_dim=6, hidden_dim=hidden_dim)

        # Device
        if config and config.hybrid_mlp:
            device = config.hybrid_mlp.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.mlp.to(self.device)

        # Load checkpoint if provided
        if config and config.hybrid_mlp and config.hybrid_mlp.checkpoint_path:
            self.load_checkpoint(config.hybrid_mlp.checkpoint_path)

    def compute_camera_confidence(
        self,
        camera_id: int,
        graph: Any,  # SceneGraph
        features: Dict[str, Any],
    ) -> float:
        """
        Compute camera confidence using hybrid model

        Returns:
            Confidence score in [0, 1]
        """
        camera = graph.cameras.get(camera_id)
        if camera is None:
            return 0.0

        # Extract rule-based features
        feature_vec = self.rule_calculator.extract_feature_vector(camera, graph)

        # Run through MLP
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(0).to(self.device)
            confidence = self.mlp(feature_tensor).item()

        return confidence

    def compute_point_confidence(
        self,
        point_id: int,
        point_data: Dict[str, Any],
        cameras: Dict[int, Any],
        images: Dict[int, Any],
    ) -> float:
        """
        Compute point confidence (uses rule-based, no learning)

        Returns:
            Confidence score in [0, 1]
        """
        # For points, use rule-based (not enough data for learning)
        return self.rule_calculator.compute_point_confidence(
            point_id, point_data, cameras, images
        )

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the MLP on labeled data

        Args:
            train_features: Training features, shape (N, 6)
            train_labels: Training labels (confidence), shape (N,)
            val_features: Validation features, shape (M, 6) or None
            val_labels: Validation labels, shape (M,) or None

        Returns:
            Training history
        """
        self.logger.info("Training hybrid confidence MLP...")

        # Get training config
        if self.config and self.config.hybrid_mlp:
            lr = self.config.hybrid_mlp.learning_rate
            batch_size = self.config.hybrid_mlp.batch_size
            num_epochs = self.config.hybrid_mlp.num_epochs
        else:
            lr = 1e-3
            batch_size = 32
            num_epochs = 50

        # Create datasets
        train_dataset = ConfidenceDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_features is not None and val_labels is not None:
            val_dataset = ConfidenceDataset(val_features, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and loss
        optimizer = optim.Adam(self.mlp.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            # Training
            self.mlp.train()
            train_loss = 0.0

            for features_batch, labels_batch in train_loader:
                features_batch = features_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.mlp(features_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            if val_loader is not None:
                self.mlp.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for features_batch, labels_batch in val_loader:
                        features_batch = features_batch.to(self.device)
                        labels_batch = labels_batch.to(self.device)

                        outputs = self.mlp(features_batch)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)

                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{num_epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}"
                    )

        self.logger.info("Training completed")
        return history

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.mlp.state_dict(),
            'config': {
                'input_dim': 6,
                'hidden_dim': self.mlp.network[0].out_features,
            }
        }, checkpoint_path)

        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.mlp.load_state_dict(checkpoint['model_state_dict'])

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


# Fallback if torch not available
if not TORCH_AVAILABLE:
    class HybridConfidence:
        """Dummy class when PyTorch is not available"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for HybridConfidence. "
                "Install with: pip install torch"
            )
