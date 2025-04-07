# config/model_config.py
from typing import Optional, Dict, Any
import os
import json

class BaseConfig:
    """Base configuration with validation and serialization"""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate(self) -> bool:
        """Validate configuration."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class UniRepModelConfig(BaseConfig):
    """Configuration for UniRep models."""

    def __init__(
        self,
        rnn_size: int = 64,
        embed_dim: int = 10,
        num_layers: int = 4,
        weight_norm: bool = True,
        model_path: Optional[str] = None,
        batch_size: int = 256,
        seed: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        residual_connections: bool = False,
        **kwargs: Any
    ):
        """
        Initialize UniRep model configuration.

        Args:
            rnn_size: Number of units in RNN cell
            embed_dim: Dimension of embedding layer
            num_layers: Number of layers in stacked model
            weight_norm: Whether to use weight normalization
            model_path: Path to pretrained weights
            batch_size: Default batch size
            seed: Random seed for deterministic behavior
            dropout_rate: Optional dropout rate
            residual_connections: Whether to use residual connections
        """
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.weight_norm = weight_norm
        self.model_path = model_path
        self.batch_size = batch_size
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.residual_connections = residual_connections
        super(UniRepModelConfig, self).__init__(**kwargs)

    def validate(self) -> bool:
        """Validate UniRep model configuration."""
        # Validate RNN size
        if self.rnn_size <= 0:
            raise ValueError(f"rnn_size must be positive, got {self.rnn_size}")

        # Validate embedding dimension
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        # Validate number of layers
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate dropout rate
        if self.dropout_rate is not None and (self.dropout_rate < 0 or self.dropout_rate >= 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        # Validate model path
        if self.model_path is not None and not os.path.exists(self.model_path):
            raise ValueError(f"model_path does not exist: {self.model_path}")

        return True