# models/unirep64.py
import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any

from config.model_config import UniRepModelConfig
from models.base_model import BaseModel
from cells.mlstm_stack import StackedMlstmCell


class UniRep64(BaseModel):
    """
    UniRep model with 64 units in 4 stacked mLSTM cells.
    This is the smallest model, corresponding to the original babbler64.
    """

    def __init__(
        self,
        config: Optional[UniRepModelConfig] = None,
        **kwargs: Any
    ):
        """
        Initialize UniRep64 model.

        Args:
            config: Configuration for the model (optional)
            **kwargs: Additional arguments for BaseModel
        """
        # Create config if not provided
        if config is None:
            config = UniRepModelConfig(
                rnn_size=64,
                embed_dim=10,
                num_layers=4,
                weight_norm=True,
                **kwargs
            )
        else:
            # Override config with correct RNN size for this model
            config.rnn_size = 64
            config.num_layers = 4

        # Validate configuration
        config.validate()

        super(UniRep64, self).__init__(config=config, name="unirep64", **kwargs)

        # Create the stacked mLSTM cell
        self.cell = StackedMlstmCell(
            num_units=self.config.rnn_size,
            num_layers=self.config.num_layers,
            dropout_rate=self.config.dropout_rate,
            residual_connections=self.config.residual_connections,
            weight_norm=self.config.weight_norm,
            seed=self.config.seed,
            name="mlstm_stack_64"
        )

        # Create RNN layer with the cell
        self.rnn_layer = tf.keras.layers.RNN(
            self.cell,
            return_sequences=True,
            return_state=True,
            name="rnn_layer"
        )

        # Load weights if provided
        if self.config.model_path:
            self._build_and_load_weights()

    def call(
        self,
        inputs: tf.Tensor,
        initial_state: Optional[Tuple] = None,
        training: bool = False,
        return_state: bool = False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, Tuple]]:
        """
        Forward pass for the model.

        Args:
            inputs: Input tensor of token IDs [batch_size, seq_len]
            initial_state: Optional initial state for the RNN
            training: Whether in training mode
            return_state: Whether to return final state

        Returns:
            If return_state is False, returns output tensor [batch_size, seq_len, num_units]
            If return_state is True, returns (output, final_state)
        """
        # Embed the inputs
        embedded = self.embedding(inputs)

        # Use initial_state if provided, otherwise get default initial state
        if initial_state is None:
            initial_state = self.get_initial_state(tf.shape(inputs)[0])

        # Process through RNN
        outputs, *states = self.rnn_layer(
            embedded,
            initial_state=initial_state,
            training=training
        )

        # For stacked mLSTM, the states are a nested tuple:
        # ([cell_state_layer1, ..., cell_state_layerN], [hidden_state_layer1, ..., hidden_state_layerN])
        final_state = tuple(states)

        if return_state:
            return outputs, final_state
        else:
            return outputs

    def get_initial_state(self, batch_size: int) -> Tuple:
        """
        Get initial state for the RNN.

        Args:
            batch_size: Batch size

        Returns:
            Initial state tuple
        """
        return self.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def _process_final_state(self, state: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process final state to extract hidden and cell representations.

        Args:
            state: Final RNN state from stacked mLSTM

        Returns:
            Tuple of (final_hidden, final_cell) as numpy arrays
        """
        # For stacked models, state is (cell_states, hidden_states)
        # where each is a tuple of states for each layer
        cell_states, hidden_states = state

        # Get the state from the last layer
        final_cell = cell_states[-1]
        final_hidden = hidden_states[-1]

        # Convert to numpy and remove batch dimension
        return final_hidden.numpy()[0], final_cell.numpy()[0]

    def _build_and_load_weights(self) -> None:
        """Build the model and load weights from the specified path."""
        # Create a small dummy sequence input
        dummy_batch_size = 1
        dummy_seq_len = 1
        dummy_seq = tf.zeros((dummy_batch_size, dummy_seq_len), dtype=tf.int32)

        # Process through the full model to build weights
        dummy_state = self.get_initial_state(batch_size=dummy_batch_size)

        # Call the model once with the dummy input to initialize all weights
        self(dummy_seq, initial_state=dummy_state, training=False)

        # Now load the weights
        if hasattr(self.cell, 'load_weights_from_numpy'):
            self.cell.load_weights_from_numpy(self.config.model_path)

    def save_weights_to_numpy(self, save_path: str) -> None:
        """
        Save model weights to numpy files for compatibility.

        Args:
            save_path: Directory to save weights
        """
        # Call parent method to save embedding and output projection weights
        super().save_weights_to_numpy(save_path)

        # Save stacked cell-specific weights
        # Implementation would need to save each layer's weights