# models/base_model.py
import tensorflow as tf
import numpy as np
import os
from typing import Tuple, List, Dict, Optional, Union, Any

from config.model_config import UniRepModelConfig
from utils.sequence import aa_to_int, int_to_aa, aa_seq_to_int, int_to_aa_seq
from cells.mlstm_cell import mLSTMCell
from cells.mlstm_stack import StackedMlstmCell

class BaseModel(tf.keras.Model):
    """
    Base model for UniRep implementation.
    Provides common functionality for all UniRep models.
    """

    def __init__(
        self,
        config: UniRepModelConfig,
        name: str = "unirep_base",
        **kwargs: Any
    ):
        """
        Initialize the base model with configuration.

        Args:
            config: Configuration for the model
            name: Name of the model
        """
        super(BaseModel, self).__init__(name=name, **kwargs)
        self.config = config

        # Common attributes for all models
        self._vocab_size = 26  # 20 standard AAs + special tokens
        self._embed_dim = config.embed_dim

        # Set random seed for deterministic behavior
        if config.seed is not None:
            tf.random.set_seed(config.seed)
            np.random.seed(config.seed)

        # Create the embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self._vocab_size,
            output_dim=self._embed_dim,
            name="aa_embedding"
        )

        # Initialize with the weights specified in config if provided
        if config.model_path and os.path.exists(os.path.join(config.model_path, "embed_matrix:0.npy")):
            embed_weights = np.load(os.path.join(config.model_path, "embed_matrix:0.npy"))
            self.embedding.build((None,))
            self.embedding.set_weights([embed_weights])

        # Output projection for aa prediction (for babbling)
        self.output_projection = tf.keras.layers.Dense(
            units=self._vocab_size - 1,  # -1 because we don't predict the start token
            name="output_projection"
        )

        # Initialize with the weights specified in config if provided
        if config.model_path:
            weights_path = os.path.join(config.model_path, "fully_connected_weights:0.npy")
            bias_path = os.path.join(config.model_path, "fully_connected_biases:0.npy")

            if os.path.exists(weights_path) and os.path.exists(bias_path):
                weights = np.load(weights_path)
                biases = np.load(bias_path)
                self.output_projection.build((None, config.rnn_size))
                self.output_projection.set_weights([weights, biases])

        # Cell initialization is left to subclasses since it varies by model

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
        # This is a base implementation to be overridden by subclasses
        # Embed the inputs
        embedded = self.embedding(inputs)

        # The RNN processing is implemented by subclasses
        raise NotImplementedError("Subclasses must implement the call method")

    def get_initial_state(self, batch_size: int) -> Tuple:
        """
        Get initial state for the RNN.

        Args:
            batch_size: Batch size

        Returns:
            Initial state tuple
        """
        # This is a placeholder to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_initial_state")

    def get_representation(self, sequence: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get representation for a protein sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Tuple of (avg_hidden, final_hidden, final_cell)
        """
        # Convert sequence to token IDs
        sequence = sequence.strip()
        seq_ids = aa_seq_to_int(sequence, include_stop=False)

        # Add batch dimension
        inputs = tf.convert_to_tensor([seq_ids], dtype=tf.int32)

        # Get initial state for a batch size of 1
        initial_state = self.get_initial_state(batch_size=1)

        # Forward pass with state
        outputs, final_state = self(
            inputs=inputs,
            initial_state=initial_state,
            training=False,
            return_state=True
        )

        # Process outputs for representation
        # Outputs shape: [1, seq_len, rnn_size]
        outputs = outputs.numpy()[0]  # Remove batch dimension

        # Get average hidden state
        avg_hidden = np.mean(outputs, axis=0)

        # Get final hidden and cell states (specific implementation depends on model)
        final_hidden, final_cell = self._process_final_state(final_state)

        return avg_hidden, final_hidden, final_cell

    def _process_final_state(self, state: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process final state to extract hidden and cell representations.
        This method should be implemented by subclasses based on their state structure.

        Args:
            state: Final RNN state

        Returns:
            Tuple of (final_hidden, final_cell) as numpy arrays
        """
        raise NotImplementedError("Subclasses must implement _process_final_state")

    def generate_sequence(
        self,
        seed: str,
        length: int = 250,
        temperature: float = 1.0
    ) -> str:
        """
        Generate a sequence starting with seed.

        Args:
            seed: Seed sequence
            length: Total length of sequence to generate
            temperature: Sampling temperature (0-1, lower = more conservative)

        Returns:
            Generated sequence
        """
        # Initial sequence
        sequence = seed.strip()

        # Convert to token IDs
        seq_ids = aa_seq_to_int(sequence, include_stop=False)

        # Get initial state
        state = self.get_initial_state(batch_size=1)

        # Generate sequence one token at a time
        while len(sequence) < length:
            # Prepare input
            inputs = tf.convert_to_tensor([seq_ids], dtype=tf.int32)

            # Forward pass
            outputs, state = self(
                inputs=inputs,
                initial_state=state,
                training=False,
                return_state=True
            )

            # Get logits for the last token
            logits = outputs[0, -1, :]

            # Sample next token
            next_token = self._sample_with_temperature(logits, temperature)

            # Convert token to amino acid and add to sequence
            next_aa = int_to_aa[next_token + 1]  # +1 because output doesn't include start token
            sequence += next_aa

            # Update sequence IDs with only the new token for the next iteration
            seq_ids = [next_token + 1]

        return sequence

    def _sample_with_temperature(self, logits: tf.Tensor, temperature: float) -> int:
        """
        Sample from logits with temperature.

        Args:
            logits: Logits tensor
            temperature: Sampling temperature (0-1, lower = more conservative)

        Returns:
            Sampled token ID
        """
        # Adjust logits by temperature
        logits = logits / max(temperature, 1e-8)

        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits)

        # Sample from categorical distribution
        sample = tf.random.categorical(
            tf.math.log(probabilities + 1e-8)[tf.newaxis],
            num_samples=1,
            seed=self.config.seed
        )

        return sample[0, 0].numpy()

    def is_valid_sequence(self, sequence: str, max_len: int = 2000) -> bool:
        """
        Check if sequence is valid for the model.

        Args:
            sequence: Amino acid sequence
            max_len: Maximum allowed length

        Returns:
            Whether the sequence is valid
        """
        length = len(sequence)
        valid_aas = set("MRHKDESTNQCUGPAVIFYWLO")

        return (length <= max_len) and set(sequence).issubset(valid_aas)

    def save_weights_to_numpy(self, save_path: str) -> None:
        """
        Save model weights to numpy files for compatibility.

        Args:
            save_path: Directory to save weights
        """
        os.makedirs(save_path, exist_ok=True)

        # Save embedding weights
        np.save(os.path.join(save_path, "embed_matrix:0.npy"), self.embedding.get_weights()[0])

        # Save output projection weights
        weights, biases = self.output_projection.get_weights()
        np.save(os.path.join(save_path, "fully_connected_weights:0.npy"), weights)
        np.save(os.path.join(save_path, "fully_connected_biases:0.npy"), biases)

        # Cell weights are saved by subclasses