# cells/mlstm_stack.py
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Any, Dict
import numpy as np
import os
from .mlstm_cell import mLSTMCell

class StackedMlstmCell(tf.keras.layers.Layer):
    """
    Stacked mLSTM cell implementation with residual connections and dropout support.
    Creates multiple mLSTM cells and manages their interaction.
    """
    
    def __init__(
        self,
        num_units: int = 64,
        num_layers: int = 4,
        dropout_rate: Optional[float] = None,
        residual_connections: bool = False,
        weight_norm: bool = True,
        seed: Optional[int] = None,
        name: str = "stacked_mlstm",
        **kwargs: Any
    ):
        """
        Initialize a stack of mLSTM cells.
        
        Args:
            num_units: Number of units in each mLSTM cell
            num_layers: Number of layers in the stack
            dropout_rate: Optional dropout rate for regularization
            residual_connections: Whether to use residual connections
            weight_norm: Whether to use weight normalization
            seed: Random seed for deterministic behavior
            name: Name of the stack
        """
        super(StackedMlstmCell, self).__init__(name=name, **kwargs)
        self._num_units = num_units
        self._num_layers = num_layers
        self._dropout_rate = dropout_rate
        self._residual_connections = residual_connections
        self._weight_norm = weight_norm
        self._seed = seed
        
        # Create the stack of cells
        self._cells = []
        for i in range(self._num_layers):
            layer_seed = None if seed is None else seed + i
            cell = mLSTMCell(
                num_units=self._num_units,
                weight_norm=self._weight_norm,
                seed=layer_seed,  # Unique seed per layer
                name=f"{name}_layer_{i}"
            )
                
            self._cells.append(cell)
            
        # Create dropout layers if specified
        if self._dropout_rate is not None:
            self._dropout_layers = []
            for i in range(self._num_layers - 1):  # No dropout after last layer
                dropout_seed = None if seed is None else seed + self._num_layers + i
                self._dropout_layers.append(
                    tf.keras.layers.Dropout(
                        rate=self._dropout_rate,
                        seed=dropout_seed
                    )
                )
        else:
            self._dropout_layers = None
    
    @property
    def state_size(self) -> Tuple[Tuple[tf.TensorShape, ...], Tuple[tf.TensorShape, ...]]:
        """Return size of cell and hidden states for all layers."""
        # Each layer has its own cell and hidden state
        cell_sizes = tuple(tf.TensorShape([self._num_units]) for _ in range(self._num_layers))
        hidden_sizes = tuple(tf.TensorShape([self._num_units]) for _ in range(self._num_layers))
        return (cell_sizes, hidden_sizes)
    
    @property
    def output_size(self) -> tf.TensorShape:
        """Return size of output (hidden state of the last layer)."""
        return tf.TensorShape([self._num_units])
    
    def get_initial_state(
        self,
        inputs: Optional[tf.Tensor] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[tf.DType] = None
    ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        """
        Create zero states for all layers.
        
        Args:
            inputs: Optional inputs tensor to infer batch size
            batch_size: Optional explicit batch size
            dtype: Optional data type
            
        Returns:
            Tuple of (cell_states, hidden_states) for all layers
        """
        if batch_size is None:
            if inputs is not None:
                batch_size = tf.shape(inputs)[0]
            else:
                raise ValueError("Either inputs or batch_size must be provided")
                
        if dtype is None:
            if inputs is not None:
                dtype = inputs.dtype
            else:
                dtype = tf.float32
                
        # Create zero states for each layer
        cell_states = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) 
                            for _ in range(self._num_layers))
        hidden_states = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) 
                              for _ in range(self._num_layers))
                              
        return (cell_states, hidden_states)
    
    def call(
        self,
        inputs: tf.Tensor,
        states: Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]],
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]]:
        """
        Run one step of the stacked mLSTM cell.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_dim]
            states: Tuple of (cell_states, hidden_states) for all layers
            training: Whether in training mode (for dropout)
            
        Returns:
            Tuple of (output, next_states)
        """
        # Unpack states
        cell_states, hidden_states = states
        
        new_cell_states = []
        new_hidden_states = []
        outputs = []
        
        layer_input = inputs
        
        # Process each layer
        for i, cell in enumerate(self._cells):
            cell_state = cell_states[i]
            hidden_state = hidden_states[i]
            
            # Apply the cell
            output, (new_cell, new_hidden) = cell(
                layer_input, 
                (cell_state, hidden_state),
                training=training
            )
            
            # Store new states
            new_cell_states.append(new_cell)
            new_hidden_states.append(new_hidden)
            outputs.append(output)

             # Apply dropout to the output (except for the last layer)
            if i < self._num_layers - 1 and self._dropout_rate is not None:
                layer_input = self._dropout_layers[i](output, training=training)
            else:
                layer_input = output
            
        
        # Apply residual connections if specified
        if self._residual_connections:
            # Scale output to maintain magnitude
            scale_factor = 1.0 / float(self._num_layers)
            final_output = tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)
        else:
            # Just use the output from the last layer
            final_output = outputs[-1]
        
        return final_output, (tuple(new_cell_states), tuple(new_hidden_states))
    
    def load_weights_from_numpy(self, load_path: str) -> None:
        """
        Load cell weights from numpy files.
        
        Args:
            load_path: Directory containing weight files
        """
        base_scope = "rnn_mlstm_stack_mlstm_stack"
        
        # Load weights for each cell in the stack
        for i, cell in enumerate(self._cells):
            # Path prefix for this layer's weights
            layer_prefix = f"{base_scope}{i}_mlstm_stack{i}_"
            
            # Create a subdirectory for the cell weights
            cell_path = os.path.join(load_path, f"layer_{i}")
            os.makedirs(cell_path, exist_ok=True)
            
            # Map original weight names to cell weight names
            weight_mapping = {
                f"{layer_prefix}wx:0.npy": "rnn_mlstm_mlstm_wx:0.npy",
                f"{layer_prefix}wh:0.npy": "rnn_mlstm_mlstm_wh:0.npy",
                f"{layer_prefix}wmx:0.npy": "rnn_mlstm_mlstm_wmx:0.npy",
                f"{layer_prefix}wmh:0.npy": "rnn_mlstm_mlstm_wmh:0.npy",
                f"{layer_prefix}b:0.npy": "rnn_mlstm_mlstm_b:0.npy"
            }
            
            if self._weight_norm:
                weight_mapping.update({
                    f"{layer_prefix}gx:0.npy": "rnn_mlstm_mlstm_gx:0.npy",
                    f"{layer_prefix}gh:0.npy": "rnn_mlstm_mlstm_gh:0.npy",
                    f"{layer_prefix}gmx:0.npy": "rnn_mlstm_mlstm_gmx:0.npy",
                    f"{layer_prefix}gmh:0.npy": "rnn_mlstm_mlstm_gmh:0.npy"
                })
            
            # Link original weights to cell weights
            for orig_name, cell_name in weight_mapping.items():
                orig_path = os.path.join(load_path, orig_name)
                cell_weight_path = os.path.join(cell_path, cell_name)
                
                if os.path.exists(orig_path):
                    # Copy the weight data to the cell-specific path
                    weight_data = np.load(orig_path)
                    np.save(cell_weight_path, weight_data)
            
            # Load the weights into the cell
            cell.load_weights_from_numpy(cell_path)