# cells/base_cell.py
import tensorflow as tf
from typing import Tuple, List, Optional, Union, Any, Dict
import numpy as np

class BaseLSTMCell(tf.keras.layers.Layer):
    """
    Base LSTM cell implementation that serves as a foundation for specialized cells.
    Implements weight normalization if specified.
    """
    
    def __init__(
        self,
        num_units: int,
        weight_norm: bool = True,
        name: str = "base_lstm",
        **kwargs: Any
    ):
        """
        Initialize BaseLSTMCell with configuration.
        
        Args:
            num_units: Number of units in the LSTM cell
            weight_norm: Whether to use weight normalization
            name: Name of the cell
        """
        super(BaseLSTMCell, self).__init__(name=name, **kwargs)
        self._num_units = num_units
        self._weight_norm = weight_norm
        
    @property
    def state_size(self) -> Tuple[tf.TensorShape, tf.TensorShape]:
        """Return size of cell and hidden states."""
        return (tf.TensorShape([self._num_units]), tf.TensorShape([self._num_units]))
    
    @property
    def output_size(self) -> tf.TensorShape:
        """Return size of output (hidden state)."""
        return tf.TensorShape([self._num_units])

    
    def get_initial_state(
        self, 
        inputs: Optional[tf.Tensor] = None, 
        batch_size: Optional[int] = None, 
        dtype: Optional[tf.DType] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create zero states with batch dimension.
        
        Args:
            inputs: Optional inputs tensor to infer batch size
            batch_size: Optional explicit batch size
            dtype: Optional data type
            
        Returns:
            Tuple of (cell_state, hidden_state)
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
                
        # Create zero states
        c: tf.Tensor = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h: tf.Tensor = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    
    def apply_weight_norm(self, weight: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
        """
        Apply weight normalization to the weight tensor.
        
        Args:
            weight: Weight tensor to normalize
            scale: Scale factor for normalized weights
            
        Returns:
            Normalized weight tensor
        """
        if self._weight_norm:
            # L2 normalize across first dimension (TF 2.x uses axis instead of dim)
            normalized_weight = tf.nn.l2_normalize(weight, axis=0) * scale
            return normalized_weight
        else:
            return weight

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the cell weights based on input shape.
        This method should be overridden by subclasses to create weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        # Base implementation just calls parent's build method
        super(BaseLSTMCell, self).build(input_shape)
    
    def call(
        self, 
        inputs: tf.Tensor, 
        states: Tuple[tf.Tensor, tf.Tensor], 
        training: Optional[bool]=None) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Run one step of the LSTM cell.
        This method must be implemented by subclasses.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_dim]
            states: Tuple of (cell_state, hidden_state)
            training: Whether in training mode
            
        Returns:
            Tuple of (output, next_state)
        """
        raise NotImplementedError("BaseLSTMCell is an abstract class. Subclasses must implement call().")
    
    def deterministic_reduce_mean(
        self, 
        tensor: tf.Tensor, 
        axis: Optional[Union[int, List[int]]] = None, 
        keepdims: bool = False) -> tf.Tensor:
        """
        A deterministic implementation of reduce_mean.
        
        Args:
            tensor: Input tensor
            axis: Dimensions to reduce over
            keepdims: Whether to keep reduced dimensions
            
        Returns:
            Reduced tensor
        """
        # The original TF reduce_mean can be non-deterministic in some GPU operations
        # This implementation ensures determinism
        return tf.reduce_sum(tensor, axis=axis, keepdims=keepdims) / tf.cast(
            tf.reduce_prod(tf.gather(tf.shape(tensor), axis)), dtype=tensor.dtype
        )
    def set_seed(self, seed: Optional[int]=None) -> None:
        """
        Set random seed for deterministic behavior.
    
        Args:
            seed: Optional random seed
        """
        if seed is not None:
            tf.random.set_seed(seed)