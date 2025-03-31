# cells/mlstm_cell.py
import tensorflow as tf
import numpy as np
import os
from typing import Tuple, List, Optional, Union, Any, Dict
from .base_cell import BaseLSTMCell

class mLSTMCell(BaseLSTMCell):
    """
    Multiplicative LSTM Cell implementation.
    Implements a multiplicative interaction between input and previous hidden state.
    """
    
    def __init__(
        self,
        num_units: int,
        weight_norm: bool = True,
        seed: Optional[int] = None,
        name: str = "mlstm",
        **kwargs: Any
    ):
        """
        Initialize mLSTMCell with configuration.
        
        Args:
            num_units: Number of units in the LSTM cell
            weight_norm: Whether to use weight normalization
            seed: Random seed for deterministic behavior
            name: Name of the cell
        """
        super(mLSTMCell, self).__init__(
            num_units=num_units, 
            weight_norm=weight_norm,
            name=name,
            **kwargs
        )
        self._seed = seed
        self.set_seed(seed)
        
    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the cell weights based on input shape.
        
        Args:
            input_shape: Shape of the input tensor [batch_size, input_dim]
        """
        input_dim = input_shape[-1]
        
        # Create weight initializers
        ortho_init = tf.initializers.orthogonal(seed=self._seed)
        ones_init = tf.ones_initializer()
        
        # Input weights
        self.wx = self.add_weight(
            name="wx",
            shape=[input_dim, self._num_units * 4],
            initializer=ortho_init
        )
        
        # Hidden weights
        self.wh = self.add_weight(
            name="wh",
            shape=[self._num_units, self._num_units * 4],
            initializer=ortho_init
        )
        
        # Multiplicative weights for input
        self.wmx = self.add_weight(
            name="wmx",
            shape=[input_dim, self._num_units],
            initializer=ortho_init
        )
        
        # Multiplicative weights for hidden
        self.wmh = self.add_weight(
            name="wmh",
            shape=[self._num_units, self._num_units],
            initializer=ortho_init
        )
        
        # Bias
        self.b = self.add_weight(
            name="b",
            shape=[self._num_units * 4],
            initializer=tf.zeros_initializer()
        )
        
        # Gain parameters for weight normalization
        if self._weight_norm:
            self.gx = self.add_weight(
                name="gx",
                shape=[self._num_units * 4],
                initializer=ones_init
            )
            
            self.gh = self.add_weight(
                name="gh",
                shape=[self._num_units * 4],
                initializer=ones_init
            )
            
            self.gmx = self.add_weight(
                name="gmx",
                shape=[self._num_units],
                initializer=ones_init
            )
            
            self.gmh = self.add_weight(
                name="gmh",
                shape=[self._num_units],
                initializer=ones_init
            )
        
        self.built = True
        
    def call(
        self, 
        inputs: tf.Tensor, 
        states: Tuple[tf.Tensor, tf.Tensor], 
        training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Run one step of the mLSTM cell.
        
        Args:
            inputs: Input tensor of shape [batch_size, input_dim]
            states: Tuple of (cell_state, hidden_state)
            training: Whether in training mode
            
        Returns:
            Tuple of (output, next_state)
        """
        # Unpack states
        c_prev, h_prev = states
        
        # Apply weight normalization if enabled
        if self._weight_norm:
            wx = self.apply_weight_norm(self.wx, self.gx)
            wh = self.apply_weight_norm(self.wh, self.gh)
            wmx = self.apply_weight_norm(self.wmx, self.gmx)
            wmh = self.apply_weight_norm(self.wmh, self.gmh)
        else:
            wx = self.wx
            wh = self.wh
            wmx = self.wmx
            wmh = self.wmh
        
        # Multiplicative interaction
        m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        
        # Input, forget, output, and cell gates
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + self.b
        i, f, o, u = tf.split(z, 4, axis=1)
        
        # Apply activation functions
        i = tf.nn.sigmoid(i)  # Input gate
        f = tf.nn.sigmoid(f)  # Forget gate
        o = tf.nn.sigmoid(o)  # Output gate
        u = tf.tanh(u)        # Cell update
        
        # Update cell state
        c = f * c_prev + i * u
        
        # Update hidden state
        h = o * tf.tanh(c)
        
        return h, (c, h)
    
    def load_weights_from_numpy(self, load_path: str) -> None:
        """
        Load cell weights from numpy files.
        
        Args:
            load_path: Directory containing weight files
        """
        if not self.built:
            raise ValueError("Cell must be built before loading weights")
            
        # Helper to load weights
        def load_weight(name: str, var: tf.Variable) -> None:
            weight_file = os.path.join(load_path, f"{name}.npy")
            if os.path.exists(weight_file):
                weight_data = np.load(weight_file)
                var.assign(weight_data)
            else:
                print(f"Warning: Weight file {weight_file} not found")
        
        # Load all weights
        load_weight("rnn_mlstm_mlstm_wx:0", self.wx)
        load_weight("rnn_mlstm_mlstm_wh:0", self.wh)
        load_weight("rnn_mlstm_mlstm_wmx:0", self.wmx)
        load_weight("rnn_mlstm_mlstm_wmh:0", self.wmh)
        load_weight("rnn_mlstm_mlstm_b:0", self.b)
        
        if self._weight_norm:
            load_weight("rnn_mlstm_mlstm_gx:0", self.gx)
            load_weight("rnn_mlstm_mlstm_gh:0", self.gh)
            load_weight("rnn_mlstm_mlstm_gmx:0", self.gmx)
            load_weight("rnn_mlstm_mlstm_gmh:0", self.gmh)