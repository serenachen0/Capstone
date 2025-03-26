# Implement tensor utilities (including deterministic operation handling)
import tensorflow as tf

def get_tensor_shape(tensor: tf.Tensor) -> List[int]:
    """
    Get the shape of a tensor, handling both static and dynamic dimensions.
    
    Args:
        tensor: Input tensor
        
    Returns:
        List of dimension sizes
    """
    # Get static shape where available
    static_shape = tensor.shape.as_list()
    
    # Get dynamic shape for dimensions that are None in static shape
    dynamic_shape = tf.unstack(tf.shape(tensor))
    
    # Combine static and dynamic dimensions
    dims = [s[1] if s[0] is None else s[0] 
            for s in zip(static_shape, dynamic_shape)]
    
    return dims

def sample_with_temperature(
    logits: tf.Tensor,
    temperature: float,
    seed: Optional[int] = None
    ) -> tf.Tensor: 
    """
    Sample from logits with temperature adjustment.
    
    Args:
        logits: Raw logits tensor
        temperature: Temperature parameter (0-1, lower = more conservative)
        seed: Random seed for deterministic sampling
        
    Returns:
        Sampled indices tensor
    """
    # Adjust logits by temperature
    scaled_logits = logits / max(temperature, 1e-8)  # Avoid division by zero
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(scaled_logits, axis=-1)
    
    # Sample from categorical distribution (updated for TF 2.x)
    return tf.random.categorical(
        tf.math.log(probabilities + 1e-8),  # Add small value to avoid log(0)
        num_samples=1,
        seed=seed
    )


def deterministic_reduce_mean(tensor, axis=None, keepdims=False):
    """A deterministic implementation of reduce_mean."""
    # The original TF reduce_mean can be non-deterministic in some GPU operations
    # This implementation ensures determinism
    return tf.reduce_sum(tensor, axis=axis, keepdims=keepdims) / tf.cast(
        tf.reduce_prod(tf.gather(tf.shape(tensor), axis)), dtype=tensor.dtype
    )

def set_deterministic_ops():
    """Configure TensorFlow for deterministic operations."""
    # TF 2.x approach
    if hasattr(tf.config.experimental, 'enable_op_determinism'):
        tf.config.experimental.enable_op_determinism()
    else:
        # Fallback for older TF versions
        os.environ['TF_DETERMINISTIC_OPS'] = '1'