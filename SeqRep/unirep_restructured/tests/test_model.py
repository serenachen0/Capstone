# test_model.py
import os
import tempfile
import numpy as np
import tensorflow as tf
from typing import Dict, Any

from config.model_config import BaseConfig, UniRepModelConfig
from models.unirep64 import UniRep64
from models.unirep256 import UniRep256
from models.unirep1900 import UniRep1900


def create_mock_weights(directory: str, size: int) -> None:
    """
    Create mock weight files for testing in the specified directory.

    Args:
        directory: Directory to create weight files in
        size: Size of the model (64, 256, or 1900)
    """
    # Create embedding matrix
    embed_matrix = np.ones((26, 10)) * 0.01
    np.save(os.path.join(directory, "embed_matrix:0.npy"), embed_matrix)

    # Create output projection weights
    fc_weights = np.ones((size, 25)) * 0.02  # 25 = 26 - 1 (no start token)
    fc_biases = np.ones(25) * 0.001
    np.save(os.path.join(directory, "fully_connected_weights:0.npy"), fc_weights)
    np.save(os.path.join(directory, "fully_connected_biases:0.npy"), fc_biases)

    # Create cell weights based on model size
    if size == 1900:
        # Single cell
        wx = np.ones((10, 1900 * 4)) * 0.03
        wh = np.ones((1900, 1900 * 4)) * 0.04
        wmx = np.ones((10, 1900)) * 0.05
        wmh = np.ones((1900, 1900)) * 0.06
        b = np.ones(1900 * 4) * 0.07

        np.save(os.path.join(directory, "rnn_mlstm_mlstm_wx:0.npy"), wx)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_wh:0.npy"), wh)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_wmx:0.npy"), wmx)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_wmh:0.npy"), wmh)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_b:0.npy"), b)

        # Weight normalization parameters
        gx = np.ones(1900 * 4) * 1.0
        gh = np.ones(1900 * 4) * 1.0
        gmx = np.ones(1900) * 1.0
        gmh = np.ones(1900) * 1.0

        np.save(os.path.join(directory, "rnn_mlstm_mlstm_gx:0.npy"), gx)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_gh:0.npy"), gh)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_gmx:0.npy"), gmx)
        np.save(os.path.join(directory, "rnn_mlstm_mlstm_gmh:0.npy"), gmh)
    else:
        # Stacked cell (4 layers)
        for i in range(4):
            layer_dir = os.path.join(directory, f"layer_{i}")
            os.makedirs(layer_dir, exist_ok=True)

            base_name = f"rnn_mlstm_stack_mlstm_stack{i}_mlstm_stack{i}"

            wx = np.ones((10 if i == 0 else size, size * 4)) * (0.03 + i * 0.001)
            wh = np.ones((size, size * 4)) * (0.04 + i * 0.001)
            wmx = np.ones((10 if i == 0 else size, size)) * (0.05 + i * 0.001)
            wmh = np.ones((size, size)) * (0.06 + i * 0.001)
            b = np.ones(size * 4) * (0.07 + i * 0.001)

            np.save(os.path.join(directory, f"{base_name}_wx:0.npy"), wx)
            np.save(os.path.join(directory, f"{base_name}_wh:0.npy"), wh)
            np.save(os.path.join(directory, f"{base_name}_wmx:0.npy"), wmx)
            np.save(os.path.join(directory, f"{base_name}_wmh:0.npy"), wmh)
            np.save(os.path.join(directory, f"{base_name}_b:0.npy"), b)

            # Weight normalization parameters
            gx = np.ones(size * 4) * 1.0
            gh = np.ones(size * 4) * 1.0
            gmx = np.ones(size) * 1.0
            gmh = np.ones(size) * 1.0

            np.save(os.path.join(directory, f"{base_name}_gx:0.npy"), gx)
            np.save(os.path.join(directory, f"{base_name}_gh:0.npy"), gh)
            np.save(os.path.join(directory, f"{base_name}_gmx:0.npy"), gmx)
            np.save(os.path.join(directory, f"{base_name}_gmh:0.npy"), gmh)


def test_base_config():
    print("\n=== Testing BaseConfig ===")
    # Create a BaseConfig instance
    config = BaseConfig(param1=42, param2="test")

    # Test to_dict
    config_dict = config.to_dict()
    print(f"Config dictionary: {config_dict}")
    assert config_dict["param1"] == 42, "param1 value mismatch"
    assert config_dict["param2"] == "test", "param2 value mismatch"

    # Test save and load
    temp_path = "temp_config.json"
    config.save(temp_path)
    print(f"Config saved to {temp_path}")

    loaded_config = BaseConfig.load(temp_path)
    print(f"Loaded config: {loaded_config.to_dict()}")
    assert loaded_config.param1 == 42, "Loaded param1 value mismatch"
    assert loaded_config.param2 == "test", "Loaded param2 value mismatch"

    # Clean up
    os.remove(temp_path)
    print("BaseConfig tests passed!")


def test_unirep_model_config():
    print("\n=== Testing UniRepModelConfig ===")
    # Create a UniRepModelConfig instance
    config = UniRepModelConfig(
        rnn_size=64,
        embed_dim=10,
        num_layers=4,
        batch_size=256,
        dropout_rate=0.5,
        model_path="./weights"
    )

    # Test validate
    print("Validating configuration...")
    try:
        assert config.validate(), "Validation failed for valid configuration"
    except ValueError as e:
        if "model_path does not exist" in str(e):
            print("Note: Expected error about model_path not existing - this is normal for testing")
            # Create a temporary path for model_path testing
            with tempfile.TemporaryDirectory() as temp_dir:
                config.model_path = temp_dir
                assert config.validate(), "Validation failed with valid model_path"
        else:
            raise  # Re-raise if it's a different error

    # Test invalid configurations
    try:
        invalid_config = UniRepModelConfig(rnn_size=-1)
        invalid_config.validate()
        assert False, "Validation did not fail for invalid rnn_size"
    except ValueError as e:
        print(f"Expected error for invalid rnn_size: {e}")

    try:
        invalid_config = UniRepModelConfig(embed_dim=0)
        invalid_config.validate()
        assert False, "Validation did not fail for invalid embed_dim"
    except ValueError as e:
        print(f"Expected error for invalid embed_dim: {e}")

    try:
        invalid_config = UniRepModelConfig(num_layers=-1)
        invalid_config.validate()
        assert False, "Validation did not fail for invalid num_layers"
    except ValueError as e:
        print(f"Expected error for invalid num_layers: {e}")

    try:
        invalid_config = UniRepModelConfig(batch_size=0)
        invalid_config.validate()
        assert False, "Validation did not fail for invalid batch_size"
    except ValueError as e:
        print(f"Expected error for invalid batch_size: {e}")

    try:
        invalid_config = UniRepModelConfig(dropout_rate=1.5)
        invalid_config.validate()
        assert False, "Validation did not fail for invalid dropout_rate"
    except ValueError as e:
        print(f"Expected error for invalid dropout_rate: {e}")

    # Reset config to valid values
    with tempfile.TemporaryDirectory() as temp_dir:
        config = UniRepModelConfig(
            rnn_size=64,
            embed_dim=10,
            num_layers=4,
            batch_size=256,
            dropout_rate=0.5,
            model_path=temp_dir
        )

        # Test to_dict and from_dict
        config_dict = config.to_dict()
        print(f"Config dictionary: {config_dict}")
        new_config = UniRepModelConfig.from_dict(config_dict)
        print(f"New config from dictionary: {new_config.to_dict()}")
        assert new_config.rnn_size == 64, "rnn_size mismatch"
        assert new_config.embed_dim == 10, "embed_dim mismatch"
        assert new_config.num_layers == 4, "num_layers mismatch"
        assert new_config.batch_size == 256, "batch_size mismatch"
        assert new_config.dropout_rate == 0.5, "dropout_rate mismatch"
        assert new_config.model_path == temp_dir, "model_path mismatch"

        # Test save and load
        temp_path = "temp_unirep_config.json"
        config.save(temp_path)
        print(f"Config saved to {temp_path}")

        loaded_config = UniRepModelConfig.load(temp_path)
        print(f"Loaded config: {loaded_config.to_dict()}")
        assert loaded_config.rnn_size == 64, "Loaded rnn_size mismatch"
        assert loaded_config.embed_dim == 10, "Loaded embed_dim mismatch"
        assert loaded_config.num_layers == 4, "Loaded num_layers mismatch"
        assert loaded_config.batch_size == 256, "Loaded batch_size mismatch"
        assert loaded_config.dropout_rate == 0.5, "Loaded dropout_rate mismatch"
        assert loaded_config.model_path == temp_dir, "Loaded model_path mismatch"

        # Clean up
        os.remove(temp_path)

    print("UniRepModelConfig tests passed!")


def test_model_initialization():
    """Test model initialization with configurations."""
    print("\n=== Testing Model Initialization ===")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create configs with fixed seeds
    config64 = UniRepModelConfig(
        rnn_size=64,
        embed_dim=10,
        num_layers=4,
        seed=42
    )

    config256 = UniRepModelConfig(
        rnn_size=256,
        embed_dim=10,
        num_layers=4,
        seed=42
    )

    config1900 = UniRepModelConfig(
        rnn_size=1900,
        embed_dim=10,
        num_layers=1,
        seed=42
    )

    # Create models with configs
    print("Creating UniRep64 model...")
    model64 = UniRep64(config=config64)
    print("Creating UniRep256 model...")
    model256 = UniRep256(config=config256)
    print("Creating UniRep1900 model...")
    model1900 = UniRep1900(config=config1900)

    # Check model configuration
    assert model64.config.rnn_size == 64, "UniRep64 rnn_size mismatch"
    assert model256.config.rnn_size == 256, "UniRep256 rnn_size mismatch"
    assert model1900.config.rnn_size == 1900, "UniRep1900 rnn_size mismatch"

    # Check correct number of layers
    assert model64.config.num_layers == 4, "UniRep64 num_layers mismatch"
    assert model256.config.num_layers == 4, "UniRep256 num_layers mismatch"
    assert model1900.config.num_layers == 1, "UniRep1900 num_layers mismatch"

    # Ensure all models created the right type of cell
    assert isinstance(model64.cell, tf.keras.layers.Layer), "UniRep64 cell type mismatch"
    assert isinstance(model256.cell, tf.keras.layers.Layer), "UniRep256 cell type mismatch"
    assert isinstance(model1900.cell, tf.keras.layers.Layer), "UniRep1900 cell type mismatch"

    print("Model initialization tests passed!")


def test_model_forward_pass():
    """Test model forward passes with input data."""
    print("\n=== Testing Model Forward Pass ===")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create models with smaller sizes for faster testing
    config64 = UniRepModelConfig(rnn_size=64, embed_dim=10, num_layers=2, seed=42)
    model64 = UniRep64(config=config64)

    # Create a simple input tensor
    # Shape: [batch_size=2, seq_len=5]
    inputs = tf.constant([
        [24, 1, 15, 7, 4],  # Start, M, A, S, K
        [24, 13, 6, 6, 21]   # Start, G, E, E, L
    ], dtype=tf.int32)

    # Check forward pass output shapes
    print("Testing forward pass output shapes...")
    outputs64 = model64(inputs, training=False)

    # Check output shapes
    assert outputs64.shape == (2, 5, 64), f"Expected shape (2, 5, 64), got {outputs64.shape}"

    # Check forward pass with state
    print("Testing forward pass with state...")
    outputs64_state, state64 = model64(inputs, training=False, return_state=True)

    # Check states are returned correctly
    assert isinstance(state64, tuple), "State should be a tuple"

    # Make sure outputs are the same with or without state returned
    outputs_equal = np.allclose(outputs64.numpy(), outputs64_state.numpy())
    assert outputs_equal, "Outputs differ between with/without state returned"

    print("Model forward pass tests passed!")


def test_get_representation():
    """Test getting representations from sequences."""
    print("\n=== Testing Model Representation Generation ===")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create models with smaller sizes for faster testing
    config64 = UniRepModelConfig(rnn_size=64, embed_dim=10, num_layers=2, seed=42)
    model64 = UniRep64(config=config64)

    # Test sequence
    test_seq = "MASKGEELFT"

    # Get representations
    print(f"Getting representation for sequence: {test_seq}")
    avg64, final_h64, final_c64 = model64.get_representation(test_seq)

    # Check shapes
    assert avg64.shape == (64,), f"Expected avg_hidden shape (64,), got {avg64.shape}"
    assert final_h64.shape == (64,), f"Expected final_hidden shape (64,), got {final_h64.shape}"
    assert final_c64.shape == (64,), f"Expected final_cell shape (64,), got {final_c64.shape}"

    # Check determinism - run again and make sure we get identical results
    print("Checking deterministic behavior...")
    avg64_2, final_h64_2, final_c64_2 = model64.get_representation(test_seq)

    # Verify all results are identical to the first run
    avg_equal = np.allclose(avg64, avg64_2)
    h_equal = np.allclose(final_h64, final_h64_2)
    c_equal = np.allclose(final_c64, final_c64_2)

    assert avg_equal, "Average hidden representations not deterministic"
    assert h_equal, "Final hidden representations not deterministic"
    assert c_equal, "Final cell representations not deterministic"

    print("Model representation tests passed!")


def test_sequence_generation():
    """Test deterministic sequence generation."""
    print("\n=== Testing Sequence Generation ===")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enable deterministic ops
    os.environ['PYTHONHASHSEED'] = '0'  # Fix Python hash seed

    # Create model with fixed seed
    config64 = UniRepModelConfig(rnn_size=64, embed_dim=10, num_layers=2, seed=42)
    model64 = UniRep64(config=config64)

    # For testing, use fixed logits instead of model output
        # This eliminates model initialization as a source of randomness
    def simple_deterministic_test():
        # Create a fixed logit tensor
        fixed_logits = tf.constant([1.0, 2.0, 0.5, 1.5, 0.8] * 5)  # 25 values

        # Sample twice with same temperature
        sample1 = model64._sample_with_temperature(fixed_logits, 0.8)
        sample2 = model64._sample_with_temperature(fixed_logits, 0.8)

        print(f"Sample 1: {sample1}")
        print(f"Sample 2: {sample2}")

        # These should be identical
        return sample1 == sample2

    print("Testing sampling determinism...")
    assert simple_deterministic_test(), "Sampling is not deterministic"

    print("Sampling is deterministic. Sequence generation test passed!")

    # Generate sequences with the same seed
    print("Generating sequences...")
    seed_seq = "MAS"
    seq1 = model64.generate_sequence(seed_seq, length=10, temperature=0.8)
    seq2 = model64.generate_sequence(seed_seq, length=10, temperature=0.8)

    # Sequences should be identical with the same seed
    print(f"Generated sequence 1: {seq1}")
    print(f"Generated sequence 2: {seq2}")
    assert seq1 == seq2, "Sequence generation not deterministic"

    # Check sequence properties
    assert len(seq1) == 10, f"Expected length 10, got {len(seq1)}"
    assert model64.is_valid_sequence(seq1), "Generated sequence is not valid"

    # Test with different temperature
    seq3 = model64.generate_sequence(seed_seq, length=10, temperature=0.1)
    print(f"Generated sequence with different temperature: {seq3}")

    # Not guaranteed, but likely to be different
    if seq1 != seq3:
        print("Different temperature produced different sequence (expected)")

    print("Sequence generation tests passed!")


def test_weight_loading():
    """Test loading weights from mock files."""
    print("\n=== Testing Weight Loading ===")

    # Create temporary directory with mock weights
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create mock weights for smaller model for faster testing
        print(f"Creating mock weights in {tmp_dir}")
        create_mock_weights(tmp_dir, 64)

        # Create config with mock weight path
        config64 = UniRepModelConfig(
            rnn_size=64,
            embed_dim=10,
            num_layers=4,
            seed=42,
            model_path=tmp_dir
        )

        # Create model with mock weights
        print("Creating model with mock weights...")
        model64 = UniRep64(config=config64)

        # Process a sequence to ensure weights are used
        test_seq = "MASKGEEL"
        print(f"Testing representation with loaded weights for sequence: {test_seq}")
        avg64, _, _ = model64.get_representation(test_seq)

        # Check that we get results of the right shape
        assert avg64.shape == (64,), f"Expected shape (64,), got {avg64.shape}"

        print("Weight loading tests passed!")


def run_all_tests():
    print("\n=== Running all tests ===")
    # Test configuration classes
    test_base_config()
    test_unirep_model_config()

    # Test model implementations
    test_model_initialization()
    test_model_forward_pass()
    test_get_representation()
    test_sequence_generation()
    test_weight_loading()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()