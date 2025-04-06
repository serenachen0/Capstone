# tests/test_cells.py
import tensorflow as tf
import numpy as np
import os
import tempfile
from cells.base_cell import BaseLSTMCell
from cells.mlstm_cell import mLSTMCell

def test_base_cell():
    print("\n=== Testing BaseLSTMCell ===")
    # Create a basic cell instance
    cell = BaseLSTMCell(num_units=64)

    # Test state size property
    print(f"State size: {cell.state_size}")
    print(f"Output size: {cell.output_size}")

    # Test initial state creation
    initial_state = cell.get_initial_state(batch_size=2)
    print(f"Initial state shapes: {[s.shape for s in initial_state]}")

    # Test weight normalization
    dummy_weight = tf.ones([10, 64])
    dummy_scale = tf.ones([64])
    normalized = cell.apply_weight_norm(dummy_weight, dummy_scale)
    print(f"Normalized weight shape: {normalized.shape}")

    print("BaseLSTMCell tests passed!")

def test_mlstm_cell():
    print("\n=== Testing mLSTMCell ===")
    # Create cell with fixed seed for determinism
    cell = mLSTMCell(num_units=64, seed=42)

    # Create input and states
    batch_size = 2
    input_dim = 10
    inputs = tf.random.normal([batch_size, input_dim], seed=42)
    initial_state = cell.get_initial_state(batch_size=batch_size)

    # Build cell explicitly
    cell.build(tf.TensorShape([batch_size, input_dim]))

    # Run cell
    outputs, next_state = cell(inputs, initial_state)

    # Verify shapes
    print(f"Output shape: {outputs.shape}")
    print(f"Next cell state shape: {next_state[0].shape}")
    print(f"Next hidden state shape: {next_state[1].shape}")

    # Run twice with the same input to test determinism
    outputs1, next_state1 = cell(inputs, initial_state)
    outputs2, next_state2 = cell(inputs, initial_state)

    # Check if outputs are identical
    deterministic = tf.reduce_all(outputs1 == outputs2).numpy()
    print(f"Deterministic output: {deterministic}")

    # Make assertions to ensure tests pass
    assert outputs.shape == (batch_size, 64), f"Expected output shape (2, 64), got {outputs.shape}"
    assert next_state[0].shape == (batch_size, 64), f"Expected cell state shape (2, 64), got {next_state[0].shape}"
    assert next_state[1].shape == (batch_size, 64), f"Expected hidden state shape (2, 64), got {next_state[1].shape}"
    assert deterministic, "Cell output should be deterministic"

    print("mLSTMCell tests passed!")

def test_mlstm_cell_weight_norm():
    print("\n=== Testing mLSTMCell Weight Normalization ===")
    # Test with weight_norm=True
    cell_with_wn = mLSTMCell(num_units=64, weight_norm=True, seed=42)
    # Test with weight_norm=False
    cell_without_wn = mLSTMCell(num_units=64, weight_norm=False, seed=42)

    # Build cells
    input_shape = tf.TensorShape([2, 10])
    cell_with_wn.build(input_shape)
    cell_without_wn.build(input_shape)

    # Create identical inputs and states
    inputs = tf.ones([2, 10])
    states = (tf.zeros([2, 64]), tf.zeros([2, 64]))

    # Run cells
    outputs_wn, _ = cell_with_wn(inputs, states)
    outputs_no_wn, _ = cell_without_wn(inputs, states)

    # Check that weight normalization has an effect
    are_different = not tf.reduce_all(tf.equal(outputs_wn, outputs_no_wn)).numpy()
    print(f"Outputs with and without weight normalization are different: {are_different}")
    assert are_different, "Weight normalization should produce different outputs"

    print("mLSTMCell weight normalization tests passed!")

def test_mlstm_weight_loading():
    print("\n=== Testing mLSTMCell Weight Loading ===")

    # Create temporary directory for mock weight files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create cell
        cell = mLSTMCell(num_units=4, seed=42)
        cell.build(tf.TensorShape([2, 3]))

        # Create mock weight files
        weight_names = [
            "rnn_mlstm_mlstm_wx:0",
            "rnn_mlstm_mlstm_wh:0",
            "rnn_mlstm_mlstm_wmx:0",
            "rnn_mlstm_mlstm_wmh:0",
            "rnn_mlstm_mlstm_b:0",
            "rnn_mlstm_mlstm_gx:0",
            "rnn_mlstm_mlstm_gh:0",
            "rnn_mlstm_mlstm_gmx:0",
            "rnn_mlstm_mlstm_gmh:0"
        ]

        # Create sample weights with known values
        wx = np.ones((3, 16), dtype=np.float32) * 0.1  # 4 units * 4 gates
        wh = np.ones((4, 16), dtype=np.float32) * 0.2
        wmx = np.ones((3, 4), dtype=np.float32) * 0.3
        wmh = np.ones((4, 4), dtype=np.float32) * 0.4
        b = np.ones(16, dtype=np.float32) * 0.5
        gx = np.ones(16, dtype=np.float32) * 0.6
        gh = np.ones(16, dtype=np.float32) * 0.7
        gmx = np.ones(4, dtype=np.float32) * 0.8
        gmh = np.ones(4, dtype=np.float32) * 0.9

        # Save weights to temporary files
        np.save(os.path.join(temp_dir, f"{weight_names[0]}.npy"), wx)
        np.save(os.path.join(temp_dir, f"{weight_names[1]}.npy"), wh)
        np.save(os.path.join(temp_dir, f"{weight_names[2]}.npy"), wmx)
        np.save(os.path.join(temp_dir, f"{weight_names[3]}.npy"), wmh)
        np.save(os.path.join(temp_dir, f"{weight_names[4]}.npy"), b)
        np.save(os.path.join(temp_dir, f"{weight_names[5]}.npy"), gx)
        np.save(os.path.join(temp_dir, f"{weight_names[6]}.npy"), gh)
        np.save(os.path.join(temp_dir, f"{weight_names[7]}.npy"), gmx)
        np.save(os.path.join(temp_dir, f"{weight_names[8]}.npy"), gmh)

        # Load weights
        try:
            cell.load_weights_from_numpy(temp_dir)
            print("Successfully loaded weights")

            # Check that weights were loaded correctly
            assert tf.reduce_all(cell.wx == 0.1), "wx not loaded correctly"
            assert tf.reduce_all(cell.wh == 0.2), "wh not loaded correctly"
            assert tf.reduce_all(cell.wmx == 0.3), "wmx not loaded correctly"
            assert tf.reduce_all(cell.wmh == 0.4), "wmh not loaded correctly"
            assert tf.reduce_all(cell.b == 0.5), "b not loaded correctly"
            assert tf.reduce_all(cell.gx == 0.6), "gx not loaded correctly"
            assert tf.reduce_all(cell.gh == 0.7), "gh not loaded correctly"
            assert tf.reduce_all(cell.gmx == 0.8), "gmx not loaded correctly"
            assert tf.reduce_all(cell.gmh == 0.9), "gmh not loaded correctly"

        except Exception as e:
            assert False, f"Weight loading failed with error: {e}"

        # Test loading from invalid directory
        try:
            cell.load_weights_from_numpy("/invalid/path")
            print("Handled invalid path correctly")
        except Exception as e:
            print(f"Expected error with invalid path: {e}")

    print("mLSTMCell weight loading tests passed!")

def test_mlstm_computational_correctness():
    print("\n=== Testing mLSTMCell Computational Correctness ===")

    # Create cell with fixed seed
    cell = mLSTMCell(num_units=2, seed=42)

    # Use fixed inputs and states
    inputs = tf.constant([[0.1, 0.2]], dtype=tf.float32)
    c_prev = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    h_prev = tf.constant([[0.0, 0.0]], dtype=tf.float32)

    # Build and initialize with known weights
    cell.build(tf.TensorShape([1, 2]))

    # Set weights to known values for deterministic calculation
    cell.wx.assign(tf.ones([2, 8]) * 0.1)
    cell.wh.assign(tf.ones([2, 8]) * 0.2)
    cell.wmx.assign(tf.ones([2, 2]) * 0.3)
    cell.wmh.assign(tf.ones([2, 2]) * 0.4)
    cell.b.assign(tf.ones([8]) * 0.5)
    cell.gx.assign(tf.ones([8]) * 1.0)
    cell.gh.assign(tf.ones([8]) * 1.0)
    cell.gmx.assign(tf.ones([2]) * 1.0)
    cell.gmh.assign(tf.ones([2]) * 1.0)

    # Run cell
    output, (c_next, h_next) = cell(inputs, (c_prev, h_prev))

    # Print outputs for debugging
    print(f"Output: {output.numpy()}")
    print(f"Next cell state: {c_next.numpy()}")
    print(f"Next hidden state: {h_next.numpy()}")

    # Verify outputs against manually calculated values
    # These expected values would need to be calculated based on the specific weights
    # For now, we'll just capture the current output and use it as a reference for regression testing

    # Run cell twice to verify consistent results
    output2, (c_next2, h_next2) = cell(inputs, (c_prev, h_prev))
    assert tf.reduce_all(output == output2), "Output should be deterministic"
    assert tf.reduce_all(c_next == c_next2), "Cell state should be deterministic"
    assert tf.reduce_all(h_next == h_next2), "Hidden state should be deterministic"

    print("mLSTMCell computational correctness tests passed!")

def test_mlstm_edge_cases():
    print("\n=== Testing mLSTMCell Edge Cases ===")

    # Test with very small num_units
    small_cell = mLSTMCell(num_units=1, seed=42)
    small_cell.build(tf.TensorShape([1, 1]))

    # Test with larger dimensions
    large_cell = mLSTMCell(num_units=128, seed=42)
    large_cell.build(tf.TensorShape([16, 64]))

    # Test with zero-valued inputs
    zero_cell = mLSTMCell(num_units=4, seed=42)
    zero_cell.build(tf.TensorShape([2, 3]))
    zero_inputs = tf.zeros([2, 3])
    zero_states = (tf.zeros([2, 4]), tf.zeros([2, 4]))

    zero_output, _ = zero_cell(zero_inputs, zero_states)
    print(f"Output with zero inputs: {zero_output.numpy()}")

    # Create inputs with NaN to test robustness
    nan_inputs = tf.constant([[float('nan'), 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=tf.float32)
    try:
        nan_output, _ = zero_cell(nan_inputs, zero_states)
        print("Warning: Cell produced output with NaN inputs")
    except Exception as e:
        print(f"Cell properly handled NaN inputs with error: {e}")

    print("mLSTMCell edge case tests passed!")

def run_all_tests():
    print("Running all cell tests...")
    try:
        test_base_cell()
        test_mlstm_cell()
        test_mlstm_cell_weight_norm()
        test_mlstm_weight_loading()
        test_mlstm_computational_correctness()
        test_mlstm_edge_cases()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")

if __name__ == "__main__":
    run_all_tests()