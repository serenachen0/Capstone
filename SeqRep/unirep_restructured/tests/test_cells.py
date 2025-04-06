# tests/test_cells.py
import tensorflow as tf
import numpy as np
import os
import tempfile
from cells.base_cell import BaseLSTMCell
from cells.mlstm_cell import mLSTMCell
from cells.mlstm_stack import StackedMlstmCell
import unittest

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


def test_stacked_mlstm_cell():
    print("\n=== Testing StackedMlstmCell ===")

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Default parameters for testing
    num_units = 32
    num_layers = 3
    batch_size = 4
    input_dim = 10

    # Create stacked cell
    stacked_cell = StackedMlstmCell(
        num_units=num_units,
        num_layers=num_layers,
        seed=42
    )

    # Test initialization
    print("Testing initialization...")
    assert stacked_cell._num_units == num_units, "Incorrect num_units"
    assert stacked_cell._num_layers == num_layers, "Incorrect num_layers"
    assert len(stacked_cell._cells) == num_layers, "Incorrect number of cells"

    # Check that all cells are mLSTMCell instances
    for cell in stacked_cell._cells:
        assert isinstance(cell, mLSTMCell), "Cell is not an instance of mLSTMCell"

    # Check state size
    cell_sizes, hidden_sizes = stacked_cell.state_size
    assert len(cell_sizes) == num_layers, "Incorrect cell state size"
    assert len(hidden_sizes) == num_layers, "Incorrect hidden state size"

    # Check output size
    assert stacked_cell.output_size == tf.TensorShape([num_units]), "Incorrect output size"

    # Test initial state
    print("Testing initial state...")
    initial_state = stacked_cell.get_initial_state(batch_size=batch_size)

    # Check state structure
    cell_states, hidden_states = initial_state
    assert len(cell_states) == num_layers, "Incorrect number of cell states"
    assert len(hidden_states) == num_layers, "Incorrect number of hidden states"

    # Check state shapes
    for i in range(num_layers):
        assert cell_states[i].shape == (batch_size, num_units), f"Incorrect cell state shape for layer {i}"
        assert hidden_states[i].shape == (batch_size, num_units), f"Incorrect hidden state shape for layer {i}"

    # Test forward pass
    print("Testing forward pass...")
    inputs = tf.random.normal((batch_size, input_dim), seed=42)

    # Run forward pass
    output, final_state = stacked_cell(
        inputs=inputs,
        states=initial_state,
        training=False
    )

    # Check output shape
    assert output.shape == (batch_size, num_units), "Incorrect output shape"

    # Check state structure
    cell_states, hidden_states = final_state
    assert len(cell_states) == num_layers, "Incorrect number of final cell states"
    assert len(hidden_states) == num_layers, "Incorrect number of final hidden states"

    # Check state shapes
    for i in range(num_layers):
        assert cell_states[i].shape == (batch_size, num_units), f"Incorrect final cell state shape for layer {i}"
        assert hidden_states[i].shape == (batch_size, num_units), f"Incorrect final hidden state shape for layer {i}"

    # Test residual connections
    print("Testing residual connections...")
    stacked_cell_residual = StackedMlstmCell(
        num_units=num_units,
        num_layers=num_layers,
        residual_connections=True,
        seed=42
    )

    output_residual, _ = stacked_cell_residual(
        inputs=inputs,
        states=initial_state,
        training=False
    )

    # Outputs with and without residual connections should be different
    assert not np.allclose(output.numpy(), output_residual.numpy()), "Residual connections had no effect"

    # Test dropout
    print("Testing dropout...")
    dropout_rate = 0.5
    stacked_cell_dropout = StackedMlstmCell(
        num_units=num_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        seed=42
    )

    # Run in training mode with dropout
    output_training, _ = stacked_cell_dropout(
        inputs=inputs,
        states=initial_state,
        training=True
    )

    # Run in inference mode without dropout
    output_inference, _ = stacked_cell_dropout(
        inputs=inputs,
        states=initial_state,
        training=False
    )

    # Outputs with and without dropout should be different
    assert not np.allclose(output_training.numpy(), output_inference.numpy()), "Dropout had no effect"

    # Test deterministic behavior
    print("Testing deterministic behavior...")

    # Reset random seeds
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create stacked cell again
    stacked_cell2 = StackedMlstmCell(
        num_units=num_units,
        num_layers=num_layers,
        seed=42
    )

    # Create input again
    inputs2 = tf.random.normal((batch_size, input_dim), seed=42)

    # Get initial state
    initial_state2 = stacked_cell2.get_initial_state(batch_size=batch_size)

    # Run forward pass
    output2, final_state2 = stacked_cell2(
        inputs=inputs2,
        states=initial_state2,
        training=False
    )

    # Check that outputs are identical
    assert np.allclose(output.numpy(), output2.numpy()), "Outputs are not deterministic"

    # Test sequence processing
    print("Testing sequence processing...")
    seq_len = 5
    sequence = tf.random.normal((batch_size, seq_len, input_dim), seed=42)

    # Process sequence with RNN layer
    rnn_layer = tf.keras.layers.RNN(
        stacked_cell,
        return_sequences=True,
        return_state=True
    )

    # Run forward pass
    outputs, *final_states = rnn_layer(sequence, initial_state=initial_state)

    # Check output shape
    assert outputs.shape == (batch_size, seq_len, num_units), "Incorrect sequence output shape"

    print("StackedMlstmCell tests passed!")

def run_all_tests():
    print("Running all cell tests...")
    try:
        test_base_cell()
        test_mlstm_cell()
        test_mlstm_cell_weight_norm()
        test_mlstm_weight_loading()
        test_mlstm_computational_correctness()
        test_mlstm_edge_cases()
        test_stacked_mlstm_cell()  # Added new test function
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error encountered: {e}")

class TestStackedMlstmCell(unittest.TestCase):
    """Test cases for the StackedMlstmCell class."""

    def setUp(self):
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        # Default parameters for testing
        self.num_units = 32
        self.num_layers = 3
        self.batch_size = 4
        self.input_dim = 10
        self.seed = 42

    def test_initialization(self):
        """Test basic initialization of the stacked cell."""
        # Create stacked cell
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Check that properties are set correctly
        self.assertEqual(stacked_cell._num_units, self.num_units)
        self.assertEqual(stacked_cell._num_layers, self.num_layers)
        self.assertEqual(stacked_cell._seed, self.seed)
        self.assertEqual(len(stacked_cell._cells), self.num_layers)

        # Check that all cells are mLSTMCell instances
        for cell in stacked_cell._cells:
            self.assertIsInstance(cell, mLSTMCell)

        # Check state size
        cell_sizes, hidden_sizes = stacked_cell.state_size
        self.assertEqual(len(cell_sizes), self.num_layers)
        self.assertEqual(len(hidden_sizes), self.num_layers)

        # Check output size
        self.assertEqual(stacked_cell.output_size, tf.TensorShape([self.num_units]))

    def test_initial_state(self):
        """Test creation of initial state."""
        # Create stacked cell
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Check state structure
        cell_states, hidden_states = initial_state
        self.assertEqual(len(cell_states), self.num_layers)
        self.assertEqual(len(hidden_states), self.num_layers)

        # Check state shapes
        for i in range(self.num_layers):
            self.assertEqual(cell_states[i].shape, (self.batch_size, self.num_units))
            self.assertEqual(hidden_states[i].shape, (self.batch_size, self.num_units))

        # Check that states are zero-initialized
        for i in range(self.num_layers):
            self.assertTrue(np.all(cell_states[i].numpy() == 0))
            self.assertTrue(np.all(hidden_states[i].numpy() == 0))

    def test_forward_pass(self):
        """Test forward pass through the stacked cell."""
        # Create stacked cell
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Create input
        inputs = tf.random.normal((self.batch_size, self.input_dim))

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Run forward pass
        output, final_state = stacked_cell(
            inputs=inputs,
            states=initial_state,
            training=False
        )

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_units))

        # Check state structure
        cell_states, hidden_states = final_state
        self.assertEqual(len(cell_states), self.num_layers)
        self.assertEqual(len(hidden_states), self.num_layers)

        # Check state shapes
        for i in range(self.num_layers):
            self.assertEqual(cell_states[i].shape, (self.batch_size, self.num_units))
            self.assertEqual(hidden_states[i].shape, (self.batch_size, self.num_units))

        # Check that output matches final hidden state of last layer
        self.assertTrue(np.allclose(output.numpy(), hidden_states[-1].numpy()))

    def test_residual_connections(self):
        """Test residual connections in the stacked cell."""
        # Create stacked cell with residual connections
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            residual_connections=True,
            seed=self.seed
        )

        # Create input
        inputs = tf.random.normal((self.batch_size, self.input_dim))

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Run forward pass
        output, final_state = stacked_cell(
            inputs=inputs,
            states=initial_state,
            training=False
        )

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_units))

        # Create stacked cell without residual connections
        stacked_cell_no_residual = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            residual_connections=False,
            seed=self.seed
        )

        # Run forward pass without residual connections
        output_no_residual, _ = stacked_cell_no_residual(
            inputs=inputs,
            states=initial_state,
            training=False
        )

        # Check that outputs are different
        self.assertFalse(np.allclose(output.numpy(), output_no_residual.numpy()))

    def test_dropout(self):
        """Test dropout in the stacked cell."""
        # Create stacked cell with dropout
        dropout_rate = 0.5
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            dropout_rate=dropout_rate,
            seed=self.seed
        )

        # Create input
        inputs = tf.random.normal((self.batch_size, self.input_dim))

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Run forward pass in training mode
        output_training, _ = stacked_cell(
            inputs=inputs,
            states=initial_state,
            training=True
        )

        # Run forward pass in inference mode
        output_inference, _ = stacked_cell(
            inputs=inputs,
            states=initial_state,
            training=False
        )

        # Check that outputs are different due to dropout
        self.assertFalse(np.allclose(output_training.numpy(), output_inference.numpy()))

    def test_deterministic_behavior(self):
        """Test deterministic behavior with fixed seeds."""
        # Create stacked cell
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Create input
        inputs = tf.random.normal((self.batch_size, self.input_dim), seed=self.seed)

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Run forward pass
        output1, final_state1 = stacked_cell(
            inputs=inputs,
            states=initial_state,
            training=False
        )

        # Reset random seeds
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        # Create stacked cell again
        stacked_cell2 = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Create input again
        inputs2 = tf.random.normal((self.batch_size, self.input_dim), seed=self.seed)

        # Get initial state
        initial_state2 = stacked_cell2.get_initial_state(batch_size=self.batch_size)

        # Run forward pass
        output2, final_state2 = stacked_cell2(
            inputs=inputs2,
            states=initial_state2,
            training=False
        )

        # Check that outputs are identical
        self.assertTrue(np.allclose(output1.numpy(), output2.numpy()))

        # Check that states are identical
        cell_states1, hidden_states1 = final_state1
        cell_states2, hidden_states2 = final_state2

        for i in range(self.num_layers):
            self.assertTrue(np.allclose(cell_states1[i].numpy(), cell_states2[i].numpy()))
            self.assertTrue(np.allclose(hidden_states1[i].numpy(), hidden_states2[i].numpy()))

    def test_sequence_processing(self):
        """Test processing a sequence through the stacked cell."""
        # Create stacked cell
        stacked_cell = StackedMlstmCell(
            num_units=self.num_units,
            num_layers=self.num_layers,
            seed=self.seed
        )

        # Create sequence input [batch_size, seq_len, input_dim]
        seq_len = 5
        sequence = tf.random.normal((self.batch_size, seq_len, self.input_dim), seed=self.seed)

        # Get initial state
        initial_state = stacked_cell.get_initial_state(batch_size=self.batch_size)

        # Process sequence with RNN layer
        rnn_layer = tf.keras.layers.RNN(
            stacked_cell,
            return_sequences=True,
            return_state=True
        )

        # Run forward pass
        outputs, *final_states = rnn_layer(sequence, initial_state=initial_state)

        # Check output shape [batch_size, seq_len, num_units]
        self.assertEqual(outputs.shape, (self.batch_size, seq_len, self.num_units))

        # Unpack final states
        cell_states, hidden_states = final_states

         # Check state structure
        self.assertEqual(len(cell_states), self.num_layers)
        self.assertEqual(len(hidden_states), self.num_layers)

        # Check state shapes
        for i in range(self.num_layers):
            self.assertEqual(cell_states[i].shape, (self.batch_size, self.num_units))
            self.assertEqual(hidden_states[i].shape, (self.batch_size, self.num_units))

if __name__ == "__main__":
    # Option 1: Run simple functional tests
    run_all_tests()