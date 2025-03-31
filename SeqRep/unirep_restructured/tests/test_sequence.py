import tensorflow as tf
import numpy as np

# Import module
from utils.sequence import (
    aa_to_int, int_to_aa, aa_seq_to_int, int_to_aa_seq,
    is_valid_seq, seq_to_tensor, SequenceBatcher
)

def test_aa_conversion():
    """Test amino acid sequence conversion functions"""
    # Test sequence to int conversion
    seq = "MASKGEELFT"
    int_seq = aa_seq_to_int(seq)
    
    # Check if start token is added
    print("Start token - Expected:", 24)
    print("Start token - Actual:", int_seq[0])
    assert int_seq[0] == 24, "Start token not added correctly"
    
    # Check if sequence is converted correctly
    print("Sequence conversion - Expected:", [1, 15, 7, 4, 13, 6, 6, 21, 18, 8])
    print("Sequence conversion - Actual:", int_seq[1:-1])
    assert int_seq[1:-1] == [1, 15, 7, 4, 13, 6, 6, 21, 18, 8], "Sequence conversion incorrect"
    
    # Check if stop token is added
    print("Stop token - Expected:", 25)
    print("Stop token - Actual:", int_seq[-1])
    assert int_seq[-1] == 25, "Stop token not added correctly"
    
    # Test conversion without stop token
    int_seq_no_stop = aa_seq_to_int(seq, include_stop=False)
    print("No stop token - Expected: 25 not in sequence")
    print("No stop token - Actual:", 25 in int_seq_no_stop)
    assert 25 not in int_seq_no_stop, "Stop token should not be present"
    
    # Test int to sequence conversion
    decoded_seq = int_to_aa_seq(int_seq)
    print("Round-trip conversion - Expected:", seq)
    print("Round-trip conversion - Actual:", decoded_seq)
    assert decoded_seq == seq, f"Round-trip conversion failed: {decoded_seq} != {seq}"
    
    # Test with token removal
    decoded_with_tokens = int_to_aa_seq(int_seq, remove_tokens=False)
    print("Token retention - Expected:", "start" + seq + "stop")
    print("Token retention - Actual:", decoded_with_tokens)
    assert decoded_with_tokens == "start" + seq + "stop", "Token retention failed"

def test_sequence_validation():
    """Test sequence validation function"""
    # Valid sequence
    valid_result = is_valid_seq("MASKGEELFT")
    print("Valid sequence check - Expected:", True)
    print("Valid sequence check - Actual:", valid_result)
    assert valid_result == True, "Valid sequence incorrectly marked invalid"
    
    # Invalid sequence (contains B which is not in the valid set)
    invalid_result = is_valid_seq("MASKGEBLFT")
    print("Invalid sequence check - Expected:", False)
    print("Invalid sequence check - Actual:", invalid_result)
    assert invalid_result == False, "Invalid sequence incorrectly marked valid"
    
    # Too long sequence
    long_seq = "A" * 2001
    too_long_result = is_valid_seq(long_seq)
    print("Too long sequence check - Expected:", False)
    print("Too long sequence check - Actual:", too_long_result)
    assert too_long_result == False, "Too long sequence incorrectly marked valid"

def test_sequence_tensor_conversion():
    """Test converting sequences to tensors"""
    seq = "MASKGEELFT"
    tensor = seq_to_tensor(seq)
    
    # Check shape and type
    is_tensor = isinstance(tensor, tf.Tensor)
    print("Is tensor - Expected:", True)
    print("Is tensor - Actual:", is_tensor)
    assert is_tensor, "Result is not a TensorFlow tensor"
    
    tensor_shape = tensor.shape.as_list()
    print("Tensor shape - Expected:", [1, 11])
    print("Tensor shape - Actual:", tensor_shape)
    assert tensor_shape == [1, 11], f"Unexpected tensor shape: {tensor_shape}"
    
    # Check tensor content (should have batch dimension)
    first_token = tensor.numpy()[0][0]
    print("First token - Expected:", 24)
    print("First token - Actual:", first_token)
    assert first_token == 24, "Start token not in tensor"
    
    middle_content = tensor.numpy()[0][1:].tolist()
    print("Middle content - Expected:", [1, 15, 7, 4, 13, 6, 6, 21, 18, 8])
    print("Middle content - Actual:", middle_content)
    assert middle_content == [1, 15, 7, 4, 13, 6, 6, 21, 18, 8], "Sequence conversion in tensor incorrect"
    
    # Test with remove_stop=False
    tensor_with_stop = seq_to_tensor(seq, remove_stop=False)
    last_token = tensor_with_stop.numpy()[0][-1]
    print("Last token (with stop) - Expected:", 25)
    print("Last token (with stop) - Actual:", last_token)
    assert last_token == 25, "Stop token not in tensor"

def test_sequence_batcher():
    """Test the SequenceBatcher class"""
    # Create sample sequences
    sequences = [
        "MASKGEELFT",
        "MVSKGEEDNM",
        "MSKGPVLSRDK"
    ]
    
    # Initialize batcher
    batcher = SequenceBatcher(batch_size=2, max_seq_len=20, seed=42, include_stop=True)
    
    # Test batch_pad_sequences
    padded_tensor, seq_lengths = batcher.batch_pad_sequences(sequences[:2])
    
    # Check shapes and values
    padded_shape = padded_tensor.shape
    print("Padded shape - Expected:", (2, 12))
    print("Padded shape - Actual:", padded_shape)
    assert padded_shape == (2, 12), f"Unexpected padded shape: {padded_shape}"
    
    print("Sequence lengths - Expected:", [12, 12])
    print("Sequence lengths - Actual:", seq_lengths)
    assert seq_lengths == [12, 12], f"Unexpected sequence lengths: {seq_lengths}"
    
    # Test create_dataset
    dataset = batcher.create_dataset(sequences, repeat=1)
    
    # Check if dataset is created correctly
    is_dataset = isinstance(dataset, tf.data.Dataset)
    print("Is dataset - Expected:", True)
    print("Is dataset - Actual:", is_dataset)
    assert is_dataset, "Result is not a TensorFlow dataset"
    
    # Get a batch from the dataset
    for batch in dataset:
        seq_ids, seqs, lengths = batch
        batch_size_check = len(seq_ids) <= 2
        print("Batch size check - Expected:", True)
        print("Batch size check - Actual:", batch_size_check)
        assert batch_size_check, f"Batch size exceeded: {len(seq_ids)}"
        
        seq_length_check = seqs.shape[1] >= 11
        print("Sequence padding check - Expected:", True)
        print("Sequence padding check - Actual:", seq_length_check)
        assert seq_length_check, f"Sequence padding insufficient: {seqs.shape}"
        break

def run_all_tests():
    """Run all tests"""
    print("\n=== RUNNING aa_conversion TESTS ===")
    test_aa_conversion()
    print("\n=== RUNNING sequence_validation TESTS ===")
    test_sequence_validation()
    print("\n=== RUNNING sequence_tensor_conversion TESTS ===")
    test_sequence_tensor_conversion()
    print("\n=== RUNNING sequence_batcher TESTS ===")
    test_sequence_batcher()
    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    run_all_tests()