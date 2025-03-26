"""
Sequence processing utilities for amino acid sequences
"""

from typing import Dict, List, Tuple, Union, Optional, Any, Set
import tensorflow as tf
import numpy as np
import os
from Bio import SeqIO

# Type-annotated dictionaries
aa_to_int: Dict[str, int] = {
    'M': 1, 'R': 2, 'H': 3, 'K': 4, 'D': 5, 'E': 6, 'S': 7, 'T': 8, 'N': 9, 'Q': 10, 'C': 11, 
    'U': 12, 'G': 13, 'P': 14, 'A': 15, 'V': 16, 'I': 17, 'F': 18, 'Y': 19, 'W': 20, 'L': 21, 
    'O': 22, # Pyrrolysine
    'X': 23, # Unkown
    'Z': 23, # Glutamic acid or GLutamine
    'B': 23, # Asparagine or aspartic acid
    'J': 23, # Leucine or isoleucine
    'start': 24, 'stop': 25,
}

int_to_aa: Dict[int, str] = {value: key for key, value in aa_to_int.items()}

def aa_seq_to_int(s: str, include_stop: bool = True) -> List[int]:
    """
    Convert an amino acid sequence to integer representation.
    
    Args:
        s: String of amino acid characters
        include_stop: Whether to include the stop token (default True)
        
    Returns:
        List of integers representing the sequence with start/stop tokens
    """
    if include_stop:
        return [24] + [aa_to_int.get(a, 23) for a in s] + [25]
    else:
        return [24] + [aa_to_int.get(a, 23) for a in s]

def int_to_aa_seq(int_seq: List[int], remove_tokens: bool = True) -> str:
    """
    Convert an integer representation back to an amino acid sequence.
    
    Args:
        int_seq: List of integers representing the sequence
        remove_tokens: Whether to remove start/stop tokens
        
    Returns:
        String of amino acid characters
    """
    if remove_tokens:
        # Remove start and stop tokens (usually at positions 0 and -1)
        if int_seq[0] == 24:  # start token
            int_seq = int_seq[1:]
        if int_seq and int_seq[-1] == 25:  # stop token
            int_seq = int_seq[:-1]
    
    return ''.join([int_to_aa.get(i, 'X') for i in int_seq])

def is_valid_seq(seq: str, max_len: int = 2000) -> bool:
    """
    Check if a sequence is valid for UniRep processing.
    
    Args:
        seq: String of amino acid characters
        max_len: Maximum allowed sequence length
        
    Returns:
        Boolean indicating if sequence is valid
    """
    valid_aas: Set[str] = set("MRHKDESTNQCUGPAVIFYWLO")
    return len(seq) < max_len and set(seq).issubset(valid_aas)

def seq_to_tensor(seq: str, remove_stop: bool = True) -> tf.Tensor:
    """
    Convert a sequence to a TensorFlow tensor with batch dimension.
    
    Args:
        seq: Amino acid sequence string
        remove_stop: Whether to remove the stop token
        
    Returns:
        TensorFlow tensor of sequence integers
    """
    # Make sure we're working with consistent ordering of amino acids
    int_seq = aa_seq_to_int(seq.strip())
    
    # Remove stop token if requested
    if remove_stop:
        int_seq = int_seq[:-1]
        
    return tf.convert_to_tensor([int_seq], dtype=tf.int32)

# ------------------------
# Batch processing class
# ------------------------

class SequenceBatcher:
    """
    Efficient batch processor for protein sequences.
    """
    
    def __init__(
        self, 
        batch_size: int = 256, 
        max_seq_len: int = 2000, 
        include_stop: bool = False, 
        shuffle: bool = True, 
        seed: int = 42
    ):
        """
        Initialize the sequence batcher.
        
        Args:
            batch_size: Number of sequences per batch
            max_seq_len: Maximum sequence length to process
            include_stop: Whether to include stop tokens
            shuffle: Whether to shuffle the data
            seed: Random seed for reproducible shuffling
        """
        self.batch_size: int = batch_size
        self.max_seq_len: int = max_seq_len
        self.include_stop: bool = include_stop
        self.shuffle: bool = shuffle
        self.seed: int = seed
    
    def batch_pad_sequences(
        self, sequences: List[str]
    ) -> Tuple[tf.Tensor, List[int]]:
        """
        Create efficiently batched sequences with padding.
        
        Args:
            sequences: List of amino acid sequences
            
        Returns:
            Padded tensor of shape [batch_size, max_length]
            List of sequence lengths
        """
        # Convert sequences to integer lists
        int_seqs: List[List[int]] = [
            aa_seq_to_int(seq.strip(), include_stop=self.include_stop) 
            for seq in sequences
        ]
        
        # Get sequence lengths
        seq_lengths: List[int] = [len(seq) for seq in int_seqs]
        
        # Determine padding length (dynamic based on batch)
        pad_len: int = min(max(seq_lengths), self.max_seq_len)
        
        # Pad sequences
        padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(
            int_seqs, 
            maxlen=pad_len,
            padding='post',
            truncating='post',
            value=0
        )
        
        return tf.convert_to_tensor(padded_seqs, dtype=tf.int32), seq_lengths
    
    def create_dataset(
        self, 
        sequences: Union[List[str], Dict[str, str]], 
        repeat: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from sequences for efficient processing.
        
        Args:
            sequences: List of amino acid sequences or dict with seq_id:seq pairs
            repeat: Number of epochs to repeat (None for infinite)
            
        Returns:
            TensorFlow Dataset object
        """
        # Handle dictionary input
        if isinstance(sequences, dict):
            seq_ids: List[str] = list(sequences.keys())
            seq_list: List[str] = list(sequences.values())
        else:
            seq_ids: List[int] = list(range(len(sequences)))
            seq_list: List[str] = sequences
        
        # Convert sequences to integer lists
        int_seqs: List[List[int]] = [
            aa_seq_to_int(seq.strip(), include_stop=self.include_stop) 
            for seq in seq_list
        ]
        
        # Get sequence lengths
        seq_lengths: List[int] = [len(seq) for seq in int_seqs]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            seq_ids,
            int_seqs, 
            seq_lengths
        ))
        
        # Shuffle if requested
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(len(int_seqs), 10000),
                seed=self.seed,
                reshuffle_each_iteration=True
            )
        
        # Repeat if requested
        if repeat is not None:
            dataset = dataset.repeat(repeat)
        
        # Batch and pad
        padded_shapes = (
            tf.TensorShape([]),        # seq_id shape
            tf.TensorShape([None]),    # sequence shape (variable length)
            tf.TensorShape([])         # length shape
        )
        
        padding_values = (
            "",   # seq_id padding (not used for string ids)
            0,    # sequence padding 
            0     # length padding (not used)
        )
        
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values
        )
        
        return dataset
    
    def process_fasta(self, fasta_path: str) -> Dict[str, str]:
        """
        Process a FASTA file into a dictionary of sequences.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            Dict of sequence_id:sequence pairs
        """
        sequences: Dict[str, str] = {}

        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id: str = record.id
            sequence: str = str(record.seq)
                
            # Check if sequence is valid
            if is_valid_seq(sequence, self.max_seq_len):
                sequences[seq_id] = sequence
                
        return sequences
    
    def save_to_h5(
        self, 
        sequences: Dict[str, str], 
        output_file: str, 
        model_fn: callable,
        representation_key: str = 'avg_hidden'
    ) -> str:
        """
        Process sequences through a model and save the representations to an HDF5 file.
        
        Args:
            sequences: Dictionary of sequence_id:sequence pairs
            output_file: Path to output HDF5 file
            model_fn: Function that takes a sequence and returns representations
            representation_key: Key to extract from model output dictionary
            
        Returns:
            Path to the created HDF5 file
        """
        import h5py
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Process and save representations
        with h5py.File(output_file, 'w') as f:
            for seq_id, seq in sequences.items():
                # Get representation from model
                rep = model_fn(seq)
                
                # Extract desired representation
                if isinstance(rep, dict):
                    data = rep.get(representation_key)
                elif isinstance(rep, tuple):
                    # Assuming tuple of (avg_hidden, final_hidden, final_cell)
                    data = rep[0]
                else:
                    data = rep
                
                # Convert to numpy if it's a tensor
                if isinstance(data, tf.Tensor):
                    data = data.numpy()
                
                # Save to HDF5
                f.create_dataset(seq_id, data=data)
        
        return output_file