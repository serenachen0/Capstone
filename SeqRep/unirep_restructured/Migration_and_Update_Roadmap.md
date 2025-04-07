# UniRep2 Development Journal

Objective: refactor the UniRep model to ensure deterministic output and improve performance

<a id="toc"></a>
## 📑 Table of Contents

<summary>Click to expand</summary>

### [📋 Entry 0: Project Analysis and Planning](#entry-0)

- [🔍 Problem Analysis](#problem-analysis)
- [🧩 Root Causes](#root-causes)
    - [📦 Original Library Dependencies](#original-library-dependencies)
    - [🚧 Key Migration Challenges](#migration-challenges)
- [📁 Preliminary Repository Structure](#preliminary-repository-structure)

### [🛠️ Entry 1: Utility Modules Implementation](#entry-1)
- [🎯 Goals](#util-goals)
- [⚙️ Implementation Details](#util-impl-0)
  - [1️⃣ Created a structured `utils/` directory](#util-impl-1)
  - [2️⃣ Extracted the amino acid encoding functionality](#util-impl-2)
  - [3️⃣ Implement three additional utility methods](#util-impl-3)
  - [4️⃣ Refactored `bucketbatchpad()` to `SequenceBatcher()` class](#util-impl-4)
  - [5️⃣ Extract `tf_get_shape()` to `utils/tensor_utils.py`](#util-impl-5)
  - [6️⃣ Refactor `sample_with_temp()` to `utils/tensor_utils.py`](#util-impl-6)
- [🔄 TensorFlow API Changes](#util-tf-changes)
- [📝 Explanation with Examples](#util-explain-0)
    - [Step 1: Validate the sequence](#util-explain-1)
    - [Step 2: Convert amino acids to integers](#util-explain-2)
    - [Step 3: Add start and stop tokens](#util-explain-3)
    - [Step 4: Record sequence lengths](#util-explain-4)
    - [Step 5: Pad sequences to the same length](#util-explain-5)
    - [Step 6: Create TensorFlow dataset](#util-explain-6)
    - [Step 7: Optinally Shuffle the dataset](#util-explain-7)
    - [Step 8: Batch the dataset](#util-explain-8)
    - [Step 9: Feed into mLSTM model](#util-explain-9)
- [🧪 Testing](#util-testing)

### [🧠 Entry 2: LSTM Cell Implementation](#entry-2)
- [🎯 Goals](#cell-goals)
- [⚙️ Implementation Details](#cell-impl-0)
  - [1️⃣ Base LSTM Cell Extraction & Refactoring](#cell-impl-1)
  - [2️⃣ New Methods for Determinstic Behavior and API Consistency](#cell-impl-2)
  - [3️⃣ Implement mLSTM Cell](#cell-impl-3)
  - [4️⃣ Stacked mLSTM Cell Implementation](#cell-impl-4)
- [🔄 TensorFlow API Changes](#cell-tf-changes)
- [🧪 Testing](#cell-testing)
### [🧩 Entry 3: Refactor Model Architecture](#entry-3)
- [🎯 Goals](#model-goals)
- [⚙️ Implementation Details](#model-impl-0)
    - [1️⃣ Base Configuration Implementation](#model-impl-1)
    - [2️⃣ Base Model Implementation](#model-impl-2)
- [🔄 Tensorflow API Changes](#model-tf-changes)
- [🧪 Testing](#model-testing)

<a id="entry-0"></a>
## 📋 Entry 0: Project Analysis and Planning

<summary>Click to expand</summary>

<a id="problem-analysis"></a>
### 🔍 Problem Analysis

Three major issues are identified with the original UniRep implementation:

1. Non-deterministic Output: Running the same sequence through the model produces different results each time

2. Performance Issues: Processing a single protein sequence takes ~1 second, which is 25x slower than transformer-based alternatives

3. Outded Libraries: library dependencies used in the original implementation were severely outdated, createing challenges to address to the first 2 issues.

<a id="root-causes"></a>
### 🧩 Root Causes

For the non-deterministic behavior, I identified the following potential causes:
- Random initialization of weights/parameters
- Non-deterministic GPU operations (especially reduce_mean)
- Threading or parallelism issues

<a id="original-library-dependencies"></a>
#### 📦 Original Library Dependencies

The original UniRep implementation used severely outdated libraries:

| Library    | Original Version | Current Latest | Status            |
|------------|-----------------|----------------|-------------------|
| TensorFlow | 1.3.0          | 2.18.0        | Major API changes |
| NumPy      | 1.15.4         | 2.1.0+        | API changes       |
| Biopython  | Not specified   | 1.82+         | Minor changes     |
| h5py       | Not specified   | 3.10.0+       | API changes       |

This creates significant challenges as TensorFlow 1.3 was released in 2017 and lacks critical features for performance and reproducibility.

<a id="migration-challenges"></a>
#### 🚧 Key Migration Challenges

- TensorFlow 1.x to 2.x migration (completely different API)
- RNN cell implementation changes
- Session vs eager execution model differences
- Deterministic operation support differences

<a id="preliminary-repository-structure"></a>
### 📁 Preliminary Repository Structure

```bash
unirep/
├── __init__.py           # Package exports
├── config.py             # Configuration classes
├── cells/                # LSTM cell implementations
│   ├── __init__.py
│   ├── base_cell.py      # Base LSTM cell class
│   ├── mlstm_cell.py     # mLSTM cell implementation
│   └── mlstm_stack.py    # Stacked mLSTM implementation
├── models/               # Model implementations
│   ├── __init__.py
│   ├── base_model.py     # Base model class
│   ├── unirep64.py       # 64-unit model
│   ├── unirep256.py      # 256-unit model
│   └── unirep1900.py     # 1900-unit model
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── sequence.py       # Sequence processing utilities
│   ├── representation.py # Representation utilities
│   └── tensor_utils.py   # TensorFlow utilities
├── cli/                  # Command line interface
│   ├── __init__.py
│   ├── seq2rep.py        # Sequence to representation CLI
│   └── rep_compare.py    # Representation comparison CLI
├── requirements.txt      # Modern dependencies
├── setup.py              # Package installation
├── tests/                # Test suite
│   ├── test_config.py    # Configuration tests
│   ├── test_cells.py     # Cell implementation tests
│   ├── test_models.py    # Model implementation tests
│   └── test_determinism.py # Determinism tests
└── examples/             # Usage examples
    ├── basic_usage.py    # Basic model usage
    ├── batch_processing.py # Batch processing example
    └── deterministic_comparison.py # Determinism demonstration
```
[🔝 Back to Table of Contents](#toc)


<a id="entry-1"></a>
## 🛠️ Entry 1: Utility Modules Implementation

<summary>Click to expand</summary>

<a id="util-goals"></a>
### 🎯 Goals

- Create utilities for handling protein sequences
- Implement batch processing for performance
- Ensure all data operations are deterministic
- Separate different utility functions into logical modules

<a id="util-impl-0"></a>
### ⚙️ Implementation Details

<a id="util-impl-1"></a>
#### 1️⃣ Created a structured `utils/` directory:
- `sequence.py` - Sequence processing utilities :white_check_mark:
- `representation.py` - Representation analysis utilities
- `tensor_utils.py` - TensorFlow-specific utilities :white_check_mark:

<a id="util-impl-2"></a>
#### 2️⃣ Extracted the amino acid encoding functionality (**lines 12-50**) from `unirep.py`

Original code snippet in `unirep.py`
```python
# Lines 12 - 26
aa_to_int = {
    'M':1, 'R':2, 'H':3, 'K':4, 'D':5, 'E':6, 'S':7, 'T':8, 'N':9, 'Q':10, 'C':11,
    'U':12, 'G':13,'P':14, 'A':15, 'V':16, 'I':17, 'F':18, 'Y':19, 'W':20, 'L':21,
    'O':22, # Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24, 'stop':25,
}

# Line 44
int_to_aa = {value:key for key, value in aa_to_int.items()}

# Line 46 - 50
def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

```

Refactored and enhanced in `utils/sequence.py`
```python
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

# Enhanced function with type annotations and options
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
```

<a id="util-impl-3"></a>
#### 3️⃣ Implement three additional utility methods

`int_to_aa_seq()` decodes integers back to amino acids
```python
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
```

`is_valid_seq()` corresponds to a validation check that was in `unirep.py` (**lines 665-674**)

Original code snippet in `unirep.py`
```python
# inside babbler1900 class
# Lines 665-674
def is_valid_seq(self, seq, max_len=2000):
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
    if (l < max_len) and set(seq) <= set(valid_aas):
        return True
    else:
        return False
```

Extract to a standalone method `is_valid_seq()` with better typing in `sequence.py`
```python
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
```

Implement coverting a sequence to tensor method `seq_to_tensor()` in `sequence.py`
```python
def seq_to_tensor(seq: str, remove_stop: bool = True) -> tf.Tensor:
    """
    Convert a sequence to a TensorFlow tensor with batch dimension.

    Args:
        seq: Amino acid sequence string
        remove_stop: Whether to remove the stop token

    Returns:
        TensorFlow tensor of sequence integers
    """
    if remove_stop:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    else:
        int_seq = aa_seq_to_int(seq.strip())
    return tf.convert_to_tensor([int_seq], dtype=tf.int32)
```

<a id="util-impl-4"></a>
#### 4️⃣ Refactored `bucketbatchpad()` (lines 52-94) to `SequenceBatcher()` class

`bucketbatchpad()` is being decoupled and transformed to a class for several reasons:
- To maintain state across multiple calls (batch size, maximum sequence length, etc.)
- To group related functionality together (batch processing, FASTA handling, etc.)
- To have better typing
- To allow for more flexible configuration options



Key improvements:
>1. Function Parameters → Class Constructor
```python
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
```
- `batch_size` parameter retained with same default (256)
- `path_to_data` moved to method parameters for greater flexibility
- `compressed` and `bounds` parameters modernized due to TensorFlow API changes
- Added `seed` parameter for deterministic behavior
- Added type annotations for better code safety

>2. Dataset Creation Logic → Method-specific Implementation
```python
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
```
- Replaced `tf.contrib` APIs with modern **TensorFlow 2.x** equivalents
- Uses `tf.keras.preprocessing.sequence.pad_sequences` for padding
- Handles dynamic padding based on actual sequence lengths
- Returns both padded sequences and length information

>3.Dataset Creation with Modern TensorFlow API

```python
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

        # Pre-pad all sequences to the maximum length in this dataset
        max_len = min(max(seq_lengths), self.max_seq_len)
        padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(
            int_seqs,
            maxlen=max_len,
            padding='post',
            truncating='post',
            value=0
        )

        # Create dataset with pre-padded sequences
        dataset = tf.data.Dataset.from_tensor_slices((
            seq_ids,
            padded_seqs,
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
        dataset = dataset.padded_batch(self.batch_size)

        return dataset
```
- Updates to modern **TensorFlow 2.x tf.data** API
- Adds support for both list and dictionary inputs
- Implements deterministic shuffling with seed
- Uses proper padding configurations for different tensor types


>4.File Processing Enhancement
```python
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
```

- Added dedicated FASTA file processing
- Validates sequences during loading

<a id="util-impl-5"></a>
#### 5️⃣ Extract `tf_get_shape()` (line 96-101) to `utils/tensor_utils.py`

Original code snippet in `unirep.py`
```python
def tf_get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims
```

Extracted to `utils/tensor_utils.py`
```python
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
```

<a id="util-impl-6"></a>
#### 6️⃣ Refactor `sample_with_temp()` (lines 103-111) to `utils/tensor_utils.py`


Original code snippet in `unirep.py`
```python
def sample_with_temp(logits, t):
    """
    Takes temperature between 0 and 1 -> zero most conservative, 1 most liberal. Samples.
    """
    t_adjusted = logits / t  # broadcast temperature normalization
    softed = tf.nn.softmax(t_adjusted)

    # Make a categorical distribution from the softmax and sample
    return tf.distributions.Categorical(probs=softed).sample()
```

Extracted to `utils/tensor_util.py`
```python
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
```

<a id="util-tf-changes"></a>
### 🔄 TensorFlow API Changes

Several parameters from the original function required special handling due to TensorFlow API changes:
1. **Compressed File Handling**: The compressed parameter used with `tf.contrib.data.TextLineDataset` has been updated to use the modern compression_type parameter in **TensorFlow 2.x**'s `tf.data.TextLineDataset`.
2. **Bucketing Strategy**: The original bucketing approach using `group_by_window` has been replaced with standard padding in the initial implementation, with potential for a future dedicated bucketing method using `tf.data.experimental.bucket_by_sequence_length`.
3. **Eager Execution**[^1]: I removed `initialized_uninitialzied()`since **TensorFlow 2.x** uses eager execution by default, which eliminates the needs for explicit variable initialization that was requried in **TensorFlow 1.x**'s graph mode. Varabiesl are automatically initalized when createed, without explicit call to `session.run`.

[^1]: Tensorflow's immediate evaluation mode where operations are executed as they're called, without first building a computational graph. It became the default execution model in TensorFlow 2.x, replacing the graph-based execution model of TensorFlow 1.x. It eliminates the needs for explicit variable initialization that was required in the original code, removing calls to `session.run(tf.global_variables_initializer())`, removing session management, and provides different mechanisms for ensuring reproducible results.

<a id="util-explain-0"></a>
### 📝 Explanantion with Examples

Since UniRep uses mLSTM architecture, we need to convert the amino acid sequences (strings of letters) into numerical representations that neural networks can process.

Suppose our example sequences are:
```
sequences = [
    "MGKV",         # Sequence 1
    "ACDEFGH",      # Sequence 2
    "STPQR"         # Sequence 3
]
```

The sequences will go through the following pre-processing steps:

<a id="util-explain-1"></a>
#### Step 1: Validate the sequence

The code checks if all characters in the sequence are valid amino acids using the `is_valid_seq` function:

```python
def is_valid_seq(seq: str, max_len: int = 2000) -> bool:
    valid_aas: Set[str] = set("MRHKDESTNQCUGPAVIFYWLO")
    return len(seq) < max_len and set(seq).issubset(valid_aas)
```

For our examples:
```python
is_valid_seq("MGKV")     # Returns True
is_valid_seq("ACDEFGH")  # Returns True
is_valid_seq("STPQR")    # Returns True
```

<a id="util-explain-2"></a>
#### Step 2: Convert amino acids to integers

The code maps each amino acid to a unique integer using the `aa_to_int` dictionary:

```python
aa_to_int: Dict[str, int] = {
    'M': 1, 'R': 2, 'H': 3, 'K': 4, 'D': 5, 'E': 6, 'S': 7, 'T': 8, 'N': 9, 'Q': 10, 'C': 11,
    'U': 12, 'G': 13, 'P': 14, 'A': 15, 'V': 16, 'I': 17, 'F': 18, 'Y': 19, 'W': 20, 'L': 21,
    'O': 22, # Pyrrolysine
    'X': 23, # Unknown
    'Z': 23, # Glutamic acid or Glutamine
    'B': 23, # Asparagine or aspartic acid
    'J': 23, # Leucine or isoleucine
    'start': 24, 'stop': 25,
}
```

For our example:
```
"MGKV" → [1, 13, 4, 16]
"ACDEFGH" → [15, 11, 5, 6, 18, 13, 8]
"STPQR" → [7, 8, 14, 10, 2]
```
<a id="util-explain-3"></a>
#### Step 3: Add start and stop tokens

Adding `start` and `stop` tokens in RNN-based models serve as crucial sequence boundary markers that help the model understand where meaningful data begins and ends. For protein sequences, these tokens (represented as integers 24 and 25) enable the model to recognize the natural N-terminus and C-terminus of proteins while providing the hidden state a chance to "warm up" before processing the actual sequence. This approach creates more consistent fixed-size representations across variable-length sequences, improves position-specific learning, and provides a mechanism for controlled sequence generation by signaling when to terminate the output. These special markers are fundamental to effective sequence modeling regardless of whether using LSTMs (as in UniRep) or other architectures.

The `aa_seq_to_int` function adds start and optionally stop tokens to the sequence:

```python
def aa_seq_to_int(s: str, include_stop: bool = True) -> List[int]:
    if include_stop:
        return [24] + [aa_to_int.get(a, 23) for a in s] + [25]
    else:
        return [24] + [aa_to_int.get(a, 23) for a in s]
```

For our examples with `include_stop=True`:
```
"MGKV" → [24, 1, 13, 4, 16, 25]
"ACDEFGH" → [24, 15, 11, 5, 6, 18, 13, 8, 25]
"STPQR" → [24, 7, 8, 14, 10, 2, 25]
```

<a id="util-explain-4"></a>
#### Step 4: Record sequence lengths

The code keeps track of the original sequence lengths (including start/stop tokens)
```python
seq_lengths = [len(seq) for seq in int_seqs]
```

For our examples:
```
seq_lengths = [6, 9, 7]
```

<a id="util-explain-5"></a>
#### Step 5: Pad sequences to the same length

To process sequences in batches, they need to have the same length. When working with protein sequences of varying lengths (which is almost always the case), padding helps standardize the input shape. The code pads shorter sequences with zeros:
```python
max_len = min(max(seq_lengths), self.max_seq_len)
padded_seqs = tf.keras.preprocessing.sequence.pad_sequences(
    int_seqs,
    maxlen=max_len,
    padding='post',
    truncating='post',
    value=0
)
```

For our examples (max length is 9):
```
"MGKV" → [24, 1, 13, 4, 16, 25, 0, 0, 0]
"ACDEFGH" → [24, 15, 11, 5, 6, 18, 13, 8, 25]
"STPQR" → [24, 7, 8, 14, 10, 2, 25, 0, 0]
```

<a id="util-explain-6"></a>
#### Step 6: Create TensorFlow dataset

After all the preparatory work () comes together, we can now build a data pipeline that prepares protein sequences for processing through the neural network. The `create_dataset` method creates the TensorFlow dataset for efficient processing. The dataset is created from the padded sequences:

```python
dataset = tf.data.Dataset.from_tensor_slices((
    seq_ids,
    padded_seqs,
    seq_lengths
))
```

This creates a dataset where each element is a tuple of (id, sequence, length):

```
(0, [24, 1, 13, 4, 16, 25, 0, 0, 0], 6)
(1, [24, 15, 11, 5, 6, 18, 13, 8, 25], 9)
(2, [24, 7, 8, 14, 10, 2, 25, 0, 0], 7)
```

<a id="util-explain-7"></a>
#### Step 7: Optinally Shuffle the dataset

If `shuffle=True`, the dataset elements are randomly shuffled:

```python
if self.shuffle:
    dataset = dataset.shuffle(
        buffer_size=min(len(int_seqs), 10000),
        seed=self.seed,
        reshuffle_each_iteration=True
    )
```

This might reorder our examples to something like:
```
(1, [24, 15, 11, 5, 6, 18, 13, 8, 25], 9)
(2, [24, 7, 8, 14, 10, 2, 25, 0, 0], 7)
(0, [24, 1, 13, 4, 16, 25, 0, 0, 0], 6)
```

<a id="util-explain-8"></a>
#### Step 8: Batch the dataset

The code groups the sequences into batches of size `batch_size`:
```python
dataset = dataset.batch(self.batch_size)
```

With `batch_size=2`, our examples would be batched as:
**Batch 1:**
```
ids = [1, 2]
sequences = [
    [24, 15, 11, 5, 6, 18, 13, 8, 25],
    [24, 7, 8, 14, 10, 2, 25, 0, 0]
]
lengths = [9, 7]
```

**Batch 2:**
```
ids = [0]
sequences = [
    [24, 1, 13, 4, 16, 25, 0, 0, 0]
]
lengths = [6]
```

<a id="util-explain-9"></a>
#### Step 9: Feed into mLSTM model

Finally, the batched sequences are fed into the mLSTM model:

```python
for batch in dataset:
    seq_ids, sequences, lengths = batch
    # Process with model
    representations = model(sequences, lengths)
```

The model processes each batch and generates representations for each sequence. The lengths tensor tells the model which positions are actual sequence data and which are padding.
For each sequence, the model produces three types of representations:
1. **Average hidden state** (`avg_hidden`): The average of all hidden states across the sequence
2. **Final hidden state** (`final_hidden`): The hidden state after processing the last amino acid
3. **Final cell state** (`final_cell`): The cell state after processing the last amino acid

These representations capture different aspects of the protein sequence and can be used for downstream tasks like protein function prediction, stability prediction, or protein engineering.

<a id="util-testing"></a>
### 🧪 Testing

All refactored methods are unit tested by `test_sequence.py`. The test can be run by `python3 -m tests.test_sequence` at the project root level.

```
=== RUNNING aa_conversion TESTS ===
Start token - Expected: 24
Start token - Actual: 24
Sequence conversion - Expected: [1, 15, 7, 4, 13, 6, 6, 21, 18, 8]
Sequence conversion - Actual: [1, 15, 7, 4, 13, 6, 6, 21, 18, 8]
Stop token - Expected: 25
Stop token - Actual: 25
No stop token - Expected: 25 not in sequence
No stop token - Actual: False
Round-trip conversion - Expected: MASKGEELFT
Round-trip conversion - Actual: MASKGEELFT
Token retention - Expected: startMASKGEELFTstop
Token retention - Actual: startMASKGEELFTstop

=== RUNNING sequence_validation TESTS ===
Valid sequence check - Expected: True
Valid sequence check - Actual: True
Invalid sequence check - Expected: False
Invalid sequence check - Actual: False
Too long sequence check - Expected: False
Too long sequence check - Actual: False

=== RUNNING sequence_tensor_conversion TESTS ===
Is tensor - Expected: True
Is tensor - Actual: True
Tensor shape - Expected: [1, 11]
Tensor shape - Actual: [1, 11]
First token - Expected: 24
First token - Actual: 24
Middle content - Expected: [1, 15, 7, 4, 13, 6, 6, 21, 18, 8]
Middle content - Actual: [1, 15, 7, 4, 13, 6, 6, 21, 18, 8]
Last token (with stop) - Expected: 25
Last token (with stop) - Actual: 25

=== RUNNING sequence_batcher TESTS ===
Padded shape - Expected: (2, 12)
Padded shape - Actual: (2, 12)
Sequence lengths - Expected: [12, 12]
Sequence lengths - Actual: [12, 12]
Is dataset - Expected: True
Is dataset - Actual: True
Batch size check - Expected: True
Batch size check - Actual: True
Sequence padding check - Expected: True
Sequence padding check - Actual: True

=== ALL TESTS PASSED ===
```

[🔝 Back to Table of Contents](#toc)

---

<a id="entry-2"></a>
## 🧠 Entry 2: LSTM Cell Implementation

<summary>Click to expand</summary>

<a id="cell-goals"></a>
### 🎯 Goals

- Refactor the LSTM cell implementations from TensorFlow 1.3 to 2.x API
- Implement deterministic behavior
- Create a clear class hierarchy
- Maintain compatibility with original model weights

<a id="cell-impl-0"></a>
### ⚙️ Implementation Details

<a id="cell-impl-1"></a>
#### 1️⃣ Base LSTM Cell Extraction & Refactoring

The first step in our refactoring was to create a foundational `BaseLSTMCell` class that would serve as the parent for all specialized LSTM implementations. This addressed several architectural goals:

- **Code Organization**: Extract common functionality from the original `mLSTMCell` and `mLSTMCell1900` classes
- **TensorFlow 2.x Compatibility**: Update to modern Keras layer inheritance
- **Deterministic Behavior**: Add infrastructure for reproducible results

Analysis of the original code `unirep.py` reveals that there are two LSTM cell variants and one stack implementation:
1. `mLSTMCell1900` (**lines 125-207**) : a fixed-size multiplicative LSTM cell with 1900 units
2. `mLSTMCell` (**lines 209-399**) - a configurable multiplicative LSTM cell with customizable initialization.
3. `mLSTMCellStackNPY` (**lines 301-390**): Stack of mLSTM cells for deep networks

Moreover, original code in `mLSTMCell` and `mLSTMCell1900` had nearly identical:
- `state_size` and `output_size` properties
- `zero_state` methods
- weight normalization logic
- basic LSTM computation pattern

Therefore, a new base LSTM cell class, `BaseLSTMCell`, was created in `cells/base_cell.py` to generalize a RNN cell structure from `mLSTMCell` and `mLSTMCell1900` from `unirep.py` and abstracts common functionality from the original classes.

```python
class BaseLSTMCell(tf.keras.layers.Layer):
    def __init__(
        self,
        num_units: int,
        weight_norm: bool = True,
        name: str = "base_lstm",
        **kwargs: Any
):
```

The `BaseLSTMCell` constructor parameters were derived from previous analysis of the original implementations

- `num_units`:  represents the size of cell's hidden state. Originally used in both `mLSTMCell1900` (**line 127**) and `mLSTMCell` (**line 212**)
- `weight_norm`: controls whether weight normalization is applied. Corresponds to `wn` in the original code (**lines 129** and **222**)
- `name`: replaces `scope` from the original (**lines 130** and **223**) to align with TensorFlow 2.x Keras layer naming conventions
- `**kwargs`: added to support future extensibility and Keras layer configuration options

Extracted common patterns into `BaseLSTMCell` in `cells/base_cell.py`:

| Original Feature | Original Location | Refactored Implementation |
|------------------|-------------------|---------------------------|
| State size property | **lines 141-144, 242-245** | `state_size` property |
| Output size property | **lines 146-149, 247-250** | `output_size` property |
| Zero state creation | **lines 151-154, 252-255** | `get_initial_state()` method |
| Weight normalization | **lines 193-197, 285-289** | `apply_weight_norm()` method |

Refactored implementations have better type annotation, and updated return types to match TensorFlow 2.x expectations.

Example: refactored `state_size` and  `output_size` properties

```python
@property
def state_size(self) -> Tuple[tf.TensorShape, tf.TensorShape]:
	"""Return size of cell and hidden states."""
    return (tf.TensorShape([self._num_units]), tf.TensorShape([self._num_units]))

@property
def output_size(self) -> tf.TensorShape:
    """Return size of output (hidden state)."""
    return tf.TensorShape([self._num_units])
```


**Key TensorFlow API Changes**

- Class inheritance: replace `tf.nn.rnn_ce ll.RNNCell` with `tf.keras.layers.Layer`
- Dimension arguments: replace`dim=0` with `axis=0`
- State initialization: `zero_state()` → `get_initial_state()`

<a id="cell-impl-2"></a>
#### 2️⃣ New Methods for Deterministic Behavior and API Consistency

To address the deterministic behavior requirement, we added two utility methods to the base class:

- `deterministic_reduce_mean()` ensures consistent reduction operations on GPU
- `set_seed()` standardizes random seed setting for reproducibility

```python
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
```

Finally, we adapted to TensorFlow 2.x's build-then-call pattern by implementing:
- An abstract `build()` method to be overridden by subclasses: required by TensorFlow 2.x Keras layer API, this is called once before the first forward pass to initialize weights dynamically based on input dimensions.
- An abstract `call()` method with the expected TensorFlow 2.x signature `(inputs, states, training=None)`: replacing the old `__call__` pattern, this core computation method adds `training` paramter for dropout/normalization behavior. Must be implmeneted by subclasses.

```python
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
```

**Note**: Several utility methods were considered for the base cell implementation but were deprioritized to focus on core functionality:

- Weight initialization helpers (orthogonal and ones initializers)
- Weight persistence methods (save/load to NumPy files)


<a id="cell-impl-3"></a>
#### 3️⃣ Implement mLSTM Cell

The multiplicative LSTM (`mLSTM`) cell is a critical component of the UniRep architecture, combining the memory capabilities of LSTM with multiplicative interactions that make it particularly effective for learning protein sequence representations. In the original code, the `mLSTMCell` class (**lines 209-299**) implements a configurable multiplicative LSTM cell with customizable initialization. Our refactoring focuses on updating to TensorFlow 2.x while maintaining the exact same mathematical operations with the following design pricniples:

- **Mathematical Fidelity**: Preserve the exact computational graph of the original implementation while updating the API
- **Deterministic Behavior**: Ensure reproducible results through explicit seeding and deterministic operations
- **Modern TensorFlow**: Leverage TensorFlow 2.x features including the Keras Layer API
- **Weight Compatibility**: Maintain ability to load pre-trained weights from the original model


Starting with the constructor, the original constructor (**lines 209-226**) contained many individual initializers and configuration parameters:
```python
# Original code (unirep.py, lines 209-226)
class mLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 wx_init=tf.orthogonal_initializer(),
                 wh_init=tf.orthogonal_initializer(),
                 wmx_init=tf.orthogonal_initializer(),
                 wmh_init=tf.orthogonal_initializer(),
                 b_init=tf.orthogonal_initializer(),
                 gx_init=tf.ones_initializer(),
                 gh_init=tf.ones_initializer(),
                 gmx_init=tf.ones_initializer(),
                 gmh_init=tf.ones_initializer(),
                 wn=True,
                 scope='mlstm',
                 var_device='cpu:0'):
        super(mLSTMCell, self).__init__()
        self._num_units = num_units
        self._wn = wn
        self._scope = scope
        self._var_device = var_device
        # ... stores initializers as instance variables
```
Our refactored constructor is streamlined to leverage the `BaseLSTMCell` parent class and focus on the unique aspects of the mLSTM cell
- `num_units`: Controls the dimensionality of the hidden and cell states
- `seed`: Added for deterministic behavior (not present in original implementation)
- `name`: Keras-style naming convention to replace the original `scope` parameter
- `weight_norm`: rename `wn` → `weight_norm` that enables L2 normalization of weights for improved training stability
- Removed device placement parameter (`var_device`) as this is handled differently in TensorFlow 2.x
- Added type annotations for better code safety

```python
# cells/mlstm_cell.py
class mLSTMCell(BaseLSTMCell):
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
```

For **weight initialization**, the original code used `tf.get_variable()` with initializers for each weight (**lines 264-283**).

```python
with tf.variable_scope(self._scope):
    wx = tf.get_variable(
        "wx", initializer=self._wx_init)
    wh = tf.get_variable(
        "wh", initializer=self._wh_init)
    wmx = tf.get_variable(
        "wmx", initializer=self._wmx_init)
    wmh = tf.get_variable(
        "wmh", initializer=self._wmh_init)
    b = tf.get_variable(
        "b", initializer=self._b_init)
    if self._wn:
        gx = tf.get_variable(
            "gx", initializer=self._gx_init)
        gh = tf.get_variable(
            "gh", initializer=self._gh_init)
        gmx = tf.get_variable(
            "gmx", initializer=self._gmx_init)
        gmh = tf.get_variable(
            "gmh", initializer=self._gmh_init)
```
In TensorFlow 2.x, we use the `build()` method to create weights.  The `build` method exemplifies the modern approach of separating weight creation from computation, which differs substantially from the TensorFlow 1.x approach where weights were created during the first call.

- Added explicit `seed` parameter for deterministic initialization
- Use `self.add_weight()` instead of `tf.get_variable()`
- Dynamically determine input dimension from `input_shape` parameter
- Set `self.built = True` to signal that the layer is built

```python
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
```

Moving onto **cell computation**, the original code implemented the core mLSTM computation in the `call()` methods (**lines 198-206, 290-298**).

```python
 m = tf.matmul(inputs, wmx) * tf.matmul(h_prev, wmh)
        z = tf.matmul(inputs, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f * c_prev + i * u
        h = o * tf.tanh(c)
```

I implemented `call()` that preserves the mathematical structure of the original mLSTM while updating to TensorFlow 2.x API conventions. The multiplicative interaction between the input and previous hidden states are the same:
- Updated parameter signature to include `training` parameter
- Updated `tf.split()` to use `axis` parameter instead of positional argument
- Improved type annotations for better code safety
- Added more detailed comments to explain each step of the computation

```python
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
```

For **weight loading**, original code loads weight differently acorss three different classes:
- for `mLSTMCell1900` (**lines 162-191**) loads pre-trained weights directly during initialization inside the `call()` method. It assumes weights will always be loaded from `NumPy` files and is hard-coded for this specific model.
- `mLSTMCell` doesn't load weights directly, and receives initializers as constructor parameters (**lines 211-225**). By default, these parameters are intialized with random orthogonal weights. If pre-trained weights need to be used, they are passed to the initializers (**lines 264-283**).
- `mLSTMStackNPY` creates a stack of `mLSTMCell` instances, and loads pre-trained weights in its constructor (**lines 323 - 339**) to each `mLSTMCell`.

This inconsistent weight loading patterns creates significant maintenance challenges. The `mLSTMCell1900` loads weights during call execution, the base `mLSTMCell` expects initializers to be passed in, and `mLSTMCellStackNPY` loads weights during construction to configure child cells. The pattern increases cognitive overhead, complicates debugging, and creates potential for silent failures when working with pre-trained weights.

To address this issue with more robust weight loading mechanism, I impmlemented `load_weights_from_numpy()` that provides explicit weight loading after the cell is built, separating model definition from weight initialization. The method first validates that the cell is already built to prevent loading weights into non-existent variables. It then uses a helper function `load_weight()` to handle individual weight loading with proper error handling, checking for file existence and providing clear warnings for missing files. The implementation directly assigns loaded values to existing variables using TensorFlow's `assign()` method, and conditionally loads weight normalization parameters only when enabled. This explicit, unified approach improves code maintainability, provides better error handling, and offers flexibility to load weights at any point after cell initialization, addressing the inconsistencies found in the original implementation.

```python
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
```
<a id="cell-impl-4"></a>
#### 4️⃣ Stacked mLSTM Cell Implementation

The stacked mLSTM cell is essential for creating deeper network architectures by combining multiple mLSTM layers. In the original implementation, `mLSTMCellStackNPY` (**lines 301-390**) creates a stack of mLSTM cells with residual connections and dropout support. Our refactoring goal was to modernize this approach for TensorFlow 2.x while maintaining the same functionality with the following design principles:

- **Modularity**: Create a clean composition of individual mLSTM cells
- **Modern Layer API**: Update to TensorFlow 2.x Keras layer inheritance
- **Regularization Options**: Support for dropout and residual connections
- **Deterministic Behavior**: Ensure reproducible results through explicit seeding
- **Weight Compatibility**: Maintain the ability to load pre-trained weights

**Constructor implementation**: as discussed previously, the original constructor (**lines 303-344**) combined multiple different responsibilities in one method, handling both cell creation and weight loading together rather than separating these distinct operations.
```python
# Original code (unirep.py, approximate lines 303-344)
class mLSTMCellStackNPY(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 num_layers,
                 dropout=None,
                 residual=False,
                 weight_path="./",
                 wn=True,
                 scope='mlstm_stack'):
        # ...cell creation logic...
        # ...weight loading logic...
```

Our refactored constructor separates these concerns by
- Moved weight loading to a separate method `load_weights_from_numpy()` for better separation of concerns
- Renamed parameters for clarity (e.g., `dropout` → `dropout_rate`, `residual` → `residual_connections`)
- Added type annotations for better code safety
- Added a `seed` parameter for deterministic behaviormodilarity

```python
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
    # ... constructor body ...
```

**Cell creation logic** in the original implementation is somewhat hardcoded (**lines 325-339**): weight files are directly loaded with hardcoded naming patterns inside the constructor. Each weight file path is then created using string formatting with fixed patterns. Dropout is also handled in a fixed way as it only applies dropout to all but the last layer, with no option for different dropout rates per layer.

```python
# Original code (lines 325-339)
layers = [mLSTMCell(
    num_units=self._num_units,
    wn=self._wn,
    scope=self._scope + str(i),
    var_device=self._var_device,
    wx_init=np.load(join(bs + "{0}_mlstm_stack{1}_wx:0.npy".format(i,i))),
    # ... other weight initializations ...
) for i in range(self._num_layers)]

if self._dropout:
    layers = [
        tf.contrib.rnn.DropoutWrapper(
            layer, output_keep_prob=1-self._dropout) for layer in layers[:-1]] + layers[-1:]
```

The refactored version improves this to better align with TensorFlow 2.x layer composition: explicit unique seed per layer for deterministic behavior, more explicit layer structure with consistent naming shceme, and clear organization of cell creation with dedicated dropout layers.
```python
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
```

For **state management**, the stacked cell needs to track states for all layers. The original implementation used TensorFlow 1.x's RNNCell interface with `zero_state()` (**lines 346-362**) and direct tuple manipulation without type annotations, while lacking training mode parameters for dropout.

```python
# Original code (lines 346-362)
@property
def state_size(self):
    # The state is a tuple of c and h
    return (
        tuple(self._num_units for _ in range(self._num_layers)),
        tuple(self._num_units for _ in range(self._num_layers))
        )

@property
def output_size(self):
    # The output is h
    return (self._num_units)

def zero_state(self, batch_size, dtype):
    c_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
    h_stack = tuple(tf.zeros([batch_size, self._num_units], dtype=dtype) for _ in range(self._num_layers))
    return (c_stack, h_stack)
```

The refactored implementation updates to TensorFlow 2.x with improved type annotations:
- `zero_state()` is replaced by `get_initial_state()`
- Added flexibility to infer parameters from inputs
- Improved type annotations for better code safety

```python
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
    # ... method implementation ...
```

**Forward pass** implementation in the original code (**lines 374-381**) used conditional logic to process the first layer differently:
```python
# lines 374-381
for i, layer in enumerate(self._layers):
    if i == 0:
        h, (c,h_state) = layer(inputs, (c_prev[i],h_prev[i]))
    else:
        h, (c,h_state) = layer(new_outputs[-1], (c_prev[i],h_prev[i]))
    new_outputs.append(h)
    new_cs.append(c)
    new_hs.append(h_state)
```

The refactored version uses a more consistent apporach by initializing `lay_input = inputs` before the loop, then updating it after each layer processes it, eliminating the needs for conditional logic that was present in the original for first layer
```python
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
```

**Residual connections** are handled similarly in the original (**lines 383-388**) and refactored implementation where refactored version uses modern TensorFlow operations:
```python
# Original code (lines 383-388)
if self._res_connect:
    # Make sure number of layers does not affect the scale of the output
    scale_factor = tf.constant(1 / float(self._num_layers))
    final_output = tf.scalar_mul(scale_factor,tf.add_n(new_outputs))
else:
    final_output = new_outputs[-1]
```

```python
# Refactored version in mlstm_stack.py
# Apply residual connections if specified
if self._residual_connections:
    # Scale output to maintain magnitude
    scale_factor = 1.0 / float(self._num_layers)
    final_output = tf.reduce_mean(tf.stack(outputs, axis=0), axis=0)
else:
    # Just use the output from the last layer
    final_output = outputs[-1]
```

Finally for **weight loading**, the original implementation loaded weights directly during initialization, hard-coding file paths and weight loading logic in the constructor (**lines 303-339**). Our refactored implementation provides a dedicated method for weight loading:
```python
def load_weights_from_numpy(self, load_path: str) -> None:
    """
    Load cell weights from numpy files.

    Args:
        load_path: Directory containing weight files
    """
    base_scope = "rnn_mlstm_stack_mlstm_stack"

    # Load weights for each cell in the stack
    for i, cell in enumerate(self._cells):
        # ... weight mapping logic ...

        # Load the weights into the cell
        cell.load_weights_from_numpy(cell_path)
```


<a id="cell-tf-changes"></a>
### 🔄 Tensorflow API Changes

| Original (TF 1.x) | Refactored (TF 2.x) | Component | Benefit |
|-------------------|---------------------|-----------|---------|
| `tf.nn.rnn_cell.RNNCell` | `tf.keras.layers.Layer` | Cell inheritance | Better integration with Keras ecosystem |
| `tf.get_variable()` | `self.add_weight()` | Weight creation | Automatic variable tracking and serialization |
| `zero_state()` | `get_initial_state()` | State initialization | Consistent with Keras RNN interfaces |
| `tf.nn.l2_normalize(x, dim=0)` | `tf.nn.l2_normalize(x, axis=0)` | Weight normalization | Updated parameter naming |
| `tf.split(z, 4, 1)` | `tf.split(z, 4, axis=1)` | Tensor splitting | Updated parameter naming |
| `tf.contrib.data.TextLineDataset` | `tf.data.TextLineDataset` | Dataset creation | Moved from contrib to core API |
| `tf.distributions.Categorical` | `tf.random.categorical` | Sampling | Updated to more stable implementation |
| `tf.global_variables_initializer()` | N/A (eager execution) | Variable initialization | Automatic initialization in eager mode |
| `sess.run()` | Direct tensor operations | Session execution | Eager execution eliminates session management |
| Explicit `scope` parameter | `name` parameter | Variable naming | Aligns with Keras naming conventions |
| `tf.contrib.rnn.DropoutWrapper` | `tf.keras.layers.Dropout` | Dropout application | Direct layer application instead of wrappers |
| `group_by_window` for bucketing | Standard padding approach | Sequence batching | Simplified data pipeline |
| `tf.scalar_mul` + `tf.add_n` | `tf.reduce_mean` | Residual connections | More efficient implementation |
| No explicit training mode | `training` parameter | Training/inference | Proper behavior switching for regularization |
| No explicit seeding | Seed parameters | Random operations | Deterministic results |
| Hard-coded device placement | TF 2.x device placement | Hardware utilization | Better automatic device management |
| `initialized_uninitialized()` | N/A | Variable initialization | No longer needed with eager execution |

<a id="cell-testing"></a>
### 🧪 Testing

All refactored methods are unit tested by `test_cells.py`. The test can be run by `python3 -m tests.test_cells` at the project root level.
```
Running all cell tests...

=== Testing BaseLSTMCell ===
State size: (TensorShape([64]), TensorShape([64]))
Output size: (64,)
2025-04-06 14:03:48.982409: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
Initial state shapes: [TensorShape([2, 64]), TensorShape([2, 64])]
Normalized weight shape: (10, 64)
BaseLSTMCell tests passed!

=== Testing mLSTMCell ===
Output shape: (2, 64)
Next cell state shape: (2, 64)
Next hidden state shape: (2, 64)
Deterministic output: True
mLSTMCell tests passed!

=== Testing mLSTMCell Weight Normalization ===
Outputs with and without weight normalization are different: True
mLSTMCell weight normalization tests passed!

=== Testing mLSTMCell Weight Loading ===
Successfully loaded weights
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_wx:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_wh:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_wmx:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_wmh:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_b:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_gx:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_gh:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_gmx:0.npy not found
Warning: Weight file /invalid/path/rnn_mlstm_mlstm_gmh:0.npy not found
Handled invalid path correctly
mLSTMCell weight loading tests passed!

=== Testing mLSTMCell Computational Correctness ===
Output: [[0.2609468 0.2609468]]
Next cell state: [[0.4105818 0.4105818]]
Next hidden state: [[0.2609468 0.2609468]]
mLSTMCell computational correctness tests passed!

=== Testing mLSTMCell Edge Cases ===
Output with zero inputs: [[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
Warning: Cell produced output with NaN inputs
mLSTMCell edge case tests passed!

=== Testing StackedMlstmCell ===
Testing initialization...
Testing initial state...
Testing forward pass...
Testing residual connections...
Testing dropout...
Testing deterministic behavior...
Testing sequence processing...
StackedMlstmCell tests passed!

✅ All tests passed!
```

[🔝 Back to Table of Contents](#toc)

<a id="entry-3"></a>
## 🧩 Entry 3: Refactor Model Architecture

<a id="model-goals"></a>
### 🎯 Goals

- Refactor model implementations from TensorFlow 1.3 to 2.x API
- Centralize configuration management with validation
- Create a clear class hierarchy for model variants
- Maintain compatibility with original model weights
- Enable deterministic behavior
- Improve code organization and reusability

<a id="model-impl-0"></a>
### ⚙️ Implementation Details

<a id="model-impl-1"></a>
#### 1️⃣ Base Configuration Implementation

The first step in our refactoring was to create a configuration system to centralize and validate all model parameters. This addressed several architectural goals:
- **Code Organization**: Separate configuration from implementation
- **Parameter Validation**: Prevent invalid configurations
- **Reproducibility**: Enable saving and loading of model configurations

Analysis of the original `unirep.py` reveals scattered configuration across three model classes:

- `babbler1900` (**lines 393-674**): the largest model with 1900 units with hardcoded parameters with limited configurability
- `babbler256` (**lines 676-791**): a medium model with 256 units with duplicated initialization with different values
- `babbler64` (**lines 794-908**): the smallest model with 64 units with further duplication with minimal unique code

Our implementation adopts a two-level configuration hierarchy in `config/model_config.py`:
- `BaseConfig`: A generic configuration class that provides core functionality like validation, serialization, and deserialization. This class serves as a foundation for all configuration types in the project, enabling a consistent interface and reducing code duplication.
- `UniRepModelConfig`: A specific configuration class for UniRep models that defines and validates model-specific parameters.

This two-level configuration hierarchy enhances the codebase architecture by separating generic functionality in `BaseConfig` from model-specific parameters in `UniRepModelConfig`. This approach facilitates easy expansion to new configuration types while maintaining a consistent interface for validation, saving, and loading operations. By centralizing serialization logic, we improve maintainability and reduce redundancy across the implementation.
```python
# config/model_config.py
from typing import Optional, Dict, Any
import os
import json

class BaseConfig:
    """Base configuration with validation and serialization"""

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate(self) -> bool:
        """Validate configuration."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: str) -> 'BaseConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

class UniRepModelConfig(BaseConfig):
    """Configuration for UniRep models."""

    def __init__(
        self,
        rnn_size: int = 64,
        embed_dim: int = 10,
        num_layers: int = 4,
        weight_norm: bool = True,
        model_path: Optional[str] = None,
        batch_size: int = 256,
        seed: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        residual_connections: bool = False,
        **kwargs: Any
    ):
        """
        Initialize UniRep model configuration.

        Args:
            rnn_size: Number of units in RNN cell
            embed_dim: Dimension of embedding layer
            num_layers: Number of layers in stacked model
            weight_norm: Whether to use weight normalization
            model_path: Path to pretrained weights
            batch_size: Default batch size
            seed: Random seed for deterministic behavior
            dropout_rate: Optional dropout rate
            residual_connections: Whether to use residual connections
        """
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.weight_norm = weight_norm
        self.model_path = model_path
        self.batch_size = batch_size
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.residual_connections = residual_connections
        super(UniRepModelConfig, self).__init__(**kwargs)

    def validate(self) -> bool:
        """Validate UniRep model configuration."""
        # Validate RNN size
        if self.rnn_size <= 0:
            raise ValueError(f"rnn_size must be positive, got {self.rnn_size}")

        # Validate embedding dimension
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        # Validate number of layers
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate dropout rate
        if self.dropout_rate is not None and (self.dropout_rate < 0 or self.dropout_rate >= 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")

        # Validate model path
        if self.model_path is not None and not os.path.exists(self.model_path):
            raise ValueError(f"model_path does not exist: {self.model_path}")

        return True
```

Below is a summary of attributes and methods that were refactored into `BaseConfig` and `UniRepModelConfig`:
| **Attribute/Method**       | **Origin in `unirep.py`**                                                                 | **Purpose**                                                                 |
|----------------------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `rnn_size`                 | `self._rnn_size` in `babbler1900` (**line 399**), `babbler256` (**line 685**), `babbler64` (**line 803**)        | Number of units in the RNN cell.                                          |
| `embed_dim`                | `self._embed_dim` in `babbler1900` (**line 401**) and `babbler256` (**line 687**), `babbler64` (**line 805**)       | Dimension of the embedding layer.                                          |
| `num_layers`               | `self._num_layers` in `babbler256` (**line 688**) and `babbler64` (**line 806**)         | Number of layers in the stacked model.                                     |
| `weight_norm`              | `self._wn` in `babbler1900` (**line 402**), `babbler256` (**line 691**), `babbler64` (**line 807**)               | Whether to use weight normalization.                                       |
| `model_path`               | `self._model_path` in `babbler1900` (**line 404**), `babbler256` (**line 677**), `babbler64` (**line 809**)       | Path to pretrained weights.                                                |
| `batch_size`               | `self._batch_size` in `babbler1900` (**line 405**), `babbler256` (**line 692**), `babbler64` (**line 810**)      | Default batch size.                                                        |
| `dropout_rate`             | Derived from `mLSTMCellStackNPY` (**lines 341-343**)                                     | Optional dropout rate for regularization.                                  |
| `residual_connections`     | Derived from `mLSTMCellStackNPY` (**lines 383-388**)                                     | Whether to use residual connections.                                       |
| `validate()`               | Derived from hardcoded parameter logic in `unirep.py`                                    | Validates configuration parameters.                                        |
| `seed`                     | **New** | Random seed for deterministic behavior.                                    |
| `to_dict()`                | **New**                                                                                 | Converts configuration to a dictionary.                                    |
| `save()`                   | **New**                                                                                 | Saves configuration to a JSON file.                                        |
| `from_dict()`              | **New**                                                                                 | Creates configuration from a dictionary.                                   |
| `load()`                   | **New**                                                                                 | Loads configuration from a JSON file.                                      |


<a id="model-impl-2"></a>
#### 2️⃣ Base Model Implementation

After establishing a robust configuration system, the next step was to refactor the model architecture.


In the original code, the model classes share significant functionality: `babbler256` and `babbler64` inheriting from `babbler1900` but overriding the constructor with different model configurations. These models use TensorFlow 1.x APIs (`tf.contrib`, `placeholders`, `sessions`).

- `babbler1900` (**lines 393-674**): The largest model with 1900 units
- `babbler256` (**lines 676-791**): Medium model with 256 units that inherits from babbler1900
- `babbler64` (**lines 794-908**): Small model with 64 units that also inherits from babbler1900

Despite the inheritance, `babbler256` and `babbler64` override the constructor with different model configurations rather than leveraging proper parameterization. All three models use TensorFlow 1.x APIs (`tf.contrib`, `placeholders`, `sessions`) that need modernization. Our refactorization creates a `BaseModel` class inherits from TensorFlow 2.x `tf.keras.Model`. This model serves as a foundation class that all babblers can inherit from.

Starting with **constructor implemention**, original code from `babbler1900` constructor (**lines 395-473**) exhibits several architectural limitations that hinder maintainability and flexibility:
- It directly hardcodes model parameters (**lines 395-405**) like RNN size (`self._rnn_size = 1900`) and embedding dimensions (`_self.embed_dim = 10`) with minimal parameterization, making it difficult to create model variants without code duplication.
- It intertwines configuration with model construction, leading to `babbler256` (**lines 681-692**) and `babbler64` (**lines 799-810**) overriding most of the constructor code just to change a few parameter values.
- This approach lacks modern features like deterministic seeding and type safety, while forcing subclasses to reimplement common functionality rather than extend it in a clean hierarchy.
```python
# lines 395-405
def __init__(self,
    model_path="./pbab_weights",
    batch_size=256
    ):
    # hard coded model parameters
    self._rnn_size = 1900
    self._vocab_size = 26
    self._embed_dim = 10
    self._wn = True
    self._shuffle_buffer = 10000
    self._model_path = model_path
    self._batch_size = batch_size
# ...
# lines 681-692
def __init__(self,
    model_path="./256_weights/",
    batch_size=256
    ):
     # Override parent constructor just to change RNN size
    self._rnn_size = 256 # Hardcoded size change
    self._vocab_size = 26
    self._embed_dim = 10
    self._num_layers = 4
    self._wn = True
    self._shuffle_buffer = 10000
    self._model_path = model_path
    self._batch_size = batch_size
# ...
# lines 799-810
def __init__(self,
    model_path="./64_weights/",
    batch_size=256
    ):
     # Override parent constructor just to change RNN size
    self._rnn_size = 64 # Hardcoded size change
    self._vocab_size = 26
    self._embed_dim = 10
    self._num_layers = 4
    self._wn = True
    self._shuffle_buffer = 10000
    self._model_path = model_path
    self._batch_size = batch_size
```
Our refactored `BaseModel` constructor improves on the following aspects:
- uses configuration object `UniRepModelConfig` instead of individual parameters.
- Adds explicit random seed setting for deterministic behavior
- Uses `tf.keras.layers.Embedding` instead of raw tensor operations
- Uses `tf.keras.layers.Dense` for output projection
- Explicit weight loading with error checking
- Type annotations for better code safety

```python
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
```

For **forward pass implementation**, the original code in `babbler1900` class used TensorFlow 1.x's session-based graph execution model, with placeholders and explicit session management for the forward pass. This approach had the following limitations: the forward pass logic was split across multiple methods, relying on numerous placeholders, and requiring explicit session management for each computation.

```python
# Lines 406-421 (Placeholder definitions)
self._batch_size_placeholder = tf.placeholder(tf.int32, shape=[], name="batch_size")
self._minibatch_x_placeholder = tf.placeholder(tf.int32, shape=[None, None], name="minibatch_x")
# ... more placeholders ...

# Lines 434-444 (Graph construction in constructor)
embed_matrix = tf.get_variable(...)
embed_cell = tf.nn.embedding_lookup(embed_matrix, self._minibatch_x_placeholder)
self._output, self._final_state = tf.nn.dynamic_rnn(...)

# Lines 504-515 (get_rep method)
with tf.Session() as sess:
    initialize_uninitialized(sess)
    final_state_, hs = sess.run(
        [self._final_state, self._output], feed_dict={...}
    )
```
Our refactored implementation adopts TensorFlow 2.x's eager execution model with a Keras-style `call()` method. This method in the `BaseModel` provides common embedding logic but leaves RNN cell-specific processing to subclasses, allowing different model variants to specialize appropriately while maintaining a consistent interface:

```python
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
```
The original code implemented **representation extraction** in the `get_rep()` methods (**lines 497-524**, **763-791**, **881-908**), with significant code duplication across model variants. Each model required creating new sessions for every sequence and used awkward feed dictionary patterns:
```python
# lines 497-524
def get_rep(self, seq):
    ...
    with tf.Session() as sess:
        initialize_uninitialized(sess)
    int_seq = aa_seq_to_int(seq.strip())[:-1]
    final_state_, hs = sess.run(
        [self._final_state, self._output], feed_dict={
            self._batch_size_placeholder: 1,
            self._minibatch_x_placeholder: [int_seq],
            self._initial_state_placeholder: self._zero_state}
    )
```
Our refactored implementation unifies this functionality in a clean, resuable method. This approach eliminates session management overhead, simplifies the interface, and delegates state processing to model-specific implementations through the abstract `_process_final_state()` method.

```python
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

    # Add batch dimension and process through model
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
```

The original code implemented **sequence generation** in `get_babble()` method (**lines 526-563**) which suffered from non-deterministic behavior and inefficient session management:

```python
# Original implementation (lines 526-563)
with tf.Session() as sess:
    initialize_uninitialized(sess)
    # Process seed sequence
    seed_samples, final_state_ = sess.run(
        [self._sample, self._final_state],
        feed_dict={
            self._minibatch_x_placeholder: [int_seed],
            self._initial_state_placeholder: self._zero_state,
            self._batch_size_placeholder: 1,
            self._temp_placeholder: temp
        }
    )
    # Then repeatedly run sessions for each new token
    for i in range(length - len(seed)):
        pred_int, final_state_ = sess.run(
            [self._sample, self._final_state],
            feed_dict={...}
        )
```

Our refactored implementation provides deterministic sequence generation through a clean, eager execution approach  by reusing state across iterations and ensures deterministic behavior through controlled sampling.
```python
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

    # Generate one token at a time
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
        next_aa = int_to_aa[next_token + 1]
        sequence += next_aa

        # Update sequence IDs with only the new token for the next iteration
        seq_ids = [next_token + 1]

    return sequence
```

We implemented two **utility methods** to centralize functionality that was previously scattered or duplicated in the original code. These methods provide consistent interfaces with improved error handling and deterministic behavior:
```python
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
```
Finally, for **weight management**, we implemented a consistent weight saving mechanism to replace the original, model-specific weight handling:
```python
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
```

#### 3️⃣ Specific Model Implementations




<a id="model-tf-changes"></a>
### 🔄 Tensorflow API Changes

<a id="model-testing"></a>
### 🧪 Testing

[🔝 Back to Table of Contents](#toc)
