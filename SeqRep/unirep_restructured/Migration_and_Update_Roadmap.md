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
- [Implementation Details](#cell-impl-0)
  - [1️⃣ Base LSTM Cell Extraction & Refactoring](#cell-impl-1)
  - [2️⃣ Implement mLSTM Cell](#cell-impl-2)
  - [Explanation](#explanation)
  - [Stacked mLSTM Cell Implementation](#4-stacked-mlstm-cell-implementation)
### [Entry 3: Refactor Model](#entry-3)


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

<a id="cell-impl-1">></a>
#### 1️⃣ Base LSTM Cell Extraction & Refactoring


Original code in `unirep.py` contains two LSTM cell variants and one stack implementation:
1. `mLSTMCell1900` (**lines 125-207**) : a fixed-size multiplicative LSTM cell with 1900 units
2. `mLSTMCell` (**lines 209-399**) - a configurable multiplicative LSTM cell with customizable initialization.
3. `mLSTMCellStackNPY` (**lines 301-390**): Stack of mLSTM cells for deep networks

A base LSTM cell class, `BaseLSTMCell`, was created in `cells/base_cell.py` to serve as the foundation for all specialized LSTM implementations. Specifically, this is to generalize a RNN cell structure from `mLSTMCell` and `mLSTMCell1900` from `unirep.py` and abstracts common functionality from the original classes.

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

The `BaseLSTMCell` constructor parameters were derived from analysis of the original implementations

- `num_units`:  represents the size of cell's hidden state. Originally used in both `mLSTMCell1900` (**line 127**) and `mLSTMCell` (**line 212**)
- `weight_norm`: controls whether weight normalization is applied. Corresponds to `wn` in the original code (**lines 129** and **222**)
- `name`: replaces `scope` from the original (**lines 130** and **223**) to align with TensorFlow 2.x Keras layer naming conventions
- `**kwargs`: added to support future extensibility and Keras layer configuration options

Extracted common patterns into `BaseLSTMCell` in `cells/base_cell.py`:

| Original Feature | Original Location | Refactored Implementation |
|------------------|-------------------|---------------------------|
| State size property | Lines 141-144, 242-245 | `state_size` property |
| Output size property | Lines 146-149, 247-250 | `output_size` property |
| Zero state creation | Lines 151-154, 252-255 | `get_initial_state()` method |
| Weight normalization | Lines 193-197, 285-289 | `apply_weight_norm()` method |

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

Added **New Methods for Deterministic Behavior**:

- `deterministic_reduce_mean()`: Ensures consistent reduction operations on GPU
- `set_seed()`: Standardizes random seed setting for reproducibility

Added **New Method Structure for TensorFlow 2.x Compatibility**:

1. **`build(input_shape)`**:
   - Required by TensorFlow 2.x Keras layer API
   - Called once before the first forward pass to initialize weights
   - Dynamically creates weight variables based on input dimensions
   - Creates a clear separation between weight initialization and computation
2. **`call(inputs, states, training=None)`**:
   - Core computation method in TensorFlow 2.x
   - Replaces the old RNNCell's `__call__` pattern
   - Adds `training` parameter for dropout/normalization behavior
   - Abstract method in the base class, must be implemented by subclasses

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

These methods may be implemented in specific cell subclasses as needed.

<a id="cell-impl-2"></a>
#### 2️⃣ Implement mLSTM Cell

In the original code, the `mLSTMCell` class (**lines 209-299**) implements a configurable multiplicative LSTM cell with customizable initialization. This cell forms the core of the UniRep model's sequence processing capabilities. Our refactoring focuses on updating to TensorFlow 2.x while maintaining the exact same mathematical operations.

The original constructor (**lines 209-226**) contained many individual initializers and configuration parameters:

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

- Added `seed` parameter for deterministic behavior
- Removed device placement parameter (`var_device`) as this is handled differently in TensorFlow 2.x
- Renamed parameters for clarity (`wn` → `weight_norm`, `scope` → `name`)
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

For **weight initialization**, the original code in `unirep.py` used `tf.get_variable()` with initializers for each weight (**lines 264-283**).

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

In TensorFlow 2.x, we use the `build()` method to create weights:

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

For **cell computation**, the original code in `unirep.py` implemented the core mLSTM computation in the `call()` method (**lines 290-298**).

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

We refactor this to use TensorFlow 2.x API by:

- Updated parameter signature to include `training` parameter

- Improved type annotations for better code safety

- Updated `tf.split()` to use `axis` parameter instead of positional argument

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

For **weight loading**, oo maintain compatibility with the original model weights, we implement a method to load weights from `NumPy` files:

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

#### 4. Stacked mLSTM Cell Implementation

In the original implementation, `mLSTMCellStackNPY` (**lines 301-390**) creates a stack of mLSTM cells with residual connections and dropout support. We refactor this to work with our new cell architecture and the TensorFlow 2.x API.

The stacked mLSTM cell creates a vertical stack of individual mLSTM cells where:

- Each cell processes the output from the previous cell
- State information is maintained separately for each layer
- Optional residual connections allow information to skip layers
- Dropout can be applied between layers during training

This completes our cell architecture implementation, making us ready to move to the model level in Entry 3.


### Key TensorFlow API Changes

| Feature | TensorFlow 1.x | TensorFlow 2.x | Impact |
|---------|---------------|----------------|--------|
| **Weight Creation** | `tf.get_variable("wx", initializer=wx_init)` | `self.add_weight(name="wx", shape=[...], initializer=...)` | Requires explicit shape specification |
| **Variable Scope** | `with tf.variable_scope(self._scope):` | Layer-based hierarchy with `name` parameter | Simplified variable organization |
| **Weight Normalization** | `tf.nn.l2_normalize(wx, dim=0)` | `tf.nn.l2_normalize(wx, axis=0)` | Parameter name change |
| **Tensor Splitting** | `tf.split(z, 4, 1)` | `tf.split(z, 4, axis=1)` | More explicit parameter naming |
| **Class Inheritance** | `tf.nn.rnn_cell.RNNCell` | `tf.keras.layers.Layer` | Aligns with Keras API |
| **Build Pattern** | Variables created in `call()` | Variables created in `build()` | Better separation of initialization and computation |
| **State Management** | `zero_state(batch_size, dtype)` | `get_initial_state(inputs, batch_size, dtype)` | More flexible state initialization |
| **Training Mode** | Not explicitly supported | `training` parameter in `call()` | Supports training-specific behaviors |
| **Initialization Seeding** | Global via `tf.set_random_seed()` | Per-operation via initializer seeds | Better control over randomness |
| **Parameter Naming** | Often used positional parameters | Requires named parameters | More explicit, less error-prone |
| **Categorical Sampling** | `tf.distributions.Categorical()` | `tf.random.categorical()` | API reorganization |


### Explanation

The multiplicative LSTM (mLSTM) combines the benefits of LSTM cells with multiplicative RNN dynamics, allowing for stronger expressivity in sequence modeling. Here's how it works:

1. Standard LSTM Components:

   - Cell state (c) - Long-term memory

   - Hidden state (h) - Current output
   - Gates: input (i), forget (f), output (o), and cell update (u)

2. Multiplicative Enhancement:

   - Before applying the standard LSTM computation, mLSTM introduces a multiplicative interaction between the input and previous hidden state
   - This creates a tensor product that allows for stronger, non-linear dependencies

3. Core Computation:

   - Multiplicative interaction: `m = (inputs · wmx) * (h_prev · wmh)`
   - Combined input: `z = (inputs · wx) + (m · wh) + b`
   - Gate activation: Split z into i, f, o, u and apply activation functions
   - Cell update: `c = f * c_prev + i * u`
   - Hidden update: `h = o * tanh(c)`

This architecture is particularly effective for protein sequences because:

- The multiplicative interaction captures complex dependencies between amino acids
- The cell state maintains information over long sequences
- The architecture can model diverse structural and functional patterns in **proteins**

### Explanation

[🔝 Back to Table of Contents](#toc)

### mLSTM Cell Architecture

The multiplicative Long Short-Term Memory (mLSTM) cell is a powerful variation of the standard LSTM that introduces multiplicative interactions. Let's explain how it works:

#### Basic Components

1. **Input (x)**: The current input to the cell
2. **Hidden State (h)**: The previous output of the cell
3. **Cell State (c)**: The internal memory of the cell


<a id="entry-3"></a>
## Entry 3: Refactor UniRep 64 Unit Model
