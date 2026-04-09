"""
seq2rep_v2.py - Convert protein sequences to mLSTM representations
==================================================================
Rewritten for TensorFlow 2.x with deterministic operations.
Usage:
    python seq2rep_v2.py <input.fasta> <model_weights_path> <output.h5>
Example:
    python seq2rep_v2.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/5y0m_round1.h5
"""
import os
import sys
import argparse
import random
import time
import h5py
import numpy as np
import tensorflow as tf
from Bio import SeqIO
from multiprocessing import Pool

# ============================================================
# CRITICAL: Set determinism BEFORE any TensorFlow operations
# ============================================================
def set_determinism(seed: int = 42) -> None:
    """Set all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
        print(f"[INFO] Deterministic mode enabled (seed={seed})")
    except AttributeError:
        print(f"[WARN] enable_op_determinism not available, using env variables")


from config.model_config import UniRepModelConfig
from models.unirep64 import UniRep64
from models.unirep256 import UniRep256
from models.unirep1900 import UniRep1900

# ============================================================
# ARGUMENT PARSER
# ============================================================
def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert protein sequences to mLSTM representations (deterministic)"
    )
    parser.add_argument("inpfasta", type=str, help="Path to the input FASTA file")
    parser.add_argument("model", type=str, help="Path to the mLSTM model weights directory")
    parser.add_argument("outrep", type=str, help="Path to the output HDF5 file (.h5)")
    parser.add_argument(
        "--units",
        type=int,
        choices=[64, 256, 1900],
        default=None,
        help="Model size: 64, 256, or 1900 units (default: auto-detect)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-type",
        type=str,
        choices=["avg", "final_hidden", "final_cell", "all"],
        default="avg",
        help="Which representation to save (default: avg)"
    )
    return parser

# ============================================================
# MODEL LOADING
# ============================================================
def detect_model_size(model_path: str) -> int:
    """
    Auto-detect model size from weight files.
    Checks the shape of the output projection weights to determine model size.
    Weight files use TF 1.x-style naming with ':0' suffix (e.g. fully_connected_weights:0.npy).
    """
    weights_file = os.path.join(model_path, "fully_connected_weights:0.npy")
    if os.path.exists(weights_file):
        weights = np.load(weights_file)
        units = weights.shape[0]
        print(f"[INFO] Auto-detected model size: {units} units")
        return units

    # Fall back to folder name
    folder_name = os.path.basename(model_path.rstrip('/'))
    for size in [64, 256, 1900]:
        if str(size) in folder_name:
            print(f"[INFO] Detected model size from folder name: {size} units")
            return size

    print(f"[WARN] Could not detect model size, defaulting to 1900")
    return 1900


def load_model(model_path: str, units: int, seed: int):
    """Load the appropriate UniRep model."""
    if units is None:
        units = detect_model_size(model_path)

    print(f"[INFO] Loading UniRep{units} from {model_path}")

    # All UniRep variants are 1-layer mLSTM in the original architecture
    config = UniRepModelConfig(
        rnn_size=units,
        embed_dim=10,
        num_layers=1,
        model_path=model_path,
        seed=seed
    )

    if units == 64:
        model = UniRep64(config=config)
    elif units == 256:
        model = UniRep256(config=config)
    else:
        model = UniRep1900(config=config)

    print(f"[INFO] Model loaded successfully")
    return model

# ============================================================
# SEQUENCE VALIDATION
# ============================================================
def validate_sequence(seq: str, max_len: int = 2000) -> tuple:
    """
    Validate a protein sequence.
    Returns: (is_valid, cleaned_sequence, error_message)
    """
    seq = seq.strip().upper()
    valid_aas = set("MRHKDESTNQCUGPAVIFYWLO")

    if len(seq) == 0:
        return False, seq, "Empty sequence"
    if len(seq) > max_len:
        return False, seq, f"Sequence too long ({len(seq)} > {max_len})"
    invalid_chars = set(seq) - valid_aas
    if invalid_chars:
        return False, seq, f"Invalid characters: {invalid_chars}"

    return True, seq, ""

def worker(args):
    seq, model_path, units, seed = args
    set_determinism(seed)
    model = load_model(model_path, units, seed)
    return model.get_representation(seq)

# ============================================================
# MAIN PROCESSING
# ============================================================
def run(args):
    """Main processing function."""
    seqfl = args.inpfasta
    model_path = args.model
    outfl = args.outrep
    units = args.units
    seed = args.seed
    output_type = args.output_type

    if not os.path.exists(seqfl):
        print(f"[ERROR] Input file not found: {seqfl}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model path not found: {model_path}")
        sys.exit(1)

    if os.path.exists(outfl):
        print(f"[INFO] Removing existing output file: {outfl}")
        os.remove(outfl)

    out_dir = os.path.dirname(outfl)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model = load_model(model_path, units, seed)

    print(f"\n[INFO] Processing sequences from: {seqfl}")
    print(f"[INFO] Output will be saved to: {outfl}")
    print(f"[INFO] Output type: {output_type}")
    print("-" * 50)

    start_time = time.time()
    processed = 0
    skipped = 0

    records = list(SeqIO.parse(seqfl, 'fasta'))

    valid_data = []
    ids = []

    for i, fasta in enumerate(records):
        id_line = fasta.description
        aa_seq = str(fasta.seq)
        target_id = id_line.split()[0]

        is_valid, cleaned_seq, error = validate_sequence(aa_seq)
        if not is_valid:
            print(f"  [{i+1}] SKIP  {target_id}: {error}")
            skipped += 1
            continue

        valid_data.append(cleaned_seq)
        ids.append(target_id)

    from multiprocessing import Pool

    args_list = [(seq, model_path, units, seed) for seq in valid_data]

    with Pool(20) as p:
        results = p.map(worker, args_list)

    with h5py.File(outfl, "w") as hf:
        for i, (target_id, res) in enumerate(zip(ids, results)):
            avg_hidden, final_hidden, final_cell = res

            if output_type == "avg":
                hf.create_dataset(target_id, data=avg_hidden)
            elif output_type == "final_hidden":
                hf.create_dataset(target_id, data=final_hidden)
            elif output_type == "final_cell":
                hf.create_dataset(target_id, data=final_cell)
            else:
                grp = hf.create_group(target_id)
                grp.create_dataset("avg_hidden", data=avg_hidden)
                grp.create_dataset("final_hidden", data=final_hidden)
                grp.create_dataset("final_cell", data=final_cell)

            processed += 1
            print(f"  [{i+1}] OK    {target_id}")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"[INFO] Processed: {processed} sequences")
    print(f"[INFO] Skipped:   {skipped} sequences")
    print(f"[INFO] Time:      {elapsed:.2f} seconds")
    if processed > 0:
        print(f"[INFO] Average:   {elapsed/processed:.3f} seconds per sequence")
    print(f"[INFO] Output saved to: {outfl}")


def main():
    parser = create_parser()
    args = parser.parse_args()
    # Set determinism with the specified seed BEFORE model is loaded
    set_determinism(args.seed)
    run(args)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    main()