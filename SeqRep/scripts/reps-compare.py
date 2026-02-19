#!/usr/bin/env python3
# compare sequence representations in two h5 files

import argparse, os, sys, h5py
import numpy as np

def create_parser():
    """
    Set up command line arguments parsing using `argparse` library:
    - rep1: Path to the first sequence embeddings file
    - rep2: Path to the second sequence embeddings file
    --verbose: Optional flag to enable detailed debug output
    """
    parser = argparse.ArgumentParser(
        description="Compare two sequence representations"
    )
    parser.add_argument(
        "rep1",
        type=str,
        help="Path to the first sequence embeddings file"
    )
    parser.add_argument(
        "rep2",
        type=str,
        help="Path to the second sequence embeddings file",
    )
    # Add verbose flag for control debug output level
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output"
    )
    # Add option to control how many positions to show
    parser.add_argument(
        "--max-pos",
        type=int,
        default=None,
        help="Maximum number of positions to show per sequence (default: show all)"
    )
    return parser

def log_message(message, verbose=False):
    """
    Helper function to print log messages.
    - message: The message to log.
    - verbose: If True, the message is treated as a debug message.
    """
    if verbose:
        print(f"[DEBUG] {message}")  # Print debug messages
    else:
        print(f"[INFO] {message}")  # Print regular info messages

def format_float(value):
    """
    Format float value to show all decimal places without scientific notation
    """
    return f"{value:.20f}".rstrip('0').rstrip('.')

def compare_arrays(arr1, arr2, max_pos=None):
    """
    Compare two arrays and return differences with full precision
    """
    comparisons = []
    for i in range(np.shape(arr1)[0]):
        val1 = arr1[i]
        val2 = arr2[i]
        comparisons.append((i, val1, val2))
        if max_pos is not None and len(comparisons) >= max_pos:
            break
    return comparisons

def run(args):
    """
    Main function to compare sequence representations in two HDF5 files.
    - args: Parsed command line arguments.
    """
    seq_rep1 = args.rep1 # Path to the first embeddings file
    seq_rep2 = args.rep2  # Path to the second embeddings file
    verbose = args.verbose  # Verbose option for detailed logging
    max_pos = args.max_pos
    
    # Load the embeddings files with error handling
    log_message("\nLoading embeddings files...", verbose)
    try:
        hfin_array1 = h5py.File(seq_rep1, 'r')
        log_message(f"Successfully loaded {seq_rep1}", verbose)
        hfin_array2 = h5py.File(seq_rep2, 'r')
        log_message(f"Successfully loaded {seq_rep2}", verbose)
    except Exception as e:
        log_message(f"Error loading files: {str(e)}", True)
        sys.exit(1)

    # Get the list of sequences (enzyme IDs) from the first file
    enzymes = list(hfin_array1.keys())
    total_enzymes = len(enzymes)
    log_message(f"Found {total_enzymes} sequences to compare", verbose)
    log_message(f"Enzymes: {enzymes}")

    # Track the number of differences found
    differences_found = 0
    identical_found = 0

    # Compare each sequence
    log_message("\nComparing arrays...", verbose)
    for idx, enz_id in enumerate(enzymes, 1):
        # logs progress every 10 sequences or when verbose mode is enabled
        if verbose or idx % 10 == 0:
            log_message(f"Progress: {idx}/{total_enzymes} sequences processed", verbose)

        # Get the embeddings for the current sequence
        arr1 = hfin_array1[enz_id][:]
        arr2 = hfin_array2[enz_id][:]

        print(f"\nSequence {enz_id}: ")

        comparisons = compare_arrays(arr1, arr2)

        is_identical = np.array_equal(arr1, arr2)
        if is_identical:
            identical_found += 1
            print("Sequences are identical")
        else:
            differences_found += 1
            print("Sequnce are different")

        for pos, val1, val2 in comparisons:
            print(f"Position {pos}:")
            print(f"  File 1: {format_float(val1)}")
            print(f"  File 2: {format_float(val2)}")
            if not is_identical:
                print(f"  Absolute difference: {format_float(abs(val1 - val2))}")

        if max_pos is not None and len(comparisons) >= max_pos:
            print(f"(Showing only first {max_pos} positions)")

    # Print the comparison summary
    log_message("\nComparison Summary:", verbose)
    log_message(f"Total sequences compared: {total_enzymes}", verbose)
    log_message(f"Sequences with differences: {differences_found}", verbose)
    log_message(f"Sequences identical: {identical_found}", verbose)
    log_message(f"Percentage different: {(differences_found / total_enzymes) * 100:.2f}%", verbose)

def main():
    """
    Entry point of the script.
    - Parses command line arguments and runs the main processing function.
    """
    parser = create_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        # Log any fatal errors and exit with a non-zero status
        log_message(f"Fatal error: {str(e)}", True)
        sys.exit(1)

if __name__ == "__main__":
    main()
