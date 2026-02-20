###
### Convert protein sequences into numerical embeddings/representation that capture protein features
###

##### imports #####
import argparse, os, sys, h5py
import tensorflow as tf
import numpy as np
from Bio import SeqIO
from datetime import datetime

# Set seeds for reproducibility
tf.set_random_seed(42)
np.random.seed(42)

# Import the mLSTM babbler model
#from unirep import babbler64 as babbler # 64-unit version
# from unirep import babbler256 as babbler
# from unirep import babbler1900 as babbler
from unirep import babbler64, babbler256, babbler1900

def create_parser():
    """
    Set up command line arguments parsing using `argparse` library:
        - inpfasta: Input FASTA file with protein sequences
        - model: Path to the pre-trained mLSTM model weights
        - outrep: Path to save the output embeddings (.h5 format)
        --verbose: Optional flag to enable detailed debug output
    """
    parser = argparse.ArgumentParser(
        description="Representation of amino acid sequences by an mLSTM model"
    )
    parser.add_argument(
        "inpfasta",
        type=str,
        help="Path to the input fasta file"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Path to the mLSTM model",
    )
    parser.add_argument(
        "outrep",
        type=str,
        help="Path to the output sequence embeddings saved in hdf5 format (.h5)",
    )
    # add verbose flags for control debug output level
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output"
    )

    return parser


def log_message(message, verbose=False):
    """
    Helper function to print timestamped log messages
    - message: the message to log.
    - verbose: If True, the message is treated as a debug message.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if verbose:
        print(f"[DEBUG {timestamp}] {message}")
    else:
        print(f"[INFO {timestamp}] {message}")

def count_sequences(fasta_file):
    """
    Count total number of sequences in FASTA files
    - fasta_file: Path to the FASTA file.
    - Returns: the total number of sequences in the file.
    """
    count = 0
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'): # FASTA sequence headers start with '>'
                count += 1
    return count


def run(args):
    """
    Main function to process the FASTA file and generate embeddings.
    - ags: Parsed command line arguments.
    """
    print("Extract Argumments...")
    seqfl = args.inpfasta # Input  FASTA file path
    model_weights = args.model # Model weights path
    outfl = args.outrep # Output file path
    verbose = args.verbose # Verbose option for detailed logging
    if "64" in model_weights:
        babbler = babbler64
    elif "256" in model_weights:
        babbler = babbler256
    else:
        babbler = babbler1900
    # check input files exist
    if not os.path.exists(seqfl):
        raise FileNotFoundError(f"Input FASTA file not found: {seqfl}")
    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights directory not found: {model_weights}")

    # Remove output file if it already exists
    if os.path.exists(outfl):
        log_message(f"Removing existing output file: {outfl}", verbose)
        os.remove(outfl)

    log_message("Initializing mLSTM model...", verbose)
    ## batch_size doesn't affect babbles because no training here
    b = babbler(batch_size=128, model_path=model_weights)
    log_message("Model initialized successfully", verbose)

    # Count the total number of sequences in the FASTA file
    total_sequences = count_sequences(seqfl)
    log_message(f"Found {total_sequences} sequences to process", verbose)

    # Start Timing the processing
    start_time = datetime.now()

    # Track the number of sequences processed and errors encountered
    processed = 0
    errors = 0

    # Parse the FASTA file
    ##  SeqIO.parse returns an iterator of SeqRecord objects
    ##  Each SeqRecord represents one sequence from the FASTA file
    ##  Each SeqRecord object has the following relevant attributes:
    ##      - .description: full description line from the FASTA file
    ##      - .seq: the actual sequence
    ##      - .name: the sequence name
    fasta_sequences = SeqIO.parse(open(seqfl), 'fasta')

    # Process each sequence
    for i, fasta in enumerate(fasta_sequences):
        try:
            # get sequence description
            id_line = fasta.description

            # get target ID
            targetID = id_line.split()[0]

            # get the amino acid sequence as a string
            aa_seq = str(fasta.seq)

            # Log progress every 10 sequences or if verbose mode is enabled
            if verbose or (i + 1) % 10 == 0:
                progress = (i+1) / total_sequences * 100 # Calculate progress percentage
                elapsed_time = datetime.now() - start_time # Calcualte elapsed time
                avge_time_per_seq = elapsed_time / (i+1) # Calculate average time per sequence
                remaining_seqs = total_sequences - (i+1) # Calculate remaining sequences
                est_remaining_time = remaining_seqs * avge_time_per_seq # estimate remaining time

                log_message(
                    f"Processing {i + 1}/{total_sequences} ({progress:.1f}%) - {targetID}\n" +
                    f"  Elapsed: {elapsed_time.total_seconds():.1f}s, Estimated remaining: {est_remaining_time.total_seconds():.1f}s",
                    verbose
                )

            # get sequence representation from model
            ## tensorflow reduce_mean or reduce_sum is not deterministic
            avg_hidden, final_hidden, final_cell = b.get_rep(aa_seq)

            # Save the representation to HDF5 file
            with h5py.File(outfl, "a") as hf_array:
                # Save representation to HDF5 file
                hf_array.create_dataset(targetID, data = avg_hidden)

            # Increment the processed sequence count
            processed += 1

        except Exception as e:
            # Log any errors encountered during processing
            errors += 1
            log_message(f"Error processing sequence {targetID}: {str(e)}", True)
            continue

    # calcualte and print elpased time
    end_time = datetime.now()
    total_time = end_time - start_time

    # Log completion and statistics
    log_message("\nProcessing complete!", verbose)
    log_message(f"Total time: {total_time.total_seconds():.1f} seconds", verbose)
    log_message(f"Average time per sequence: {total_time.total_seconds() / total_sequences:.3f} seconds", verbose)
    log_message(f"Sequences processed successfully: {processed}", verbose)

    # Log any errors encountered
    if errors > 0:
        log_message(f"Sequences with errors: {errors}", True)

    # Log the size of the output file
    if os.path.exists(outfl):
        file_size = os.path.getsize(outfl)
        log_message(f"Output file size: {file_size / 1024 / 1024:.1f} MB", verbose)
    else:
        log_message("Warning: Output file was not created!", True)
    print (f"Time spent = {total_time.total_seconds()} seconds")

def main():
    """
    Entry point of the script
    - Parse command line arguments and runs the main processing function
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
