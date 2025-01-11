# Convert sequences in a fasta file to mLSTM embeddings

##### Setup #####
import argparse, os, sys, h5py
import tensorflow as tf
import numpy as np
from Bio import SeqIO
import time

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

# Import the mLSTM babbler model
#from unirep import babbler64 as babbler
#from unirep import babbler256 as babbler
from unirep import babbler1900 as babbler

def create_parser():
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
    return parser

def run(args):
    seqfl = args.inpfasta
    model_weights = args.model
    outfl = args.outrep
    
    # generate sequence representation using the provided model and save the representation in outfl
    if os.path.exists(outfl):
        os.remove(outfl)
    # batch_size doesn't affect babbles because no training here
    b = babbler(batch_size=128, model_path=model_weights)

    start_time = time.time()
    fasta_sequences = SeqIO.parse(open(seqfl), 'fasta')
    for i, fasta in enumerate(fasta_sequences):
        id_line, aa_seq = fasta.description, str(fasta.seq)
        targetID = id_line.split()[0]
        print ("{} {}".format(i+1, targetID))
        avg_hidden, final_hidden, final_cell = b.get_rep(aa_seq) #tensorflow reduce_mean or reduce_sum is not deterministic
        hf_array = h5py.File(outfl, "a")
        hf_array.create_dataset(targetID, data = avg_hidden)
        hf_array.close()
    end_time = time.time()
    spent_time = end_time - start_time
    print ("Time spent = {} seconds".format(spent_time))

def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()