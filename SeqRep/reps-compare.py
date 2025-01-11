#!/usr/bin/env python3
# compare sequence representations in two h5 files

import argparse, os, sys, h5py
import numpy as np

def create_parser():
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
    return parser

def run(args):
    seq_rep1 = args.rep1
    seq_rep2 = args.rep2
    
    # load the embeddings
    hfin_array1 = h5py.File(seq_rep1, 'r')
    hfin_array2 = h5py.File(seq_rep2, 'r')

    # Comparing the arrays
    enzymes = list(hfin_array1.keys())
    for enz_id in enzymes:
        arr1 = hfin_array1[enz_id][:]
        arr2 = hfin_array2[enz_id][:]

        if np.array_equal(arr1, arr2):
            print("Equal")
        else:
            print("Not Equal")
            for i in range(np.shape(arr1)[0]):
                val1 = arr1[i]
                val2 = arr2[i]
                if not val1 == val2:
                    print ("{} {} {}".format(i, val1, val2))
                    
def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()