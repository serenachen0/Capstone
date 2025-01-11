# SeqRep

This project is to improve reproducibility and scalability of an existing mLSTM model for protein sequence representation. This mLSTM model is named UniRep, developed by Surojit Biswas *et al.* in George Church Lab at Harvard. 

To start, we will use a protein sequence that degrades Nylon as an example. The sequence is provided in examples/inputs/5y0m.fasta. The mLSTM model weights can be found at https://github.com/churchlab/UniRep.

Three python scripts are provided:

1. seq2rep.py and 2. unirep.py— convert protein sequences to mLSTM representations. unirep.py was adapted from https://github.com/churchlab/UniRep/blob/master/unirep.py. seq2rep.py calls unirep.py.

    python seq2rep.py \<input protein sequence file> \<mLSTM model weights> \<output sequence represenation file>

    Example: python seq2rep.py examples/inputs/5y0m.fasta <path_to_the_model_weigths> examples/outputs/5y0m_round1.h5
   
    Requirements: numpy 1.15.4, tensorflow 1.3.0, biopython

3. reps-compare.py— compare sequence representations in two output sequence representation files

    python reps-compare.py <sequence represenation file #1> <sequence represenation file #2>

    Example: python reps-compare.py examples/outputs/5y0m_round1.h5 examples/outputs/5y0m_round2.h5

Currently, the sequence to mLSTM representation conversion is non-deterministic and quite slow. The three example outputs, examples/outputs/5y0m_round[1/2/3].h5, were computed using the pre-trained 1900-unit model (1900_weights) on the same example protein seqeuence (examples/inputs/5y0m.fasta), but the results are not identical. Moreover, it took ~1 second to convert one sequence with 6 Intel Xeon CPUs and 1 Nvidia V100 GPU. The goal of this project is to investigate the robustness and throughput of this model and to modify the model so that it is deterministic and scalable. 
