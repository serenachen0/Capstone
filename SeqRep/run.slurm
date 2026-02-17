#!/bin/bash
#SBATCH --job-name=seqrep
#SBATCH --output=seqrep_%j.out
#SBATCH --error=seqrep_%j.err
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# conda env
source ~/.bashrc
conda activate seqrep_gpu

# check if GPU available
echo "=== GPU Check ==="
nvidia-smi

# check TensorFlow see if we use GPU
python -c "import tensorflow as tf; print('GPU available:', tf.test.is_gpu_available())"


# Run with 1900
echo "=== Starting 1900 weights ==="
python seq2rep.py examples/inputs/random_100.fasta ./1900_weights examples/outputs/random_100_run1_1900.h5
python seq2rep.py examples/inputs/random_100.fasta ./1900_weights examples/outputs/random_100_run2_1900.h5

# compare 1900
python reps-compare.py examples/outputs/random_100_run1_1900.h5 examples/outputs/random_100_run2_1900.h5

# Run with 256
echo "=== Starting 256 weights ==="
python seq2rep.py examples/inputs/random_100.fasta ./256_weights examples/outputs/random_100_run1_256.h5
python seq2rep.py examples/inputs/random_100.fasta ./256_weights examples/outputs/random_100_run2_256.h5

# compare 256
python reps-compare.py examples/outputs/random_100_run1_256.h5 examples/outputs/random_100_run2_256.h5

# Run with 64
echo "=== Starting 64 weights ==="
python seq2rep.py examples/inputs/random_100.fasta ./64_weights examples/outputs/random_100_run1_64.h5
python seq2rep.py examples/inputs/random_100.fasta ./64_weights examples/outputs/random_100_run2_64.h5

# compare 64
python reps-compare.py examples/outputs/random_100_run1_64.h5 examples/outputs/random_100_run2_64.h5
