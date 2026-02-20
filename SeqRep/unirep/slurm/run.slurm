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

cd /home/ko.hs/projects/Capstone/SeqRep

# check if GPU available
echo "=== GPU Check ==="
nvidia-smi

# check TensorFlow see if we use GPU
python -c "import tensorflow as tf; print('GPU available:', tf.test.is_gpu_available())"

# Run with 1900
echo "=== Starting 1900 weights ==="
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../1900_weights ../examples/outputs/runs/random_100_run5_1900.h5 
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../1900_weights ../examples/outputs/runs/random_100_run6_1900.h5

# compare 1900
python ../scripts/reps-compare.py ../examples/outputs/runs/random_100_run5_1900.h5 ../examples/outputs/runs/random_100_run6_1900.h5

# Run with 256
echo "=== Starting 256 weights ==="
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../256_weights ../examples/outputs/runs/random_100_run3_256.h5
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../256_weights ../examples/outputs/runs/random_100_run4_256.h5

# compare 256
python ../scripts/scripts/reps-compare.py ../examples/outputs/random_100_run3_256.h5 ../examples/outputs/runs/random_100_run4_256.h5

# Run with 64
echo "=== Starting 64 weights ==="
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../64_weights ../examples/outputs/runs/random_100_run3_64.h5
python ../scripts/seq2rep.py ../examples/inputs/random_100.fasta ../64_weights ../examples/outputs/runs/random_100_run4_64.h5

# compare 64
python ../scripts/reps-compare.py ../examples/outputs/runs/random_100_run3_64.h5 ../examples/outputs/runs/random_100_run4_64.h5