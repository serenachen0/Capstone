#!/bin/bash
#SBATCH --job-name=seqrep
#SBATCH --output=seqrep_%j.out
#SBATCH --error=seqrep_%j.err
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# 啟動 conda 環境
source ~/.bashrc
conda activate seqrep_gpu

# 檢查 GPU 是否可用
echo "=== GPU Check ==="
nvidia-smi

# 檢查 TensorFlow 是否看到 GPU
python -c "import tensorflow as tf; print('GPU available:', tf.test.is_gpu_available())"

# 執行程式

# 跑多次比較結果
python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_1900_test1.h5
python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_1900_test2.h5
#python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_1900_test3.h5

# 比較結果
python reps-compare.py examples/outputs/gpu_1900_test1.h5 examples/outputs/gpu_1900_test2.h5
#python reps-compare.py examples/outputs/gpu_1900_test2.h5 examples/outputs/gpu_1900_test3.h5

python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_256_test1.h5
python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_256_test2.h5
#python seq2rep.py examples/inputs/5y0m.fasta ./1900_weights examples/outputs/gpu_256_test3.h5

# 比較結果
python reps-compare.py examples/outputs/gpu_256_test1.h5 examples/outputs/gpu_256_test2.h5
#python reps-compare.py examples/outputs/gpu_256_test2.h5 examples/outputs/gpu_256_test3.h5
