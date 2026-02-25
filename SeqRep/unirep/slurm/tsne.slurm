#!/bin/bash
#SBATCH --job-name=tsne_plot
#SBATCH --output=plot_%j.out
#SBATCH --error=plot_%j.err
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate seqrep_gpu

# plotting tsne for 3 outputs
python ../../scripts/plot_tsne.py --run1 ../../data/outputs/runs/random_100_run1_1900.h5 --run2 ../../data/outputs/runs/random_100_run3_1900.h5 --run3 ../../data/outputs/runs/random_100_run4_1900.h5 --output ../../data/outputs/plot/tsne_mlstm_100_3.png --title "mLSTM (TF 1.15.0)"
