#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=setup_%j.out
#SBATCH --error=setup_%j.err
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc

# 建立環境
conda create -n seqrep_gpu python=3.7 cudatoolkit=10.0 cudnn=7.6 -c anaconda -y

# 啟動並安裝套件
source activate seqrep_gpu
pip install tensorflow-gpu==1.15.0 "numpy<2" biopython pandas h5py "protobuf<3.20"
