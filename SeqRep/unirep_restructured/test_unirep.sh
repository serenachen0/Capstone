#!/bin/bash
#SBATCH --job-name=test_unirep
#SBATCH --output=test_unirep_%j.log
#SBATCH --error=test_unirep_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


echo "Running sequence tests..."
python -m tests.test_sequence


echo "Running cell tests..."
python -m tests.test_cells


echo "Running model tests..."
python -m tests.test_model


echo "All tests completed!"
