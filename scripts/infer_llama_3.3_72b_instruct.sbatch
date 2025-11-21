#!/bin/bash
#SBATCH --job-name=llama3.3-72B-infer
#SBATCH --partition=batch_bmgpu
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load conda (adjust if your system differs)
source ~/.bashrc
conda activate vllm_env

# Change to working directory
cd PRAGWORLD

# Run inference script
bash scripts/infer_llama_3.3_72b_instruct.sh synth_final synthetic_final.csv
