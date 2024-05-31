#!/bin/bash
#SBATCH --job-name=Extract_Metadata
#SBATCH --output=Extract_Metadata.out
#SBATCH --error=Extract_Metadata.err
#SBATCH --gres=gpu:1 # Request one GPU
#SBATCH --partition=gpu # Specify the GPU partition
python ./datasets/SelfDataset/extract_metadata.py ./datasets/SelfDataset/Videos 1

