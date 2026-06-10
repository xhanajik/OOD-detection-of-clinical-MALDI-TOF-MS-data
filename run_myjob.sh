#!/bin/bash

# Choose GPU ID (e.g., 0, 1, or 2) after checking with: watch -d nvidia-smi
export CUDA_VISIBLE_DEVICES=0

cd <project directory>

custom code: 
# Load basic tools
module load gcc/11.4.0
module load python/3.10.12
module load cuda/12.1.1
module load cudnn/8.9.5.29-12.1

# Load ML framework
module load pytorch/2.2.2  
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# Load ML utilities
module load faiss/1.7.4
module load umap-learn/0.5.5
module load scikit-learn/1.4.2
module load optuna/3.6.1
module load wandb/0.17.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a
module load Seaborn/0.13.2-gfbf-2023a
module load Biopython/1.83-foss-2023a
module load Optuna/3.5.0-foss-2023a
module load wandb/0.16.1
module load ETE/3.1.3-foss-2023a

# Run your main Python script
<project directory>/main.py --config_file <config_file.yml>