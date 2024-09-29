#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=cohface-effphys-regression

module load cuda/10.2
module load anaconda3/2022.05
source activate rppg-toolbox

# models
#python main.py --config_file configs/train_configs/SCAMPS_SCAMPS_SCAMPS_TSCAN_BASIC.yaml
python main.py --config_file configs/train_configs/COHFACE_COHFACE_COHFACE_EFFICIENTPHYS.yaml
#python main.py --config_file configs/train_configs/COHFACE_COHFACE_COHFACE_PHYSNET_BASIC.yaml
