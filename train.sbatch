#!/bin/bash
#SBATCH --job-name="gcbf_train_simple_3"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=36G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-gcbf.out
#SBATCH --error=/storage/slurm/logs/slurm-gcbf.out
srun python train.py --algo gcbf --env SimpleCar -n 3 --steps 1000000 --log-path /home/stud/li0/gcbf-pytorch/trained_model
