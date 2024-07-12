#!/bin/bash
#SBATCH --job-name="gcbf_test_simple"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=36G
#SBATCH --time=04:30:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python test.py --path /home/stud/li0/gcbf-pytorch/trained_model/SimpleCar/gcbf/seed0_20240708094412_agent=3 -n 20 --epi 40 --area-size 4 --no-video
