#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --account=<project>@v100                 # <<< EDIT: your project
#SBATCH --cpus-per-gpu 8
#SBATCH --partition=gpu_p13
#SBATCH --job-name=pcam_single_split
#SBATCH --output=slurm_output/pcam_%a.out
#SBATCH --error=slurm_output/pcam_%a.out

module purge
module load pytorch-gpu

params=$(awk -v  idx_param="${SLURM_ARRAY_TASK_ID}" 'NR==idx_param' configs_adapted_100seeds_5study.txt)

python pcam_deep_training_adapted.py $params