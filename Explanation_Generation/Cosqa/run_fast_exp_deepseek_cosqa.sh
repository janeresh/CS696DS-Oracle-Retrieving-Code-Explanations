#!/bin/bash
#SBATCH -A pi_wenlongzhao_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH -t 48:00:00
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --constraint=vram40
#SBATCH --mem-per-gpu=25G
#SBATCH -o slurm-%j.out 
#SBATCH --mail-user=anamikaghosh@umass.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --error=slurm-%j.err


module load conda/latest
module load cuda/12.6
conda activate gpu-env

python gen_exp_fast_deepseek_cosqa.py