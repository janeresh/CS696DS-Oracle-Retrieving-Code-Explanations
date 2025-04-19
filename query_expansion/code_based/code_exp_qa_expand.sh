#!/bin/bash
#SBATCH --job-name=code_expand
#SBATCH --output=logs/code_%A_%a.out
#SBATCH --error=logs/code_%A_%a.err
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="vram32"

module load cuda/11.8
source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate
MODEL="granite"
dataset="csn/csn_test"
mkdir -p results/${dataset}_queries_expanded_${MODEL}_code_based logs
INPUT_CSV="../data/csn_test_queries/part_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV="results/${dataset}_queries_expanded_${MODEL}_code_based/result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "Input: $INPUT_CSV"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

python code_query_exp_main.py $MODEL $INPUT_CSV $OUTPUT_CSV
