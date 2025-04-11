#!/bin/bash
#SBATCH --job-name=query_expand
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --constraint="vram32"

MODEL="deepseek"
mkdir -p results/cosqa_queries_expanded_$MODEL logs
INPUT_CSV="data/cosqa_queries/part_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV="results/cosqa_queries_expanded_$MODEL/result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "Input: $INPUT_CSV"
echo "Output: $OUTPUT_CSV"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

python query_expansion2.py $MODEL $INPUT_CSV $OUTPUT_CSV
