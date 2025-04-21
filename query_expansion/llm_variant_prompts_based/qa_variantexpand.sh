#!/bin/bash
#SBATCH --job-name=query_expand
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
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
parent="csn"
dataset="csn_test"
mkdir -p results/${parent}/${dataset}_queries_variants_expanded_${MODEL}_temp_0.5 logs
INPUT_CSV="../data/${dataset}_queries/part_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV="results/${parent}/${dataset}_queries_variants_expanded_${MODEL}_temp_0.5/result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "Input: $INPUT_CSV"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

python queryexp_main.py $MODEL $INPUT_CSV $OUTPUT_CSV
