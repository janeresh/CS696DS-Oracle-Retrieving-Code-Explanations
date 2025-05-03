#!/bin/bash
#SBATCH --job-name=query_expand
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --constraint="vram40"

module load cuda/11.8
source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate
MODEL="granite"
mkdir -p results/cosqa/cosqa_queries_expanded_${MODEL}_temp_0.5_tfidf2 results/cosqa/cosqa_queries_expanded_${MODEL}_temp_0.5_mmr2 logs
INPUT_CSV="data/cosqa_queries/part_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV1="results/cosqa/cosqa_queries_expanded_${MODEL}_temp_0.5_tfidf2/result_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV2="results/cosqa/cosqa_queries_expanded_${MODEL}_temp_0.5_mmr2/result_${SLURM_ARRAY_TASK_ID}.csv"


echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"
echo "Input: $INPUT_CSV"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

python queryexp_main.py $MODEL $INPUT_CSV $OUTPUT_CSV1 $OUTPUT_CSV2
