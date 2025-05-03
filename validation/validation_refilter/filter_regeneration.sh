#!/bin/bash
#SBATCH --job-name=Filter
#SBATCH --output=logs/regen_%A_%a.out
#SBATCH --error=logs/regen_%A_%a.err
#SBATCH --array=0-1
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1  
#SBATCH --mem=80G  
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint="vram40"
#SBATCH --export=ALL,TORCH_CUDA_ALLOC_CONF=max_split_size_mb:10000,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/11.8
source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate

MODEL="granite"
dataset="csn_test"
mkdir -p results/${dataset}_new_${MODEL} logs
INPUT_CSV="../RTC/results/${dataset}_new_${MODEL}/result_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV="results/${dataset}_new_${MODEL}/result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

# Run Python script  structural_tree_code_metrics evaluate_codebleumetrics.py 
cd /work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/validation_refilter/
python refilter_pipeline.py $INPUT_CSV $OUTPUT_CSV $MODEL 