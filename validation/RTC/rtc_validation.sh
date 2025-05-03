#!/bin/bash
#SBATCH --job-name=RTC
#SBATCH --output=logs/rtc_%A_%a.out
#SBATCH --error=logs/rtc_%A_%a.err
#SBATCH --array=0-2
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1  
#SBATCH --mem=64G  
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --constraint="vram40" 
#SBATCH --export=ALL,TORCH_CUDA_ALLOC_CONF="max_split_size_mb:10000"

module load cuda/11.8
source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate

MODEL="deepseek"
dataset="csn"
mkdir -p results/${dataset}_${MODEL}_test logs
INPUT_CSV="data/${dataset}_${MODEL}_test_explanations/part_${SLURM_ARRAY_TASK_ID}.csv"
OUTPUT_CSV="results/${dataset}_${MODEL}_test/result_${SLURM_ARRAY_TASK_ID}.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID} on host $(hostname)"

if [ ! -f "$INPUT_CSV" ]; then
    echo "Missing input file: $INPUT_CSV"
    exit 1
fi

# Run Python script  structural_tree_code_metrics evaluate_codebleumetrics.py 
cd /work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/
python validation_main.py $INPUT_CSV $OUTPUT_CSV $MODEL 