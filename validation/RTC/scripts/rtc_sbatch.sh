#!/bin/bash
#SBATCH --job-name=RTC
#SBATCH --output=/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/logs/rtc_valid_%j.out
#SBATCH --error=/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/logs/rtc_valid_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  
#SBATCH --mem=64G  
#SBATCH --time=12:00:00
#SBATCH --constraint="vram23" 
#SBATCH --export=ALL,TORCH_CUDA_ALLOC_CONF="max_split_size_mb:10000"

module load cuda/11.8
source /work/pi_wenlongzhao_umass_edu/27/.venv/bin/activate
nvidia-smi
INPUT_FILE=$1
OUTPUT_FILE=$2
MODEL=$3
NUM_BACKWARD_PASSES=$4
COL_NO=$5
echo "$MODEL model, NUM_BACKWARD_PASSES = $NUM_BACKWARD_PASSES, COL_NO= $COL_NO"

# Run Python script  structural_tree_code_metrics
cd /work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/
python /work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/evaluate_codebertmetrics.py $INPUT_FILE $OUTPUT_FILE $MODEL $COL_NO $NUM_BACKWARD_PASSES
