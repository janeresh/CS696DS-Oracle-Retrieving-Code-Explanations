#!/bin/bash

# Define input CSV files
MODEL="deepseek"
INPUT_DIR="/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/code_generation/${MODEL}_cleaned_code_results_dir/"
OUTPUT_DIR="/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/metrics/${MODEL}_codebert_metrics_results_dir/"

NUM_BACKWARD_PASSES="3"
COL_NO="4"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# List all CSV files in the input directory
for input_file in "$INPUT_DIR"/*.csv; do
    # Extract filename without extension
    filename=$(basename -- "$input_file")
    filename_no_ext="${filename%.*}"

    # Define corresponding output file
    output_file="$OUTPUT_DIR/${filename_no_ext}_results.csv"

    # Submit SLURM job with input & output file arguments
    sbatch rtc_sbatch.sh "$input_file" "$output_file" "$MODEL" "$NUM_BACKWARD_PASSES" "$COL_NO"
    
    echo " Submitted $MODEL job for: $input_file → $output_file with BACKWARD_PASSES = $NUM_BACKWARD_PASSES and COL_NO = $COL_NO" 
done
