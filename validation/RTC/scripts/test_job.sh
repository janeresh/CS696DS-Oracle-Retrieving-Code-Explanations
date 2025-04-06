#!/bin/bash

# ✅ Define input CSV files
input_file="/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/code_generation/deepseek_cleaned_code_results_dir/split_part_0_results_results.csv"
output_file="/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/codebertsample.csv"
model="deepseek"
NUM_BACKWARD_PASSES="3"
COL_NO="4"

sbatch rtc_sbatch.sh "$input_file" "$output_file" "$model" "$NUM_BACKWARD_PASSES" "$COL_NO"

echo " Submitted $model job for: $input_file → $output_file"
