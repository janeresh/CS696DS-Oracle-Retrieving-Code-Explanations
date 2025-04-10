# RTC Validation

This directory provides utilities for validating the **quality of generated code explanations** using code cleaning, LLM-based generation, and multiple code evaluation metrics (CodeBLEU, CodeBERTScore, Structural metrics).

## Directory Structure

```
validation/RTC/
├── code_cleaning/                 
├── code_generation/              
├── evaluation/                   
├── scripts/                      
├── round_trip_check_code.ipynb   
├── validate_requirements.txt     
```

## Environment Setup

Install dependencies:

```bash
pip install -r validate_requirements.txt
```

## Code Cleaning

### General Cleaning:
Run this command for general whitespace, docstring, and formatting cleanup.
```bash
python code_cleaning/clean_code.py <input_csv> <output_csv> <model_name>
```

### In-line Comment Cleaning:
Run this command for removing inline and block comments from code
```bash
python code_cleaning/clean_code_remove_comments.py <input_csv> <output_csv> <model_name>
```

## Code Generation (LLM-based)
Generate code explanations using specific models such as Deepseek or Granite.

```bash
python code_generation/main.py <input_csv> <output_csv> <model_name> <exps_no> <number_backward_passes>
```

### Run as SBatch Job
```bash
sbatch code_generation/rtc_sbatch.sh <input_csv> <output_csv> <model_name> <exps_no> <number_backward_passes>
```

### Run as Multiple SBatch Jobs
**Note:** Modify the input CSV, output CSV, model name, number of explanations and number of codes to be generated(backward passes).
```bash
sh code_generation/submit_jobs.sh 
```

## Code Generation (LLM-based)
Generate parallelized batch inference using specific models such as Deepseek or Granite.

```bash
python code_generation/codegenerator_parallel.py <input_csv> <output_csv> <model_name> <exps_no> <number_backward_passes>
```

### Run as SBatch Job (Array Job)
```bash
sbatch code_generation/parallel_run.sh 
```

## Evaluation

### CodeBERT Metrics:
Evaluates similarity using CodeBERTScore

```bash
python evaluation/evaluate_codebertmetrics.py <input_csv> <output_csv> <model_name>
```

### CodeBLEU Metrics:
Evaluates similarity using CodeBLEU
```bash
python evaluation/evaluate_codebleumetrics.py <input_csv> <output_csv> <model_name>
```

### Structural Metrics:
Computes structural similarity using Tree-Sitter

```bash
python evaluation/structural_tree_code_metrics.py <input_csv> <output_csv> <model_name>
```

### Run as SBatch job 
Metric names can be bert, bleu or struct
```bash
python evaluation/evaluate.sh
```
