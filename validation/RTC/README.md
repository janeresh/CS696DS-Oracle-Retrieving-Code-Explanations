# RTC Validation

This directory provides utilities for validating the quality of generated code from explanations using a combination of code cleaning, LLM-based generation, and multiple similarity metrics, including CodeBLEU, CodeBERTScore, and structural comparison.

## Directory Structure

```
validation/RTC/
├── clean_code.py                 
├── codegenerator_parallel.py              
├── metrics_calculation.py                   
├── rtc_validation.sh                      
├── round_trip_check_code.ipynb   
├── validate_requirements.txt     
```
## Prerequisites
- A compatible GPU (e.g., A100 / L40S with ≥32GB VRAM)
- A local model path (e.g., DeepSeek, Granite)
- Python 3.8+

## Environment Setup

Install all dependencies using:

```bash
pip install -r validate_requirements.txt
```

## Round Trip Correctness Execution

Run the full validation pipeline:
```bash
python validation_main.py <input_csv> <output_csv> <model_name>
```

This performs the following:

**1. Code Generation from Explanations**
Handled by `codegenerator_parallel.py` using LLM-based models.

**2. Code Cleaning**
Using `clean_code.py` to strip docstrings, inline comments, and update deprcdecated functions.

**3. Similarity Metrics Calculation**
Done using `metrics_calculation.py`, which includes: CodeBERTScorer, CodeStructureScorer and CodeBLEUEvaluator


## Running on Full Dataset
**1. Split the Input CSV for Parallel Jobs**
If your dataset is large, split it into smaller CSV shards (e.g., 5000 rows per file):
**Note:** Add the input_csv, output_csv and shard_no in the script.

```bash
python ../utils/split_csv.py 
```

**2. Run the Jobs in Parallel (SLURM Batch Job)**
Submit the job array using:

```bash
sbatch rtc_validation.sh 
```
Each job will process one split independently.

Note: Replace the input_csv, output_csv and model name parameters.

**3. Merge Output Files into a Single CSV**
After all shards are processed, merge the outputs into one file: 
**Note:** Add the input_csv and output_csv in the script.

```bash
python ../utils/merge_csv.py
```

