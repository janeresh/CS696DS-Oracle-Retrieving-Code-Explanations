import os
import sys
import gc
import torch
import pandas as pd
import logging
from tqdm import tqdm

from codegenerator_parallel import CodeGeneration
from clean_code import CodeCleaner
from metrics_calculation import CodeBERTScorer, CodeStructureScorer, CodeBLEUEvaluator

# ---------------- Logging Setup ----------------
logging.basicConfig(stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

# ---------------- CLI Args ----------------
input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]

# ---------------- Config ----------------
batch_size = 4
col_no = 5
num_backward_passes = 3

model_path_dict = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

generated_file = output_csv.replace(".csv", "_generated.csv")
cleaned_file = output_csv.replace(".csv", "_cleaned.csv")
scored_file = output_csv

def row_count(file):
    return sum(1 for _ in open(file)) - 1 if os.path.exists(file) else 0

# ---------------- Load Input ----------------
df = pd.read_csv(input_csv)
total_rows = len(df)
logger.info(f"Loaded {total_rows} rows from {input_csv}")

# Get explanation columns dynamically
col_prefix = f"explanation_{model}"
explanation_cols = sorted([col for col in df.columns if col.startswith(col_prefix)],
                          key=lambda x: int(x.rsplit("_", 1)[-1]))[:col_no]

# ---------------- Stage 1: Generation ----------------
generated_so_far = row_count(generated_file)
if generated_so_far >= total_rows:
    logger.info(f"[{model}] Generation already completed. Skipping Stage 1.")
else:
    logger.info(f"[{model}] Resuming generation from row {generated_so_far}")
    codegenerator = CodeGeneration(
        model_path=model_path_dict[model],
        model_name=model,
        batch_size=batch_size
    )

    for i in tqdm(range(generated_so_far, total_rows, batch_size), desc="Generating"):
        batch_df = df.iloc[i:i + batch_size].copy()
        batch_df = batch_df.drop(columns=["code", "remove_all_comments_issue"], errors="ignore")

        try:
            generated_rows = codegenerator.explanation_to_code(batch_df, explanation_cols, num_backward_passes)
        except RuntimeError as e:
            logger.error(f"CUDA OOM at batch {i}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

        pd.DataFrame(generated_rows).to_csv(
            generated_file,
            mode='a',
            index=False,
            header=not os.path.exists(generated_file)
        )
        del generated_rows
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"[{model}] Generation complete → {generated_file}")

# ---------------- Stage 2: Cleaning ----------------
cleaned_so_far = row_count(cleaned_file)
if cleaned_so_far >= total_rows:
    logger.info(f"[{model}] Cleaning already completed. Skipping Stage 2.")
else:
    logger.info(f"[{model}] Resuming cleaning from row {cleaned_so_far}")
    cleaner = CodeCleaner()
    raw_generated = pd.read_csv(generated_file, skiprows=range(1, cleaned_so_far + 1))

    for i in tqdm(range(0, len(raw_generated), batch_size), desc="Cleaning"):
        batch = raw_generated.iloc[i:i + batch_size].copy().reset_index(drop=True)
        for r in range(len(batch)):
            for d in range(col_no):
                for v in range(num_backward_passes):
                    col = f"generated_code_{d+1}_code{v+1}"
                    if pd.notna(batch.at[r, col]):
                        batch.at[r, col] = cleaner.clean_code(batch.at[r, col])
        batch.to_csv(cleaned_file, mode='a', index=False, header=not os.path.exists(cleaned_file))
        del batch
        gc.collect()

    logger.info(f"[{model}] Cleaning complete → {cleaned_file}")

# ---------------- Stage 3: Scoring ----------------
scored_so_far = row_count(scored_file)
if scored_so_far >= total_rows:
    logger.info(f"[{model}] Scoring already completed. Skipping Stage 3.")
else:
    logger.info(f"[{model}] Resuming scoring from row {scored_so_far}")
    cleaned_df = pd.read_csv(cleaned_file, skiprows=range(1, scored_so_far + 1))
    codebert = CodeBERTScorer(threshold=0.8)
    structure = CodeStructureScorer()
    codebleu = CodeBLEUEvaluator(script_path="../CodeBLEU/calc_code_bleu.py")

    for i in tqdm(range(0, len(cleaned_df), batch_size), desc="Scoring"):
        batch = cleaned_df.iloc[i:i + batch_size].copy()
        scored_batch = []
        for row in batch.itertuples(index=False):
            row_dict = row._asdict()
            row_dict = codebert.compute_metrics(row_dict, model, col_no, num_backward_passes)
            row_dict = structure.compute_structural_scores(row_dict, model, col_no, num_backward_passes)
            row_dict = codebleu.compute_bleu_metrics(row_dict, model, col_no, num_backward_passes)
            scored_batch.append(row_dict)

        pd.DataFrame(scored_batch).to_csv(
            scored_file, mode='a', index=False, header=not os.path.exists(scored_file)
        )
        del batch, scored_batch
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"[{model}] Scoring complete → {scored_file}")
