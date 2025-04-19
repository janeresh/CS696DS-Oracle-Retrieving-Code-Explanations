import sys
import pandas as pd
import os
from query_to_expansion_generation import QueryExpander
import logging
from expansion_to_code_generation import CodeGeneration
from tqdm import tqdm
from code_to_explanation_generation import CodeExplainer
import torch
import gc

def row_count(file):
    return sum(1 for _ in open(file)) - 1 if os.path.exists(file) else 0

MODEL = sys.argv[1]
INPUT_CSV = sys.argv[2]
OUTPUT_CSV = sys.argv[3]
CODE_MODEL="Codellama"
batch_size = 4
num_code_passes=5

expansion_stage_1_csv = OUTPUT_CSV.replace(".csv", ".exp1.csv")
codegen_stage_2_csv = OUTPUT_CSV.replace(".csv", "_codegen2.csv")
codeexplain_stage_3_csv = OUTPUT_CSV.replace(".csv", "_codeexp3.csv")

MODEL_PATHS = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}
CODE_MODEL_PATHS={"Codellama":"/datasets/ai/codellama/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/22cb240e0292b0b5ab4c17ccd97aa3a2f799cbed"}

logging.basicConfig(stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()

# ---------------- Load Input ----------------
df = pd.read_csv(INPUT_CSV)

# ---------------- Stage 1: Query Expansion ----------------
expander = QueryExpander(MODEL_PATHS, expansion_stage_1_csv)
expander.generate_expansions(MODEL, INPUT_CSV, query_batch_size=batch_size)
logger.info(f"[{MODEL}] Query Verbose complete → {codegen_stage_2_csv}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

df_ckpt1 = pd.read_csv(expansion_stage_1_csv)

# ---------------- Stage 2: Code Generation ----------------
generated_so_far = row_count(codegen_stage_2_csv)
total_rows = len(df_ckpt1)
logger.info(f"Loaded {total_rows} rows from {expansion_stage_1_csv}")
if generated_so_far >= total_rows:
    logger.info(f"[{CODE_MODEL}] Code generation already completed. Skipping Stage 2.")
else:
    logger.info(f"[{CODE_MODEL}] Resuming code generation from row {generated_so_far}")
    codegenerator = CodeGeneration(
        model_paths=CODE_MODEL_PATHS, 
        checkpoint_csv=codegen_stage_2_csv
    )
    codegenerator.explanation_to_code(CODE_MODEL, expansion_stage_1_csv, expansion_num=num_code_passes, query_batch_size=batch_size)
    logger.info(f"[{CODE_MODEL}] Code Generation complete → {codegen_stage_2_csv}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

df_ckpt2 = pd.read_csv(codegen_stage_2_csv)

# ---------------- Stage 3: Code Explanation ----------------
generated_so_far = row_count(codeexplain_stage_3_csv)
total_rows = len(df_ckpt2)
logger.info(f"Loaded {total_rows} rows from {codegen_stage_2_csv}")
if generated_so_far >= total_rows:
    logger.info(f"[{MODEL}] Code explanations already completed. Skipping Stage 3.")
else:
    logger.info(f"[{MODEL}] Resuming code explanation from row {generated_so_far}")
    code_explainer = CodeExplainer(MODEL_PATHS, codeexplain_stage_3_csv)
    code_explainer.generate_explanations(MODEL, codegen_stage_2_csv, query_batch_size=batch_size)
    logger.info(f"[{MODEL}] Code Explanation complete → {codeexplain_stage_3_csv}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
