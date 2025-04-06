# === codegenerator_parallel.py (Streaming Optimized) ===
import os
import csv
import json
import torch
import pandas as pd
from vllm import LLM, SamplingParams

CHECKPOINT_FILE = "progress.json"
OUTPUT_FILE = "generated_outputs.csv"
model = LLM(model=model_path, dtype="bfloat16", device=device, enforce_eager=False)
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1000)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_index": index}, f)

def generate_llm(desc, model_path, code_no):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    prompts = [
        f"Write only the Python function corresponding to the following description. "
        "Do not provide explanations, comments, markdown, parameter descriptions, or return values.\n\n"
        f"Description:\n{desc}\n\nPython Code:\n",

        f"Task: Implement the Python function as described below.\n\n"
        "Constraints:\n"
        "- Output ONLY valid Python code.\n"
        "- DO NOT include markdown.\n"
        "- DO NOT include comments or explanations.\n"
        "- Ensure the function signature and structure match the description.\n\n"
        f"Function Description:\n{desc}\n\nCode:\n",

        f"You must write only the Python function described below.\n"
        "Do not include explanations, markdown, or return values.\n"
        "Do not prefix your answer with ```python or anything else.\n\n"
        f"Description:\n{desc}\n\nPython Function:\n"
    ][:code_no]
    try:
        outputs = model.generate(prompts, sampling_params)
        output_iter = iter([clean_output(o.outputs[0].text) for o in outputs])
    except Exception as e:
        print(f"[ERROR] Batch failed: {e}")
        output_iter = iter(["ERROR"] * code_no)

    return output_iter

def clean_output(text, keyword="<think>"):
    index = text.rfind(keyword)
    return text[index + len(keyword):].strip() if index != -1 else text.strip()

def write_row(row_dict):
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def process_batch(batch, model, sampling_params, model_name, col_no=4, code_no=3):
    all_prompts = []
    row_prompt_counts = []

    for _, row_data in batch:
        row_prompts = []
        row_dict = {
            "corpus_id": row_data["corpus_id"],
            "query_id": row_data["query_id"],
            "Original_Code": row_data["code"]
        }
        for i in range(col_no):
            col = f"explanation_{model_name}_{i+1}"
            row_prompts.extend(generate_llm(row_data[col], code_no))
            key = f"Generated_Code_{model_name}_{i+1}_code{k+1}"
            row_dict[key] = next(output_iter)

        write_row(row_dict)
        save_checkpoint(row_index + 1)
        all_prompts.extend(row_prompts)
        row_prompt_counts.append(row_data)

    try:
        outputs = model.generate(all_prompts, sampling_params)
        output_iter = iter([clean_output(o.outputs[0].text) for o in outputs])
    except Exception as e:
        print(f"[ERROR] Batch failed: {e}")
        output_iter = iter(["ERROR"] * len(row_prompt_counts) * col_no * code_no)

    for row_index, row_data in zip((i for i, _ in batch), row_prompt_counts):
        row_dict = {
            "corpus_id": row_data["corpus_id"],
            "query_id": row_data["query_id"],
            "Original_Code": row_data["code"]
        }
        for j in range(col_no):
            for k in range(code_no):
                key = f"Generated_Code_{model_name}_{j+1}_code{k+1}"
                row_dict[key] = next(output_iter)

        write_row(row_dict)
        save_checkpoint(row_index + 1)

def model_worker(model_path, df, model_name, batch_size=4):
    

    start_index = load_checkpoint()
    batch = []

    for i in range(start_index, len(df)):
        row = df.iloc[i]
        row_data = {
            "corpus_id": row["corpus_id"],
            "query_id": row["query_id"],
            "code": row["code"],
            "descriptions": [row[f"explanation_{model_name}_{j}"] for j in range(1, 5)]
        }
        batch.append((i, row_data))

        if len(batch) == batch_size:
            process_batch(batch, model, sampling_params, model_name)
            batch.clear()

    if batch:
        process_batch(batch, model, sampling_params, model_name)

    print("Generation complete. Output written to", OUTPUT_FILE)
