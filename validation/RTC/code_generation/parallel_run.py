import sys
import torch
import pandas as pd
import logging
import multiprocessing as mp
from codegenerator import CodeGeneration
import json
import csv
import os
import tqdm

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

CHECKPOINT_FILE = "progress.json"
OUTPUT_FILE = "output_codegen"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f).get("last_index", 0)
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({"last_index": index}, f)

def write_row(row_dict):
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def row_feeder(df, task_queue, start_idx, model_name):
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        task_queue.put((
            i, {
                "corpus_id": row["corpus_id"],
                "query_id": row["query_id"],
                "code": row["code"],
                "descriptions": [row[f"explanation_{model_name}_{j}"] for j in range(1, 5)]
            }
        ))
    task_queue.put(None)  # Sentinel to stop model worker

def model_worker(model_path, model_name, task_queue, batch_size, pbar):
    codegen = CodeGeneration(model_path)
    buffer = []

    while True:
        task = task_queue.get()
        if task is None:
            break
        buffer.append(task)

        if len(buffer) == batch_size:
            process_batch(buffer, codegen, model_name, pbar)
            buffer.clear()

    if buffer:
        process_batch(buffer, codegen, model_name, pbar)


def process_batch(task_batch, codegen, model_name, pbar):
    all_prompts = []
    row_metas = []

    for row_index, row_data in task_batch:
        prompts = []
        for desc in row_data["descriptions"]:
            prompts.extend(codegen.construct_variants(desc))
        all_prompts.extend(prompts)
        row_metas.append((row_index, row_data))

    try:
        outputs = codegen.model.generate(all_prompts, codegen.sampling_params)
        texts = [codegen.clean_output(o.outputs[0].text) for o in outputs]
    except Exception as e:
        logger.error(f"[ERROR] Batch generation failed: {e}")
        texts = ["ERROR"] * len(all_prompts)

    for i, (row_index, row_data) in enumerate(row_metas):
        row_texts = texts[i * 12:(i + 1) * 12]
        row_dict = {
            "corpus_id": row_data["corpus_id"],
            "query_id": row_data["query_id"],
            "Original_Code": row_data["code"]
        }

        for j in range(4):
            for k in range(3):
                col = f"Generated_Code_{model_name}_{j+1}_code{k+1}"
                row_dict[col] = row_texts[j * 3 + k]

        write_row(row_dict)
        save_checkpoint(row_index + 1)
        pbar.update(1)  

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    MODEL_NAME = sys.argv[3]

    model_path_dict = {
        "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
        "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
    }

    df = pd.read_csv(input_csv)
    start_index = load_checkpoint()

    task_queue = mp.Queue(maxsize=10)
    feeder = mp.Process(target=row_feeder, args=(df, task_queue, start_index, MODEL_NAME))
    with tqdm(total=len(df) - start_index, desc=f"Generating ({MODEL_NAME})") as pbar:
        model_proc = mp.Process(target=model_worker, args=(model_path_dict[MODEL_NAME], MODEL_NAME, task_queue, 4, pbar))
        feeder.start()
        model_proc.start()
        feeder.join()
        model_proc.join()


    # Rename and cleanup
    os.rename(OUTPUT_FILE, output_csv)
    logger.info(f"[✔] Final output saved to {output_csv}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint file deleted after successful completion.")
