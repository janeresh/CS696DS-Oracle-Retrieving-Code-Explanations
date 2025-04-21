import torch
import json
from tqdm import tqdm
import os
import gc
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from vllm import LLM, SamplingParams, LLMEngine
from datasets import load_dataset
import csv
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set environment variable to help with memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean_output(text,keyword):
    # keyword = "Answer:"
    index = text.rfind(keyword)  # Find the last occurrence of "Answer:"
    if index != -1:
        return text[index + len(keyword):].strip()  
    return text

class ExplanationGeneratorLama:
    def __init__(self, model_name, max_new_tokens=500):
        self.max_new_tokens = max_new_tokens
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_new_tokens
        )
        self.llm = LLM(model=model_name, dtype="half")  # GPU half precision

    def generate_explanations_batch(self, entries):
        prompts = []
        for entry in entries:
            code = entry['cleaned_code']
            prompt_templates = [
            
            f"Code snippet: {code}\n"
            "Instruction: Provide a concise explanation of what the above code mean. "
            "Generate strictly less than 100 words in total. Please give the output just as text only. Do not return anything else.\n"
            "Answer: \n"
            , 

            
            f"Code snippet: {code}\n"
            "Instruction: Provide a detailed line-by-line explanation of this code snippet, describing the purpose and functionality of each statement, function, and control structure. "
            "Please give the output just as text only. Do not return anything else.\n"
            "Answer: \n"
            ,

            
            f"Code snippet: {code}\n"
            "Instruction: Summarize what this code snippet does in simple, non-technical language, focusing on its overall purpose and key operations for someone with little programming experience. "
            "Please give the output just as text only. Do not return anything else.\n"
            "Answer: \n"
            ,

            
            f"Code snippet: {code}\n"
            "Instruction: Generate an explanation of the code snippet in such a way that it can regenerate the code based on this explanation. "
            "Please give the output just as text only. Do not return anything else.\n"
            "Answer: \n"
            ,

        
            f"Code snippet - entry['code'] : {code}\n" \
            "Instruction: Explain how the code snippet in entry['code'] implements. Please provide the explanation as text only without any additional content.\n" \
            "Answer: \n"
        ]
            prompts.extend(prompt_templates)

        results = self.llm.generate(prompts, self.sampling_params)
        outputs = [res.outputs[0].text for res in results]

        # Group by original entry
        grouped = []
        for i in range(0, len(outputs), 5):
            grouped.append(outputs[i:i+5])

        return grouped


# === CONFIG ===
csv_path = '/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/pre_processing_CSN/CodeSearchNet_Python_train_cleaned.csv'
output_path = '/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/CS696DS-Oracle-Retrieving-Code-Explanations/Explanation_Generation/output/CSN_granite_train.csv'
checkpoint_path = '/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/CS696DS-Oracle-Retrieving-Code-Explanations/Explanation_Generation/checkpoint/CSN_granite_train.txt'
# model_path = "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"
# model = "deepseek"
model_path = "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
model = "granite"
batch_size = 128  # adjust to your GPU memory

# === HELPERS ===
def load_checkpoint():
    return int(open(checkpoint_path).read()) if os.path.exists(checkpoint_path) else 0

def save_checkpoint(line_num):
    with open(checkpoint_path, 'w') as f:
        f.write(str(line_num))

def write_header_if_needed(output_path, fieldnames):
    if not os.path.exists(output_path) or os.stat(output_path).st_size == 0:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

# === MAIN ===
def main():
    dataset = load_dataset('csv', data_files=csv_path, split='train', streaming=True)
    print(f'Dataset loaded - {csv_path}')
    print(f'Output location - {output_path}')

    start_line = load_checkpoint()
    print(f'Starting from line {start_line}')

    generator = ExplanationGeneratorLama(model_path)
    if hasattr(generator, 'model'):
        generator.model.eval()
    print(f'Model loaded - {model_path}')

    buffer = []
    line_num = 0
    written_header = False
    total_len = sum(1 for _ in dataset)

    with open(output_path, 'a', newline='', encoding='utf-8') as outfile:
        writer = None

        for row in tqdm(dataset, desc="Streaming dataset", total=total_len):
            line_num += 1
            if line_num <= start_line:
                continue

            buffer.append(row)

            if len(buffer) == batch_size:
                with torch.no_grad():
                    all_explanations = generator.generate_explanations_batch(buffer)

                for entry, explanations in zip(buffer, all_explanations):
                    for idx, explanation in enumerate(explanations):
                        entry[f'explanation_{model}_{idx+1}'] = explanation

                    # Setup writer on first pass
                    if writer is None:
                        fieldnames = list(entry.keys())
                        write_header_if_needed(output_path, fieldnames)
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

                    writer.writerow(entry)

                outfile.flush()
                save_checkpoint(line_num)
                buffer.clear()

        # Final batch (if any)
        if buffer:
            with torch.no_grad():
                all_explanations = generator.generate_explanations_batch(buffer)

            for entry, explanations in zip(buffer, all_explanations):
                for idx, explanation in enumerate(explanations):
                    entry[f'explanation_{model}_{idx+1}'] = explanation

                if writer is None:
                    fieldnames = list(entry.keys())
                    write_header_if_needed(output_path, fieldnames)
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

                writer.writerow(entry)

            outfile.flush()
            save_checkpoint(line_num)

    print(f"Finished processing. Output saved to {output_path}")

# === RUN ===
if __name__ == "__main__":
    main()
