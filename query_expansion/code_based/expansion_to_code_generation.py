import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
from tqdm import tqdm
from code_cleaning import CodeCleaner

class CodeGeneration:
    def __init__(self, model_paths, checkpoint_csv):
        self.model_paths = model_paths
        self.checkpoint_csv = checkpoint_csv
        self.codecleaner = CodeCleaner()

    def build_prompts(self, tokenizer, desc):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that writes Python code based strictly on user instructions. "
                    "Do not explain or comment, only return valid Python code."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Task: Implement the Python function as described below.\n\n"
                    f"Constraints:\n"
                    f"Output ONLY valid Python code.\n"
                    f"DO NOT include markdown (no ```python).\n"
                    f"DO NOT include comments or explanations.\n"
                    f"DO NOT describe parameters or return values.\n"
                    f"Ensure the function signature, name, and structure EXACTLY match the description.\n\n"
                    f"Function Description:\n{desc}\n\nCode:\n"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def explanation_to_code(self, model_name, input_csv, expansion_num=5, query_batch_size=4):
        print("Generating Codes..")
        model_path = self.model_paths[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)

        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=500,
            stop=["</s>", "<|endoftext|>"],
            top_p=0.95,
            frequency_penalty=0.5,
            n=expansion_num
        )

        df = pd.read_csv(input_csv)

        if os.path.exists(self.checkpoint_csv):
            df_checkpoint = pd.read_csv(self.checkpoint_csv)
            processed_ids = set(df_checkpoint["id"].apply(lambda x: str(x).split("_v")[0]))
            print(f"[Checkpoint] Resuming from {len(processed_ids)} queries")
        else:
            processed_ids = set()

        for i in tqdm(range(0, len(df), query_batch_size), desc="Batches"):
            batch_rows = df.iloc[i:i + query_batch_size]
            prompts = []
            metadata = []

            for _, row in batch_rows.iterrows():
                query_id = row["id"].split("_")[0]
                if query_id in processed_ids:
                    continue

                original_query = row["original_query"]
                corpus_id = row["corpus_id"]
                code = row["code"]
                para_query = row["para_query"]

                prompt = self.build_prompts(tokenizer, para_query)
                prompts.append(prompt)
                metadata.append({
                    "query_id": query_id,
                    "original_query": original_query,
                    "corpus_id": corpus_id,
                    "code": code,
                    "para_query": para_query,
                })

            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params)
            checkpoint_rows = []

            for meta, output in zip(metadata, outputs):
                query_id = meta["query_id"]
                original_query = meta["original_query"]
                corpus_id = meta["corpus_id"]
                code = meta["code"]
                para_query = meta["para_query"]

                for idx, completion in enumerate(output.outputs):
                    raw_text = completion.text.strip().replace("</think>", "")
                    checkpoint_rows.append({
                        "id": f"{query_id}_v{idx+1}",
                        "query_id": query_id,
                        "original_query": original_query,
                        "corpus_id": corpus_id,
                        "code": code,
                        "para_query": para_query,
                        "expansion_idx": idx + 1,
                        "generated_code": raw_text
                    })

            if checkpoint_rows:
                self.save_rows(checkpoint_rows)

            gc.collect()

    def save_rows(self, rows):
        df_to_save = pd.DataFrame(rows)
        write_header = not os.path.exists(self.checkpoint_csv) or os.stat(self.checkpoint_csv).st_size == 0
        df_to_save.to_csv(self.checkpoint_csv, mode='a', header=write_header, index=False)
        print(f"[Checkpoint] Saved {len(rows)} new entries")
