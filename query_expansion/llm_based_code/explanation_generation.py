import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QueryExpander:
    def __init__(self, model_paths, checkpoint_csv):
        self.model_paths = model_paths
        self.checkpoint_csv = checkpoint_csv

    def generate_expansions(self, model_name, input_csv, num_generations=10, query_batch_size=4):
        print("Generating Expansions..")
        model_path = self.model_paths[model_name]

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.5, max_tokens=500, stop=["</s>", "<|endoftext|>"], top_p=0.9, frequency_penalty=0.5)

        df = pd.read_csv(input_csv)
        all_query_ids = set(df["query_id"].unique())

        if os.path.exists(self.checkpoint_csv):
            df_checkpoint = pd.read_csv(self.checkpoint_csv)
            processed_ids = set(df_checkpoint["id"].unique())
            print(f"[Checkpoint] Resuming from {len(processed_ids)} queries")

            if all_query_ids.issubset(processed_ids):
                print("[Checkpoint] All queries already processed. Skipping generation.")
                return
        else:
            processed_ids = set()

        example_1 = "Query: \npython code to write bool value 1 \n\nExplanation: \nThe user wants to write a boolean value to a file in Python, specifically the True value represented as the number 1. In Python, booleans are subclasses of integers, where True maps to 1 and False to 0. The goal is to convert a boolean to its numeric form and write it to a file. This task involves using standard file I/O operations like open() and write(), along with type conversion via int() or str(). The user expects the written output to reflect a numeric boolean, not the string True. The code should represent boolean logic in persistent storage by saving True as 1 to a file using simple Python syntax and basic type handling."

        example_2 = "Query: \npython create directory using relative path \n\nExplanation: \nThe user wants Python code that creates a directory using a relative path instead of an absolute one. A relative path points to a folder location based on the script current location or the current working directory. The task requires creating a folder at a location like \"./logs\" or \"../output\". Python modules such as os and pathlib provide functions like os.mkdir(), os.makedirs(), and Path().mkdir() to achieve this. The user wants a way to form the relative path programmatically and ensure that the directory is created, enabling dynamic directory management based on the script location or runtime environment."

        for i in range(0, len(df), query_batch_size):
            print("Query Index: ", i)
            batch_rows = df.iloc[i:i + query_batch_size]
            prompts = []
            metadata = []
            checkpoint_rows = []

            for _, row in batch_rows.iterrows():
                query_id = row["query_id"]
                original_query = row["doc"]
                if query_id in processed_ids:
                    continue

                messages = [
                    {
                        "role": "system",
                        "content": f"You are an expert software engineer and technical writer. You explain programming tasks in formal, structured, and highly technical English. You avoid using code, but convey exact module usage, reasoning, and technical context as if writing documentation or whitepapers. Below are two examples on how you can generate: \n {example_1} \n {example_2}"
                    },
                    {
                        "role": "user", 
                        #"content": f"Expand the following programming-related query into a detailed, technical explanation written in natural language. Do not use code snippets, pseudocode, or bullet points. Focus on correct terminology, relevant Python modules, and best practices.\n\nQuery: {original_query}\n\nExpanded Explanation:"
                        "content": f"Expand the following query into a technical explanation that closely aligns with how a code snippet implementing it would be described in natural language. Avoid code or pseudocode. Instead, explain the logic, flow, and structure in a way that would naturally match a code comment or documentation summary. \nUse technical terms for operations (like coercion, serialization, construction, etc.), describe the data flow, and make the explanation specific enough to recover the implementation logic.\n\nQuery: {original_query}\n\nExpanded Explanation:"

                    }
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


                for _ in range(num_generations):
                    prompts.append(prompt)
                    metadata.append((query_id, original_query))

            if not prompts:
                continue
                    
            outputs = llm.generate(prompts, sampling_params)

            for meta, output in zip(metadata, outputs):
                query_id, original_query = meta
                raw_text = output.outputs[0].text.strip()
                raw_text = raw_text.replace("</think>", "")
                checkpoint_rows.append({
                    "id": query_id,
                    "original_query": original_query,
                    "generated_explanation": raw_text.strip()
                })

            if checkpoint_rows:
                pd.DataFrame(checkpoint_rows).to_csv(self.checkpoint_csv, mode='a', header=not os.path.exists(self.checkpoint_csv), index=False)
                print(f"[Checkpoint] Saved {len(checkpoint_rows)} new entries")

        print("Checkpointing complete.")




