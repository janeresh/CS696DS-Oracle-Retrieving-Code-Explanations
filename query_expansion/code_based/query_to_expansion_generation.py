import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ast
import re
import textwrap
from typing import Tuple
import gc
from tqdm import tqdm

class QueryExpander:
    def __init__(self, model_paths, checkpoint_csv):
        self.model_paths = model_paths
        self.checkpoint_csv = checkpoint_csv

    def build_prompts(self, tokenizer, original_query):
        example_1 = (
            "Query: \n"
            "python code to write bool value 1 \n\n"
            "Explanation: \n"
            "The user wants to write a boolean value to a file in Python, specifically the True value represented as the number 1. "
            "In Python, booleans are subclasses of integers, where True maps to 1 and False to 0. "
            "The goal is to convert a boolean to its numeric form and write it to a file. "
            "This task involves using standard file I/O operations like open() and write(), along with type conversion via int() or str(). "
            "The user expects the written output to reflect a numeric boolean, not the string True. "
            "The code should represent boolean logic in persistent storage by saving True as 1 to a file using simple Python syntax and basic type handling."
        )

        example_2 = (
            "Query: \n"
            "python create directory using relative path \n\n"
            "Explanation: \n"
            "The user wants Python code that creates a directory using a relative path instead of an absolute one. "
            "A relative path points to a folder location based on the script's current location or the current working directory. "
            "The task requires creating a folder at a location like \"./logs\" or \"../output\". "
            "Python modules such as os and pathlib provide functions like os.mkdir(), os.makedirs(), and Path().mkdir() to achieve this. "
            "The user wants a way to form the relative path programmatically and ensure that the directory is created, "
            "enabling dynamic directory management based on the script location or runtime environment."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skilled software engineer who explains programming concepts clearly and simply. "
                    "You write natural language descriptions that help others understand what a code snippet is doing, "
                    "without using code or pseudocode. You focus on clarity, structure, and purpose of the logic.\n\n"
                    "Make your explanations specific, well-organized, and easy to follow. Keep them accurate but avoid overly technical terms. "
                    "Use plain language to describe how the program works step by step, like you're writing helpful comments or documentation.\n\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    "Expand the following query into a clear and useful explanation that describes what the program is expected to do. "
                    "Avoid code or pseudocode. Focus on describing the flow of logic, the goal of the task, and how the data is expected to move or change.\n\n"
                    f"Query: {original_query}\n\n"
                    "Expanded Explanation:"
                )
            }
        ]

        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_expansions(self, model_name, input_csv, query_batch_size=4, expansion_num=1):
        print("Generating Expansions..")
        model_path = self.model_paths[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)
        sampling_params = SamplingParams(
            temperature=0.5,
            max_tokens=500,
            stop=["</s>", "<|endoftext|>"],
            top_p=0.9,
            frequency_penalty=0.5,
            n=expansion_num
        )

        df = pd.read_csv(input_csv)
        all_query_ids = set(df["query_id"].unique())

        if os.path.exists(self.checkpoint_csv):
            df_checkpoint = pd.read_csv(self.checkpoint_csv)
            processed_ids = set(df_checkpoint["id"].apply(lambda x: str(x).split("_v")[0]))
            print(f"[Checkpoint] Resuming from {len(processed_ids)} queries")
            if all_query_ids.issubset(processed_ids):
                print("[Checkpoint] All queries already processed. Skipping generation.")
                return
        else:
            processed_ids = set()

        for i in tqdm(range(0, len(df), query_batch_size), desc="Batches"):
            batch_rows = df.iloc[i:i + query_batch_size]
            prompts = []
            metadata = []

            for _, row in batch_rows.iterrows():
                query_id = row["query_id"]
                original_query = row["doc"]
                corpus_id = row["corpus_id"]
                code = row["cleaned_code"]
                if query_id in processed_ids:
                    continue

                prompt = self.build_prompts(tokenizer=tokenizer, original_query=original_query)
                prompts.append(prompt)
                metadata.append((query_id, original_query, corpus_id, code))

            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params)
            checkpoint_rows = []

            for meta, output in zip(metadata, outputs):
                query_id, original_query, corpus_id, code = meta
                for idx, completion in enumerate(output.outputs):
                    raw_text = completion.text.strip().replace("</think>", "")
                    checkpoint_rows.append({
                        "id": f"{query_id}_v{idx+1}",
                        "query_id": query_id,
                        "original_query": original_query,
                        "corpus_id": corpus_id,
                        "code": code,
                        "para_query": f"Query:\n{original_query}\n\nExplanation:\n{self.remove_ast_code(raw_text)}"
                    })

            if checkpoint_rows:
                self.save_rows(checkpoint_rows)

            gc.collect()

    def save_rows(self, rows):
        df_to_save = pd.DataFrame(rows)
        write_header = not os.path.exists(self.checkpoint_csv) or os.stat(self.checkpoint_csv).st_size == 0
        df_to_save.to_csv(self.checkpoint_csv, mode='a', header=write_header, index=False)
        print(f"[Checkpoint] Saved {len(rows)} new entries")

    def contains_ast_code(self, text: str) -> bool:
        if re.search(r"```(?:python)?\n.*?```", text, flags=re.DOTALL):
            return True
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (
                    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                    ast.Import, ast.ImportFrom, ast.Assign, ast.Call,
                    ast.If, ast.For, ast.While, ast.With, ast.Try, ast.Expr
                )):
                    return True
            return False
        except SyntaxError:
            return False

    def remove_ast_code(self, text: str) -> str:
        text = re.sub(r"```(?:python)?\n.*?```", "", text, flags=re.DOTALL)
        lines = textwrap.dedent(text).splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # fallback: strip lines that start with code-like keywords
            return "\n".join(line for line in lines if not re.match(r"^\s*(def|class|import|for|if|while|return|with)\b", line)).strip()

        code_lines = set()
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start)
                for i in range(start, end + 1):
                    code_lines.add(i)

        cleaned = [line for i, line in enumerate(lines) if i not in code_lines]
        text_block = "\n".join(cleaned)
        text_block = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text_block, flags=re.DOTALL)
        text_block = re.sub(r'`[^`]+`', '', text_block)
        text_block = re.sub(r"\n{2,}", "\n\n", text_block)

        return text_block.strip()
