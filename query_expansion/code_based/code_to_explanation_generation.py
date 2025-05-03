import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ast
import re
import textwrap
import gc
from tqdm import tqdm

class CodeExplainer:
    def __init__(self, model_paths, checkpoint_csv):
        self.model_paths = model_paths
        self.checkpoint_csv = checkpoint_csv

    def build_prompts(self, tokenizer, code):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant who explains Python code in clear, simple English. "
                    "Avoid using code in your explanation. Focus on what the code does, how it works, and what each part is intended for. "
                    "Think like you're helping someone understand the code logic step by step."
                )
            },
            {
                "role": "user",
                "content": (
                    "Explain what the following Python code does in plain English. "
                    "Avoid using code in the explanation.\n\n"
                    f"```python\n{code.strip()}\n```"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_explanations(self, model_name, input_csv, query_batch_size=4):
        print("Generating Explanations..")
        model_path = self.model_paths[model_name]

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)
        sampling_params = SamplingParams(
            temperature=0.5, max_tokens=500,
            stop=["</s>", "<|endoftext|>"], top_p=0.9,
            frequency_penalty=0.5
        )

        df = pd.read_csv(input_csv)

        if os.path.exists(self.checkpoint_csv):
            df_checkpoint = pd.read_csv(self.checkpoint_csv)
            processed_ids = set(df_checkpoint["id"].unique())
            print(f"[Checkpoint] Resuming from {len(processed_ids)} completions")
        else:
            processed_ids = set()

        for i in tqdm(range(0, len(df), query_batch_size), desc="Batches"):
            batch_rows = df.iloc[i:i + query_batch_size]
            prompts = []
            metadata = []
            checkpoint_rows = []

            for row_idx, row in batch_rows.iterrows():
                query_id = row["query_id"]
                original_query = row["original_query"]
                corpus_id = row["corpus_id"]
                para_query = row["para_query"]
                code = row["code"]
                gen_code = row["generated_code"]
                code_idx = row["expansion_idx"]

                checkpoint_id = f"{query_id}_v{code_idx}"
                if checkpoint_id in processed_ids:
                    continue

                prompt = self.build_prompts(tokenizer, gen_code)
                prompts.append(prompt)
                metadata.append({
                    "row_idx": row_idx,
                    "query_id": query_id,
                    "original_query": original_query,
                    "corpus_id": corpus_id,
                    "code": code,
                    "para_query": para_query,
                    "expansion_idx": code_idx,
                    "generated_code": gen_code,
                    "id": checkpoint_id
                })

            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params)

            for meta, output in zip(metadata, outputs):
                raw_text = output.outputs[0].text.strip().replace("</think>", "")
                checkpoint_rows.append({
                    "id": meta["id"],
                    "row_idx": meta["row_idx"],
                    "query_id": meta["query_id"],
                    "original_query": meta["original_query"],
                    "corpus_id": meta["corpus_id"],
                    "code": meta["code"],
                    "para_query": meta["para_query"],
                    "expansion_idx": meta["expansion_idx"],
                    "generated_code": meta["generated_code"],
                    "explanation": self.remove_ast_code(raw_text)
                })

            if checkpoint_rows:
                self.save_rows(checkpoint_rows)
                checkpoint_rows.clear()

            gc.collect()

        print("Checkpointing complete.")

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
