import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import ast
import re
import textwrap
from typing import Tuple

class QueryExpander:
    def __init__(self, model_paths, checkpoint_csv):
        self.model_paths = model_paths
        self.checkpoint_csv = checkpoint_csv

    def build_prompt_intent(self, tokenizer, original_query):
        example_1 = (
            "Query:\npython code to write bool value 1\n\n"
            "Explanation:\nThe user wants to write a boolean value to a file in a way that reflects its numeric representation. "
            "They are likely storing True as 1 for compatibility with systems or formats expecting integers."
        )
        example_2 = (
            "Query:\npython create directory using relative path\n\n"
            "Explanation:\nThe user wants to create a folder in a location relative to the current script or working directory, "
            "rather than specifying the full path."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that expands programming queries by focusing on the user’s intent and motivation. "
                    "Do not explain how the code works or what functions are used. Focus on what the user wants to accomplish and why. "
                    "Do not include code or pseudocode. Below are two examples:\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Expand the following query by describing the user's intent. Focus on the underlying purpose and what outcome the user is expecting. "
                    f"Avoid implementation or code details.\n\nQuery: {original_query}\n\nExpanded Explanation:"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def build_prompt_technical(self, tokenizer, original_query):
        example_1 = (
            "Query:\npython code to write bool value 1\n\n"
            "Explanation:\nThis involves converting a boolean value into its integer equivalent and writing it to a persistent storage medium like a file. "
            "The process includes serialization and handling of file I/O operations."
        )
        example_2 = (
            "Query:\npython create directory using relative path\n\n"
            "Explanation:\nThis task requires identifying the working directory context and creating a new directory relative to it. "
            "It may involve error handling in case the directory already exists."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant who expands programming queries into step-by-step implementation logic. "
                    "Do not include code or pseudocode, but mention the sequence of operations and key concepts involved. Below are two examples:\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Expand the following query into a detailed technical explanation. Focus on logic, data flow, and necessary steps "
                    f"without referencing any code.\n\nQuery: {original_query}\n\nExpanded Explanation:"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def build_prompt_general(self, tokenizer, original_query):
        example_1 = (
            "Query:\npython code to write bool value 1\n\n"
            "Explanation:\nThe user wants to save a yes-or-no type value, written as a 1, into a file for later use."
        )
        example_2 = (
            "Query:\npython create directory using relative path\n\n"
            "Explanation:\nThe user wants to make a new folder in a place nearby the current file or project instead of typing the whole folder location."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that explains programming tasks in simple, general language for beginners. "
                    "Avoid any technical jargon or code. Here are two examples:\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Explain what the following query is asking for in simple words. Avoid using programming terms or examples.\n\n"
                    f"Query: {original_query}\n\nExpanded Explanation:"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   
    def build_prompt_semitechnical(self, tokenizer, original_query):
        example_1 = (
            "Query:\npython code to write bool value 1\n\n"
            "Explanation:\nThe task requires converting a boolean to a format that can be saved, such as converting it to a number, "
            "and then writing that to a file using basic data output operations."
        )
        example_2 = (
            "Query:\npython create directory using relative path\n\n"
            "Explanation:\nThe user wants to make a directory using a path that is dependent on where the script is currently running, "
            "which typically helps in making the program more portable."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semi-technical assistant who explains programming tasks with moderate detail. "
                    "Mention concepts like input/output or path handling but don’t show any code. Below are two examples:\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Describe what the following query is asking for, using moderate technical depth. Avoid any actual code or syntax.\n\n"
                    f"Query: {original_query}\n\nExpanded Explanation:"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def build_prompt_highlevel(self, tokenizer, original_query):
        example_1 = (
            "Query:\npython code to write bool value 1\n\n"
            "Explanation:\nThis task reflects a data serialization workflow where a boolean needs to be encoded numerically and saved "
            "as part of a larger persistence or logging operation."
        )
        example_2 = (
            "Query:\npython create directory using relative path\n\n"
            "Explanation:\nThis query indicates a file system operation aimed at organizing outputs or resources within a dynamically scoped working directory."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software architect interpreting programming queries at a system-design level. "
                    "Explain how the task fits into a broader context like data persistence, file system orchestration, or automation workflows. No code. Below are two examples:\n"
                    f"{example_1}\n\n{example_2}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Interpret the following query at a high level. Describe the broader role or operation it serves in an application or system, "
                    f"without talking about how to implement it.\n\nQuery: {original_query}\n\nExpanded Explanation:"
                )
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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

                prompts.append(self.build_prompt_intent(tokenizer, original_query))
                prompts.append(self.build_prompt_technical(tokenizer, original_query))
                prompts.append(self.build_prompt_general(tokenizer, original_query))
                prompts.append(self.build_prompt_semitechnical(tokenizer, original_query))
                prompts.append(self.build_prompt_highlevel(tokenizer, original_query))

                for _ in range(num_generations):
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
                    "generated_explanation": "Query:\n" + original_query + "\n\n" + "Explanation:\n" + raw_text.strip()
                })

            if checkpoint_rows:
                pd.DataFrame(checkpoint_rows).to_csv(self.checkpoint_csv, mode='a', header=not os.path.exists(self.checkpoint_csv), index=False)
                print(f"[Checkpoint] Saved {len(checkpoint_rows)} new entries")

        print("Checkpointing complete.")
 

    def contains_ast_code(self, text: str) -> bool:
        """
        Returns True if the given text contains fenced code blocks or parses as valid
        Python code with executable or structural AST nodes.
        """
        # Detect fenced code blocks
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
        """
        Removes all valid Python code and markdown code blocks from the text,
        including function/class defs, imports, executable expressions,
        docstrings, inline code, and fenced code.
        """
        # First: remove all fenced code blocks (```...```)
        text = re.sub(r"```(?:python)?\n.*?```", "", text, flags=re.DOTALL)

        # AST-based structural removal
        lines = textwrap.dedent(text).splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text  # return as-is if it's not parsable at all

        code_lines = set()
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start)
                for i in range(start, end + 1):
                    code_lines.add(i)

        # Remove all AST-detected code lines
        cleaned = [line for i, line in enumerate(lines) if i not in code_lines]

        # Remove docstrings and inline code
        text_block = "\n".join(cleaned)
        text_block = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text_block, flags=re.DOTALL)
        text_block = re.sub(r'`[^`]+`', '', text_block)
        text_block = re.sub(r"\n{2,}", "\n\n", text_block)

        return text_block.strip()