import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd

class CodeGeneration:
    def __init__(self, model_path, model_name, batch_size=8):
        self.model_name = model_name
        self.device = self.setup_cuda()
        self.model = self.load_model(model_path)
        self.sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1000)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def setup_cuda(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        return LLM(model=model_path, dtype="float16", device=self.device, enforce_eager=False)

    def build_prompts(self, desc):

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code based strictly on user instructions. Do not explain or comment, only return valid Python code."
            },
            {
                "role": "user", 
                "content": f"Task: Implement the Python function as described below.\n\nConstraints:\n Output ONLY valid Python code.\n DO NOT include markdown (no ```python).\n DO NOT include comments or explanations.\n DO NOT describe parameters or return values.\nEnsure the function signature, name, and structure EXACTLY match the description.\n\nFunction Description:\n{desc}\n\nCode:\n"

            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt


    def clean_output(self, text, keyword):
        index = text.rfind(keyword)
        return text[index + len(keyword):].strip() if index != -1 else text

    def explanation_to_code(self, batch_df, cols, num_backward_passes):
        """
        Args:
            batch_df (pd.DataFrame): A batch containing corpus_id, query_id, code, etc.
            cols (List[str]): Natural language descriptions per row, e.g., ["explanation_1", ..., "explanation_5"]
            num_backward_passes (int): Number of LLM generations per description

        Returns:
            pd.DataFrame: Each row contains original metadata + multiple code generations per explanation
        """
        prompts = []
        meta_info = []

        batch_descriptions = batch_df[cols].values.tolist()

        for i, (row_idx, row) in enumerate(batch_df.iterrows()):
            explanations = batch_descriptions[i]
            corpus_id = row["corpus_id"]
            cleaned_code = row["cleaned_code"]

            for desc_idx, explanation in enumerate(explanations):
                prompt = self.build_prompts(explanation)
                for code_variant_idx in range(num_backward_passes):
                    prompts.append(prompt)
                    meta_info.append({
                        "row_idx": row_idx,
                        "corpus_id": corpus_id,
                        "cleaned_code": cleaned_code,
                        "desc_idx": desc_idx,
                        "explanation": self.remove_ast_code(explanation),
                        "code_variant_idx": code_variant_idx + 1  # 1-based index
                    })

        outputs = self.model.generate(prompts, self.sampling_params)
        row_outputs = defaultdict(dict)

        for meta, output in zip(meta_info, outputs):
            row_idx = meta["row_idx"]
            desc_idx = meta["desc_idx"]
            code_variant_idx = meta["code_variant_idx"]
            explanation = meta["explanation"]
            corpus_id = meta["corpus_id"]
            cleaned_code = meta["cleaned_code"]

            # Initialize row metadata once
            if "corpus_id" not in row_outputs[row_idx]:
                row_outputs[row_idx]["corpus_id"] = corpus_id
                row_outputs[row_idx]["cleaned_code"] = cleaned_code

            row_outputs[row_idx][f"explanation_{desc_idx + 1}"] = explanation
            row_outputs[row_idx][f"generated_code_{desc_idx + 1}_code{code_variant_idx}"] = (
                output.outputs[0].text.strip().replace("</think>", "")
            )

        # Now collect all unique rows
        results = list(row_outputs.values())
        df_final = pd.DataFrame(results)
        return df_final
    
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




