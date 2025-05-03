import re
import textwrap
import ast
import torch
import gc
from prompt_variants import PromptVariants

class ExplanationRegenerator:
    def __init__(self, llm, sampling_params, tokenizer, output_file, save_interval=10):
        self.output_file = output_file
        self.save_interval = save_interval
        self.llm=llm
        self.sampling_params= sampling_params
        self.tokenizer=tokenizer

        prompt_var = PromptVariants(self.tokenizer)
        self.prompt_builders = {
            1: prompt_var.build_prompt_intent,
            2: prompt_var.build_prompt_technical,
            3: prompt_var.build_prompt_general,
            4: prompt_var.build_prompt_semitechnical,
            5: prompt_var.build_prompt_highlevel,
        }

    def postprocessing(self, new_explanation):
        for key in ["Answer:", "</think>", "\n"]:
            new_explanation = new_explanation.replace(key, "")
        return new_explanation
           

    def regenerate_explanation(self, code, explanation_index, prev_explanation):
        """Regenerate a single explanation_{i} using the appropriate prompt builder."""
        prompt_builder = self.prompt_builders.get(explanation_index)
        if not prompt_builder:
            raise ValueError(f"No prompt builder found for index {explanation_index}")

        prompt = prompt_builder(code, prev_explanation)
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            new_explanation = outputs[0].outputs[0].text.strip()
            new_explanation=self.postprocessing(new_explanation)
            new_explanation=self.remove_ast_code(new_explanation)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            gc.collect()
            raise e

        return new_explanation
    
    def remove_ast_code(self, text: str) -> str:
        text = re.sub(r"```(?:python)?\n.*?```", "", text, flags=re.DOTALL)
        lines = textwrap.dedent(text).splitlines()

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text

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
