import torch
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import gc
from code_cleaning import CodeCleaner
from clean_code import CodeCleaner2
from metrics_calculation import CodeBERTScorer, CodeStructureScorer, CodeBLEUEvaluator


class ValidationPipeline:
    def __init__(self, llm, sampling_params, tokenizer, batch_size=4):
        self.llm = llm
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        
        self.batch_size = batch_size
        self.cleaner = CodeCleaner()
        self.cleaner2 = CodeCleaner2()
        self.codebert = CodeBERTScorer(threshold=0.8)
        self.structure = CodeStructureScorer()
        self.codebleu = CodeBLEUEvaluator(script_path="../CodeBLEU/calc_code_bleu.py")

    def build_pseudocodes_prompts_func(self, desc):
        # example_1 = (
        #     "Task: Implement a function that returns the square of a number.\n"
        #     "Constraints:\n Output ONLY valid Python code. DO NOT include markdown or comments.\n"
        #     "Ensure the function signature, name, and structure EXACTLY match the description.\n"
        #     "Return 3 significantly different implementations that achieve the same functionality.\n"
        #     "Code 1:\n"
        #     "def square(n): return n * n\n\n"
        #     "Code 2:\n"
        #     "def square(n): return pow(n, 2)\n\n"
        #     "Code 3:\n"
        #     "def square(n): return n ** 2\n"
        # )

        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a helpful assistant that writes Python code. Always return multiple diverse versions."
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"{example_1}\n\n"
        #             f"Task: Implement the function as described below.\n"
        #             f"Function Description:\n{desc}\n\n"
        #             f"Return 3 clearly different implementations (e.g., use loops, library methods, recursion, etc.).\n"
        #             f"Format output as:\n"
        #             f"Code 1:\n<code>\n\nCode 2:\n<code>\n\nCode 3:\n<code>\n"
        #         )
        #     }
        # ]
        example_1="""
            Pseudocode 1:
                FOR each item in list:
                    IF item > max:
                        max = item

                Pseudocode 2:
                Sort the list in descending order
                Return the first item

                Pseudocode 3:
                Use a divide-and-conquer approach to compare sublists
                Return the maximum from the recursive comparisons

        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that writes diverse algorithmic pseudocode. "
                    "You never write real code—only abstract, language-agnostic pseudocode. "
                    "Always return multiple clearly distinct versions of the solution."
                )
            },
            {
                "role": "user",
                "content": (
                    f"{example_1}\n\n"
                    f"Task: Design an algorithm as described below.\n"
                    f"Function Description:\n{desc}\n\n"
                    f"Return 3 clearly different pseudocode implementations "
                    f"(e.g., use loops, recursion, built-in operations, or different data structures).\n"
                    f"Format output as:\n"
                    f"Pseudocode 1:\n<pseudocode>\n\nPseudocode 2:\n<pseudocode>\n\nPseudocode 3:\n<pseudocode>\n"
                )
            }
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    # def split_generated_variants(self, text):
    #     return [m.strip() for m in re.split(r"Code\s+\d+:\s*", text.strip()) if m.strip()][:3]

    def split_generated_variants(self, text):
        splits = re.split(r"Pseudocode\s*\d+\s*:", text, flags=re.IGNORECASE)
        splits = [s.strip() for s in splits if s.strip()]
        if len(splits) < 3:
            print(f"Only {len(splits)} pseudocode blocks detected. Duplicating to meet expected count.")
    
        while len(splits) < 3:
            splits.append(splits[0])
        return {
            "pseudocode1": splits[0],
            "pseudocode2": splits[1],
            "pseudocode3": splits[2]
        }


    def build_prompts_from_3_pseudocodes(self, pseudocodes):
        prompts = []
        for i, desc in enumerate(pseudocodes, start=1):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes Python code based strictly on user instructions. Do not explain or comment, only return valid Python code."
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: Implement the Python function as described below.\n\n"
                        f"Constraints:\n"
                        f"- Output ONLY valid Python code.\n"
                        f"- DO NOT include markdown (no ```python).\n"
                        f"- DO NOT include comments or explanations.\n"
                        f"- DO NOT describe parameters or return values.\n"
                        f"- Ensure the function signature, name, and structure EXACTLY match the description.\n\n"
                        f"Function Pseudocode:\n{desc}\n\n"
                        f"Code:\n"
                    )
                }
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        return prompts


    def validation_pipeline(self, explanation, cleaned_code):
        row = {"explanation": explanation, "cleaned_code": cleaned_code}
        codebert_sim_scores = []

        prompt = self.build_pseudocodes_prompts_func(explanation)
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            text_output = outputs[0].outputs[0].text.strip()
        except RuntimeError as e:
            torch.cuda.empty_cache()
            raise e

        pseudocodes = self.split_generated_variants(text_output)

        prompts = self.build_prompts_from_3_pseudocodes(pseudocodes)
        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
            codes = [out.outputs[0].text.strip() for out in outputs]
        except RuntimeError as e:
            torch.cuda.empty_cache()
            raise e
        print("code_variants: ", len(codes))

        for idx, code in enumerate(codes):
            cleaned_generated_code = self.cleaner.split_code_and_comments(code)
            cleaned_generated_code = self.cleaner2.clean_code(cleaned_generated_code)
            codebert_score = self.codebert.compute_codebertscore(cleaned_code, cleaned_generated_code)

            row[f"generated_code_code{idx+1}"] = cleaned_generated_code
            row[f"CodeBERT_Score_code{idx+1}"] = codebert_score
            
            codebert_sim_scores.append(codebert_score)


        codebert_rtc_score = self.codebert.compute_rtc([codebert_sim_scores])

        return row, codebert_rtc_score
    
    def other_metrics_calculation(self, row, num_backward_passes):
        code = row.get("cleaned_code", "")
        codebleu_sim_scores = []
        struct_sim_scores = []

        for idx in range(num_backward_passes):
            codebleu_score = self.codebleu.run_codebleu_on_strings([code], row[f"generated_code_code{idx+1}"])
            row[f"CodeBLEU_Score_code{idx+1}"] = codebleu_score
            struct_score = self.structure.compute_cosine_similarity(code, row[f"generated_code_code{idx+1}"])
            row[f"Structural_Score_code{idx+1}"] = struct_score
            codebleu_sim_scores.append(codebleu_score)
            struct_sim_scores.append(struct_score)
        print(codebleu_sim_scores, struct_sim_scores)
        codebleu_rtc_score = self.codebleu.compute_rtc([codebleu_sim_scores])
        struct_rtc_score = self.structure.compute_rtc([struct_sim_scores])
        row[f"RTC_CodeBLEU_Score"] = codebleu_rtc_score
        row[f"RTC_Structural_Score"] = struct_rtc_score

        return row

    def rtc_pass_1_recalculation(self, bert_col_sim_scores, bleu_col_sim_scores, struct_col_sim_scores):
        bert_rtc = self.codebert.compute_rtc(bert_col_sim_scores)
        bleu_rtc = self.codebleu.compute_rtc(bleu_col_sim_scores)
        struct_rtc = self.structure.compute_rtc(struct_col_sim_scores)
        true_positive_bert = sum(score > 0.85 for scores in bert_col_sim_scores for score in scores)
        true_positive_bleu = sum(score > 0.85 for scores in bleu_col_sim_scores for score in scores)
        true_positive_struct = sum(score > 0.85 for scores in struct_col_sim_scores for score in scores)
        pass1_bert = round(self.codebert.pass_at_k(len(bert_col_sim_scores) * len(bert_col_sim_scores[0]), true_positive_bert, 1), 4)
        pass1_bleu = round(self.codebleu.pass_at_k(len(bert_col_sim_scores) * len(bleu_col_sim_scores[0]), true_positive_bleu, 1), 4)
        pass1_struct = round(self.structure.pass_at_k(len(bert_col_sim_scores) * len(struct_col_sim_scores[0]), true_positive_struct, 1), 4)

        return bert_rtc, bleu_rtc, struct_rtc, pass1_bert, pass1_bleu, pass1_struct


    
