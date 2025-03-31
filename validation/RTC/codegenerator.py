import torch
from vllm import LLM, SamplingParams

class CodeGeneration:
    def __init__(self, model_path):
        self.device = self.setup_cuda()
        self.model = self.load_model(model_path)
        self.sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=1000)
    
    def setup_cuda(self):
        """Check if CUDA is available and set the device accordingly."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        """Load the LLM model on the specified device."""
        return LLM(model=model_path, dtype="bfloat16", device=self.device, enforce_eager=False)

    def generate_llm(self, prompts):
        """Generate code for a batch of prompts using the LLM."""
        try:
            outputs = self.model.generate(prompts, self.sampling_params)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("Retrying after clearing CUDA cache...")
            outputs = self.model.generate(prompts, self.sampling_params)

        texts = [self.clean_output(output.outputs[0].text.strip(), "<think>") for output in outputs]

        # Optimize CUDA memory usage after processing the batch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        return texts

    def clean_output(self, text, keyword):
        """Extract generated code from the LLM output."""
        index = text.rfind(keyword)
        return text[index + len(keyword):].strip() if index != -1 else text

    def explanation_to_code(self, batch_descriptions, num_backward_passes):
        """
        Generate Python function code for multiple explanations in a batch.

        Args:
            batch_descriptions (pd.DataFrame): A DataFrame where each row contains four function descriptions.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains four generated functions (one per description).
        """
        # Convert DataFrame to list of lists
        batch_descriptions_list = batch_descriptions.values.tolist()

        all_prompts = []

        # Construct prompts for each description in the batch
        for row in batch_descriptions_list:
            for desc in row:
                strict_prompt_variants = [
                    f"Write only the Python function corresponding to the following description. "
                    "Do not provide explanations, comments, markdown, parameter descriptions, or return values. "
                    "Ensure that the function name and structure exactly match the description.\n\n"
                    f"Description:\n{desc}\n\nPython Code:\n",

                    f"Task: Implement the Python function as described below.\n\n"
                    f"Constraints:\n"
                    f"- Output ONLY valid Python code.\n"
                    f"- DO NOT include markdown (no ```python).\n"
                    f"- DO NOT include comments or explanations.\n"
                    f"- DO NOT describe parameters or return values.\n"
                    f"- Ensure the function signature, name, and structure EXACTLY match the description.\n\n"
                    f"Function Description:\n{desc}\n\n"
                    f"Code:\n",

                    f"You must write only the Python function described below.\n"
                    f"Do not include any explanations, comments, markdown, or extra output.\n"
                    f"Do not describe parameters or return values.\n"
                    f"Do not prefix your answer with ```python or any other text.\n\n"
                    f"The function name and structure must exactly match the description.\n\n"
                    f"Description:\n{desc}\n\n"
                    f"Python Function:\n"
                ][:num_backward_passes]

                for variant in strict_prompt_variants:
                    all_prompts.append(variant)

        # Generate code from prompts
        codes = self.generate_llm(all_prompts)

        # Reshape outputs to match the original batch shape (4 per row)
        # Reshape outputs: [row][description][variant]
        results = []
        i = 0
        for _ in batch_descriptions_list:
            row_outputs = []
            for _ in range(4):  # 4 descriptions per row
                row_outputs.append(codes[i:i + num_backward_passes])  # 3 prompt variants
                i += num_backward_passes
            results.append(row_outputs)

        return results

