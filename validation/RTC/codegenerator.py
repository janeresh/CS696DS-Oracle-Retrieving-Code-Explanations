import torch
from vllm import LLM, SamplingParams

class CodeGeneration:
    def __init__(self, model_path, model_type):
        self.device = self.setup_cuda()
        self.model = self.load_model(model_path)
        self.model_type = model_type
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=1000)
    
    def setup_cuda(self):
        """Check if CUDA is available and set the device accordingly."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        """Load the LLM model on the specified device."""
        return LLM(model=model_path, dtype="bfloat16", device=self.device, enforce_eager=False)

    def generate_llm(self, prompts):
        """Generate code for a batch of prompts using the LLM."""
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

    def explanation_to_code(self, batch_descriptions):
        """
        Generate Python function code for multiple explanations in a batch.
        
        Args:
            batch_descriptions (pd.DataFrame): A DataFrame containing explanation columns.

        Returns:
            List[List[str]]: A list of lists where each inner list contains four generated functions.
        """
        # Convert DataFrame to list of lists (each row contains four explanations)
        batch_descriptions_list = batch_descriptions.values.tolist()

        # Construct prompts for all explanations in the batch
        prompts = [
            f"Write only the Python function corresponding to the following description. "
            "Do not provide explanations, comments, markdown, parameter descriptions, or return values. "
            "Ensure that the function name and structure exactly match the description.\n\n"
            f"Description:\n{desc}\n\nPython Code:\n"
            for row in batch_descriptions_list for desc in row
        ]
        
        # Generate code for all explanations in the batch
        codes = self.generate_llm(prompts)

        # Reshape generated outputs into batches of 4 per row
        return [codes[i:i + 4] for i in range(0, len(codes), 4)]
