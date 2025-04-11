import torch
import numpy
import json
from tqdm import tqdm
import os
import gc
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from vllm import LLM, SamplingParams, LLMEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Set environment variable to help with memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def clean_output(text,keyword):
    # keyword = "Answer:"
    index = text.rfind(keyword)  # Find the last occurrence of "Answer:"
    if index != -1:
        return text[index + len(keyword):].strip()  
    return text


class ExplanationGeneratorLama:
    def __init__(self, model_name, max_new_tokens=500):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.padding_side = "left"
        
        self.max_new_tokens = max_new_tokens
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_new_tokens
        )
        self.llm = LLM(
            model=model_name,
            dtype="half"  # using half precision for GPU
        )
        # self.engine = LLMEngine(model_name, sampling_params=self.sampling_params)

    def generate_explanations_batch(self, entries, max_new_tokens=500):
        # Create prompts by combining each entry with each prompt template.
        
        prompts = []
        
        for entry in entries:
            prompt_templates = [
                f"Doc string: {entry['doc']}\n"
                f"Code snippet: {entry['code']}\n"
                "Instruction: Provide a concise explanation of what the above doc and code mean. "
                "Generate strictly less than 100 words in total. Please give the output just as text only. Do not return anything else.\n"
                "Answer: \n"
                , 

                f"Doc string: {entry['doc']}\n"
                f"Code snippet: {entry['code']}\n"
                "Instruction: Provide a detailed line-by-line explanation of this code snippet, describing the purpose and functionality of each statement, function, and control structure. "
                "Please give the output just as text only. Do not return anything else.\n"
                "Answer: \n"
                ,

                f"Doc string: {entry['doc']}\n"
                f"Code snippet: {entry['code']}\n"
                "Instruction: Summarize what this code snippet does in simple, non-technical language, focusing on its overall purpose and key operations for someone with little programming experience. "
                "Please give the output just as text only. Do not return anything else.\n"
                "Answer: \n"
                ,

                f"Doc string: {entry['doc']}\n"
                f"Code snippet: {entry['code']}\n"
                "Instruction: Generate an explanation of the code snippet in such a way that it can regenerate the code based on this explanation. "
                "Please give the output just as text only. Do not return anything else.\n"
                "Answer: \n"
                ,

                f"Doc string - entry['doc'] : {entry['doc']}\n" \
                f"Code snippet - entry['code'] : {entry['code']}\n" \
                "Instruction: Explain how the code snippet in entry['code'] implements or achieves the functionality described in the doc string in entry['doc']. Please provide the explanation as text only without any additional content.\n" \
                "Answer: \n"
            ]
            
            for template in prompt_templates:
                prompt = (
                    f"Doc string: {entry['doc']}\n"
                    f"Code snippet: {entry['code']}\n"
                    f"{template}\n"
                    "Answer: \n"
                )
                prompts.append(prompt)
                
        # results = self.engine.generate(prompts)
        results = self.llm.generate(prompts, self.sampling_params)
        explanations = [result.outputs[0].text for result in results]
        
        # Regroup explanations by entry.
        n_prompts = len(prompt_templates)
        grouped_explanations = []
        for i in range(0, len(explanations), n_prompts):
            grouped_explanations.append(explanations[i:i+n_prompts])
            
        return grouped_explanations


if __name__ == "__main__":
    csv_path = "/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/data_preprocessing/CodeSearchNet_Python_valid.csv"  # change this to your CSV input path
    output_csv_path = "/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/explanations/CodeSearchNet_Python_valid_vllm.csv"
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    print("Data loaded from CSV")
    

    models_dict = {
        "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
        'granite': "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"

    }
    
    
    batch_size = 200
    
    for model_key, model_path in tqdm(models_dict.items(), desc="Processing models"):
        print(f"\nProcessing model {model_key}...")
        generator = ExplanationGeneratorLama(model_path)
        if hasattr(generator, 'model'):
            generator.model.eval()
        
        # Process the DataFrame in batches
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch_entries = df.iloc[i:i+batch_size][["corpus_id", "query_id", "doc", "code"]].to_dict("records")
            
            # Wrap inference in a no_grad context to prevent gradient computations.
            with torch.no_grad():
                batch_explanations = generator.generate_explanations_batch(batch_entries)
            
            for j, explanation_variants in enumerate(batch_explanations):
                for idx, raw_text in enumerate(explanation_variants):
                    
                    df.loc[i+j, f'explanation_{model_key}_{idx+1}'] = raw_text
            
            # torch.cuda.empty_cache()
            # gc.collect()
        
        del generator
        # torch.cuda.empty_cache()
        # gc.collect()
        df.to_csv(output_csv_path, index=False)
    
    # df.to_csv(output_csv_path, index=False)
    print(f"\nExplanations from all models have been saved to {output_csv_path}")
