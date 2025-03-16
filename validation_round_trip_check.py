import torch
import transformers
import difflib
import code_bert_score
import ast
import pandas as pd
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
from codebleu import calc_codebleu
import numpy as np
import logging

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)

# ✅ Function to Generate Text Using DeepSeek
def deepseek_generate(prompt, max_tokens=500, temperature=0.8):
    """
    Generates text using DeepSeek Coder with token length handling.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000, padding=True).to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,  # ✅ Fixing the argument name
            do_sample=True,
            num_return_sequences=1,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ✅ Forward Pass: Code → 3 Natural Language Descriptions
def code_to_explanations(doc, code_snippet):
    """
    Generates 3 natural language explanations from a given code snippet.
    """
    prompt = (
        f"Doc string: {doc}\n"
        f"Code snippet: {code_snippet}\n"
        "Instruction: Provide a concise explanation of what the above doc and code mean. "
        "Generate strictly less than 100 words in total.\n"
        "Answer:\n"
    )
    generated_exp = deepseek_generate(prompt, max_tokens=128, temperature=0.8)
    cleaned_exp = generated_exp.strip().replace(prompt, "").strip()
    return cleaned_exp

# ✅ Backward Pass: Each Explanation → Code
def explanation_to_code(description):
    """
    Generates Python code from a cleaned natural language description.
    """
    prompt = (
        "Write only the Python function corresponding to the following description. "
        "Do not provide explanations, comments, markdown, parameter descriptions, or return values. "
        "Ensure that the function name and structure exactly match the description.\n\n"
        f"Description:\n{description}\n\nPython Code:\n"
    )
    generated_code = deepseek_generate(prompt, max_tokens=512, temperature=0.8)
    cleaned_code = generated_code.strip().replace(prompt, "").strip()
    return cleaned_code

# ✅ Normalizing Code for Better Comparison
def normalize_code(code):
    """
    Normalize Python code by parsing it into an AST and standardizing the format.
    """
    try:
        return ast.dump(ast.parse(code))
    except SyntaxError:
        return None

# ✅ Compute Correct Generation Count
def correct_generation(sim_scores):
    return sum(1 for score in sim_scores if score > 0.7)

# ✅ Evaluate Metrics
def evaluate_metrics(original_code, generated_code):
    """
    Evaluates RTC correctness using similarity metrics.
    """
    exact_match = original_code.strip() == generated_code.strip()
    similarity = code_bert_score_func(original_code, generated_code)
    
    return exact_match, similarity

# ✅ Pass@1 Computation
def pass_at_1(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

# ✅ CodeBERT Similarity Score
def code_bert_score_func(x: str, x_hat: str) -> float:
    P, R, F1, _ = code_bert_score.score(cands=[x_hat], refs=[x], lang='python')
    return F1.mean().item()

# ✅ CodeBLEU Similarity Score
def codebleu_func(x: str, x_hat: str) -> float:
    return calc_codebleu([x], [x_hat], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)

# ✅ Compute RTC
def compute_rtc(sim_scores):
    if not sim_scores:
        return 0.0
    return sum(sim_scores) / len(sim_scores)

# ✅ Compute LPass
def evaluate_lpass(codes, original_code):
    return 1 if any(code_bert_score_func(original_code, code) > 0.75 for code in codes) else 0

# ✅ File Processing
input_csv = sys.argv[1]
output_csv = sys.argv[2]
df = pd.read_csv(input_csv)

results = []
for iter, row in df.iterrows():
    logger.info(f"Processing row {iter}")
    original_code = str(row["code"]).strip()
    doc = str(row["doc"]).strip()

    codes, sim_scores, explanations, matches = [], [], [], []

    for _ in range(3):
        explanation = code_to_explanations(doc, original_code)
        generated_code = explanation_to_code(explanation)
        exact_match, similarity_score = evaluate_metrics(original_code, generated_code)

        codes.append(generated_code)
        sim_scores.append(similarity_score)
        explanations.append(explanation)
        matches.append(exact_match)

    true_count = correct_generation(sim_scores)
    final_rtcpass = compute_rtc(sim_scores)
    pass_score = pass_at_1(3, true_count, 1)

    results.append({
        "Original Code": original_code,
        "Generated Code1": codes[0],
        "Generated Code2": codes[1],
        "Generated Code3": codes[2],
        "Explanation1": explanations[0],
        "Explanation2": explanations[1],
        "Explanation3": explanations[2],
        "Exact Match": matches,
        "CodeBERTScore": sim_scores,
        "RTCPass": final_rtcpass,
        "Pass@1": pass_score
    })

    logger.info(f"Processed row {iter}")

logger.info("Writing to CSV")
pd.DataFrame(results).to_csv(output_csv, index=False)
