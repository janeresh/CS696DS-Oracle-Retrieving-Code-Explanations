import numpy as np
import code_bert_score
import pandas as pd
import sys

def compute_codebertscore(original_code, generated_code):
    P, R, F1, _ = code_bert_score.score(cands=[generated_code], refs=[original_code], lang='python')
    return F1.mean().item()

def evaluate_generated_code(original_code, generated_code):
    exact_match = original_code.strip() == generated_code.strip()
    codebertscore = compute_codebertscore(original_code, generated_code)
    return exact_match, codebertscore

def compute_rtc(sim_scores):
    Nf, Nb = len(sim_scores), len(sim_scores[0])
    denom = Nf * Nb
    rtc_score = sum(sum(row) for row in sim_scores) / denom
    return rtc_score

def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_bert_match_metrics(row, model, col_no, num_backward_passes):
    code_bert_scores = []
    exact_matches = []
    original_code = row.get("Original_Code", "")

    for i in range(col_no):
        code_bert_scores_sub = []
        exact_matches_sub = []
        for j in range(num_backward_passes):
            gen_col = f"Generated_Code_{model}_{i+1}_code{j+1}"
            score_col = f"CodeBERT_Score_{model}_{i+1}_code{j+1}"
            match_col = f"Exact_Match_{model}_{i+1}_code{j+1}"
            
            gen_code = row.get(gen_col, "")
            
            print("Original_Code", original_code, gen_code)
            
            if isinstance(gen_code, str) and isinstance(original_code, str):
                exact_match, code_bert_score = evaluate_generated_code(original_code, gen_code)
                row[score_col] = code_bert_score
                row[match_col] = exact_match
                code_bert_scores_sub.append(code_bert_score)
                exact_matches_sub.append(exact_match)
            else:
                print("Entered")
                row[score_col] = 0.0
                row[match_col] = False
                code_bert_scores_sub.append(0.0)
                exact_matches_sub.append(False)
        
        code_bert_scores.append(code_bert_scores_sub)

    row[f"RTC_{model}_CodeBERT_Score"] = compute_rtc(code_bert_scores)
    pass1_true_count = sum(score > 0.8 for scores in code_bert_scores for score in scores)
    row[f"Pass@1_{model}_CodeBERT_Score"] = pass_at_k(col_no * num_backward_passes, pass1_true_count, 1)

    return row

# --- MAIN EXECUTION ---

input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]

col_no = int(sys.argv[4])
num_backward_passes = int(sys.argv[5])

df = pd.read_csv(input_csv)

for row_idx in range(len(df)):
    try:
        print(f"Processing row {row_idx+1}/{len(df)}...")
        row = df.iloc[row_idx]
        updated_row = compute_bert_match_metrics(row, model, col_no, num_backward_passes)
        for col in updated_row.index:
            df.at[row_idx, col] = updated_row[col]
    except Exception as e:
        print(f"Error processing row {row_idx+1}: {e}")

df.to_csv(output_csv, index=False)
print(f"\nMetrics file saved: {output_csv}")
