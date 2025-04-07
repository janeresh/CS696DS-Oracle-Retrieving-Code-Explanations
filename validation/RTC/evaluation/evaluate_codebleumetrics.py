import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import sys

def compute_rtc(sim_scores):
    Nf, Nb = len(sim_scores), len(sim_scores[0])
    denom = Nf * Nb
    return sum(sum(row) for row in sim_scores) / denom

def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def run_codebleu_on_strings(reference_codes, generated_code, lang='python',
                            params='0.25,0.25,0.25,0.25', script_path='CodeBLEU/calc_code_bleu.py'):
    ref_paths = []
    # Flatten multi-line code into single line
    
    try:
        # Write reference code(s)
        for ref_code in reference_codes:
            flattened_ref = " ".join(line.strip() for line in ref_code.strip().splitlines() if line.strip())

            tmp_ref = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
            tmp_ref.write(flattened_ref + "\n")
            tmp_ref.flush()
            ref_paths.append(tmp_ref.name)
            tmp_ref.close()

        # Write hypothesis (generated code)
        flattened_hyp = " ".join(line.strip() for line in generated_code.strip().splitlines() if line.strip())
        tmp_hyp = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
        tmp_hyp.write(flattened_hyp + "\n")
        tmp_hyp.flush()
        hyp_path = tmp_hyp.name
        tmp_hyp.close()

        # Run subprocess
        cmd = ["python", script_path, "--refs", *ref_paths, "--hyp", hyp_path,
               "--lang", lang, "--params", params]
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("=== CodeBLEU OUTPUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)

        score_line = [line for line in result.stdout.splitlines() if 'CodeBLEU score:' in line]
        score = float(score_line[0].split()[-1]) if score_line else None

    finally:
        for path in ref_paths:
            os.remove(path)
        if 'hyp_path' in locals():
            os.remove(hyp_path)

    return score

# === Main ===
if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    model = sys.argv[3]  # e.g., "deepseek"

    col_no = int(sys.argv[4])
    num_backward_passes = int(sys.argv[5])

    df = pd.read_csv(input_csv)
    updated_rows = []
    new_columns = set()

    for row_idx in range(len(df)):
        try:
            print(f"\nProcessing row {row_idx + 1}/{len(df)}...")
            row = df.iloc[row_idx].copy()
            ref_code = row["Original_Code"]
            code_bleu_scores = []

            for i in range(col_no):
                code_bleu_scores_sub = []
                for j in range(num_backward_passes):
                    gen_code_col = f"Generated_Code_{model}_{i+1}_code{j+1}"
                    if gen_code_col not in row:
                        print(f"Missing column: {gen_code_col}, skipping.")
                        codebleu_metric = None
                    else:
                        gen_code = row[gen_code_col]
                        if pd.isna(ref_code) or pd.isna(gen_code):
                            codebleu_metric = 0.0
                        else:
                            codebleu_metric = run_codebleu_on_strings(
                                [str(ref_code)], str(gen_code)
                            )
                    col_name = f"CodeBLEU_Score_{model}_{i + 1}_code{j + 1}"
                    row[col_name] = codebleu_metric
                    new_columns.add(col_name)
                    code_bleu_scores_sub.append(codebleu_metric)

                code_bleu_scores.append(code_bleu_scores_sub)

            # Compute RTC and Pass@1 only if all values exist
            if all(score is not None for scores in code_bleu_scores for score in scores):
                rtc_score = compute_rtc(code_bleu_scores)
                pass1_count = sum(score > 0.8 for scores in code_bleu_scores for score in scores)
                pass1_score = pass_at_k(col_no * num_backward_passes, pass1_count, 1)
            else:
                rtc_score = None
                pass1_score = None

            rtc_col = f"RTC_{model}_CodeBLEU_Score"
            pass1_col = f"Pass@1_{model}_CodeBLEU_Score"
            row[rtc_col] = rtc_score
            row[pass1_col] = pass1_score
            new_columns.update([rtc_col, pass1_col])

            updated_rows.append(row.to_dict())

        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {e}")
            updated_rows.append(df.iloc[row_idx].to_dict())

    # Ensure all rows have all new columns
    for row_dict in updated_rows:
        for col in new_columns:
            if col not in row_dict:
                row_dict[col] = None

    # Create DataFrame and save
    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_csv(output_csv, index=False)
    print(f"\nMetrics file saved to: {output_csv}")
