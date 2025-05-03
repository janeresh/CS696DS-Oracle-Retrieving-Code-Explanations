import pandas as pd
from explanation_regenerator import ExplanationRegenerator
from validation import ValidationPipeline
import sys
import time
import logging
import gc
import torch
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
# -------- Configuration --------
model_path_dict = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

input_file = sys.argv[1]
output_file = sys.argv[2]
model = sys.argv[3]

threshold = 0.8
max_retries = 5
BATCH_SIZE = 4
col_no = 5
b_pass = 3
device = "cuda" if torch.cuda.is_available() else "cpu"
low_rtc_file = output_file.replace(".csv", "_low_rtc.csv")

# -------- Logger --------
logging.basicConfig(stream=sys.stdout, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("RTC-Regenerator")

logger.info(f"Starting regeneration with model: {model}")
model_path = model_path_dict[model]

# -------- Load Data --------
logger.info(f"Loading input file: {input_file}")
df = pd.read_csv(input_file)
df["filtered"] = False
# -------- Checkpoint: Resume from low RTC file if exists --------
if os.path.exists(low_rtc_file):
    logger.info(f"Resuming from existing low RTC file: {low_rtc_file}")
    low_rtc_df = pd.read_csv(low_rtc_file)
else:
    logger.info("Scanning for explanations with RTC < threshold...")
    low_rtc_df = pd.DataFrame(columns=["corpus_id", "cleaned_code", "explanation", "explanation_number"])
    for i in range(col_no):
        rtc_col = f'RTC_CodeBERT_Score_{i+1}'
        explanation_col = f'explanation_{i+1}'
        if rtc_col in df.columns and explanation_col in df.columns:
            filtered = df[df[rtc_col] < threshold][["corpus_id", "cleaned_code", explanation_col]].copy()
            filtered = filtered.rename(columns={explanation_col: "explanation"})
            filtered["explanation_number"] = i + 1
            low_rtc_df = pd.concat([low_rtc_df, filtered], ignore_index=True)

    logger.info(f"Total low RTC rows collected: {len(low_rtc_df)}")
    low_rtc_df.to_csv(low_rtc_file, index=False)
    logger.info(f"Saved intermediate low RTC file: {low_rtc_file}")

torch.cuda.empty_cache()
gc.collect()
llm = LLM(model=model_path, dtype="half", device=device, enforce_eager=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024)
sampling_params_2 = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=500)

# -------- Initialize Pipeline --------
validator = ValidationPipeline(llm, sampling_params, tokenizer, batch_size=BATCH_SIZE)
updated_rows = []
logger.info(f"Processing total of {len(low_rtc_df)} rows in batches of {BATCH_SIZE}")
explanation_regenerator = ExplanationRegenerator(llm, sampling_params_2, tokenizer, output_file)

# -------- Checking the Memory --------
if torch.cuda.memory_reserved() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
    logger.warning("High reserved memory detected — consider reducing BATCH_SIZE.")


# -------- Process Batches --------
for i in range(0, len(low_rtc_df), BATCH_SIZE):
    logger.info(f"\n--- Processing batch {i // BATCH_SIZE + 1} ---")
    batch = low_rtc_df.iloc[i:i + BATCH_SIZE].copy()

    for _, row in batch.iterrows():
        corpus_id = row["corpus_id"]
        explanation_number = row["explanation_number"]
        explanation = row["explanation"]
        retries = 0
        codebert_rtc_score = 0.0

        logger.info(f"Regenerating for corpus_id={corpus_id}, explanation_{explanation_number}")

        while codebert_rtc_score < threshold and retries < max_retries:
            try:
                new_explanation = explanation_regenerator.regenerate_explanation(
                    row['cleaned_code'],
                    explanation_number,
                    explanation
                )

                validated_row, codebert_rtc_score = validator.validation_pipeline(new_explanation, row['cleaned_code'])
                logger.info(validated_row.keys())
                validated_row[f"RTC_CodeBERT_Score"] = codebert_rtc_score
    
                logger.info(f"[Try {retries + 1}] RTC Score: {codebert_rtc_score:.3f}")

                validated_row.update({
                    "corpus_id": corpus_id,
                    "cleaned_code": row["cleaned_code"],
                    "explanation_number": explanation_number
                    
                })

            except RuntimeError as e:
                logger.error(f"[Try {retries + 1}] CUDA OOM: {e}")
            finally:
                del new_explanation
                torch.cuda.empty_cache()
                gc.collect()

            retries += 1
        torch.cuda.empty_cache()
        gc.collect()


        if codebert_rtc_score >= threshold:
            validated_row["filtered"]= True
            logger.info(f"Updated explanation passed threshold (RTC={codebert_rtc_score:.3f})")
        else:
            validated_row["filtered"]= False
            logger.warning(f"Failed to improve RTC beyond threshold after {max_retries} retries")
        
        validated_row = validator.other_metrics_calculation(validated_row, b_pass)

        if codebert_rtc_score < threshold:
            continue  # Skip failed attempts

        updated_rows.append(validated_row)

        # ---- Update only the matched row in the original df ----
        idx = df.index[df["corpus_id"] == corpus_id].tolist()
        if idx:
            idx = idx[0]
            df.at[idx, f"explanation_{explanation_number}"] = validated_row["explanation"]
            df.at[idx, f"generated_code_{explanation_number}_code1"] = validated_row.get("generated_code_code1", "")
            df.at[idx, f"generated_code_{explanation_number}_code2"] = validated_row.get("generated_code_code2", "")
            df.at[idx, f"generated_code_{explanation_number}_code3"] = validated_row.get("generated_code_code3", "")

            df.at[idx, f"CodeBERT_Score_{explanation_number}_code1"] = validated_row.get("CodeBERT_Score_code1", 0.0)
            df.at[idx, f"CodeBERT_Score_{explanation_number}_code2"] = validated_row.get("CodeBERT_Score_code2", 0.0)
            df.at[idx, f"CodeBERT_Score_{explanation_number}_code3"] = validated_row.get("CodeBERT_Score_code3", 0.0)

            df.at[idx, f"RTC_CodeBERT_Score_{explanation_number}"] = validated_row.get("RTC_CodeBERT_Score", 0.0)

            df.at[idx, f"CodeBLEU_Score_{explanation_number}_code1"] = validated_row.get("CodeBLEU_Score_code1", 0.0)
            df.at[idx, f"CodeBLEU_Score_{explanation_number}_code2"] = validated_row.get("CodeBLEU_Score_code2", 0.0)
            df.at[idx, f"CodeBLEU_Score_{explanation_number}_code3"] = validated_row.get("CodeBLEU_Score_code3", 0.0)

            df.at[idx, f"RTC_CodeBLEU_Score_{explanation_number}"] = validated_row.get("RTC_CodeBLEU_Score", 0.0)

            df.at[idx, f"Structural_Score_{explanation_number}_code1"] = validated_row.get("Structural_Score_code1", 0.0)
            df.at[idx, f"Structural_Score_{explanation_number}_code2"] = validated_row.get("Structural_Score_code2", 0.0)
            df.at[idx, f"Structural_Score_{explanation_number}_code3"] = validated_row.get("Structural_Score_code3", 0.0)

            df.at[idx, f"RTC_Structural_Score_{explanation_number}"] = validated_row.get("RTC_Structural_Score", 0.0)
            df.at[idx, "filtered"] = validated_row.get("filtered", False)


            struct_col_sim_scores = []
            bert_col_sim_scores = []
            bleu_col_sim_scores = []

            for i in range(col_no):
                struct_col_row_scores = []
                bert_col_row_scores = []
                bleu_col_row_scores = []
                for j in range(b_pass):
                    struct_col = f"Structural_Score_{i+1}_code_{j+1}"
                    bert_col = f"CodeBERT_Score_{i+1}_code_{j+1}"
                    bleu_col = f"CodeBLEU_Score_{i+1}_code_{j+1}"
                    struct_col_row_scores.append(df.at[idx, struct_col] )
                    bert_col_row_scores.append(df.at[idx, bert_col] )
                    bleu_col_row_scores.append(df.at[idx, bleu_col] )
                struct_col_sim_scores.append(struct_col_row_scores)
                bert_col_sim_scores.append(bert_col_row_scores)
                bleu_col_sim_scores.append(bleu_col_row_scores)
            bert_rtc, bleu_rtc, struct_rtc, pass1_bert, pass1_bleu, pass1_struct=validator.rtc_pass_1_recalculation(bert_col_sim_scores, bleu_col_sim_scores, struct_col_sim_scores)
            df.at[idx, f"RTC_CodeBERT_Score"] = bert_rtc
            df.at[idx, f"RTC_CodeBLEU_Score"] = bleu_rtc
            df.at[idx, f"RTC_Struct_Score"] = struct_rtc
            df.at[idx, f"Pass@1_CodeBERT_Score"] = pass1_bert
            df.at[idx, f"Pass@1_CodeBLEU_Score"] = pass1_bleu
            df.at[idx, f"Pass@1_Struct_Score"] = pass1_struct
            
del validator.llm
del explanation_regenerator.llm

# -------- Save Final Output --------
df.to_csv(output_file, index=False)
logger.info(f"\nCompleted. Final output saved to: {output_file}")
