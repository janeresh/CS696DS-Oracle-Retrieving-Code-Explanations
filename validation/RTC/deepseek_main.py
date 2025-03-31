import torch
import logging
import sys
import pandas as pd
from codegenerator import CodeGeneration
from tqdm import tqdm
import gc

# Logging configuration
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

# Clear CUDA memory before starting
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Define model paths
model_path_dict = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

# Load dataset
input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]
df = pd.read_csv(input_csv)

batch_size = 4  # ⚠️ Reduce batch size to fit within 40GB VRAM
num_backward_passes = 3
col_no = 4
# results = {}
results=[]

# Process each model separately to avoid VRAM overload
#for model in model_path_dict:
logger.info(f"Initializing CodeGeneration for {model}...")

#Load model (keep only one model in memory at a time)
codegenerator = CodeGeneration(model_path_dict[model])
col_name = f"explanation_{model}"
cols = [f"{col_name}_{i}_cleaned" for i in range(1,col_no+1)]

# Initialize result storage
result_list = []

for i in tqdm(range(0, len(df), batch_size), desc=f"Processing batches for {model}"):
    batch_df = df.iloc[i:i + batch_size]

    #Generate code for batch (Optimized)
    try:
        generated_codes = codegenerator.explanation_to_code(batch_df[cols], num_backward_passes)
    except RuntimeError as e:
        logger.error(f"CUDA OOM Error at batch {i}-{i + batch_size}: {e}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        continue  # Skip to next batch

    #Store results efficiently
    # batch_result_df = pd.DataFrame(
    #     generated_codes, columns=[f"Generated_Code_{model}_{j+1}" for j in range(col_no)]
    # )
    flattened_rows = []

    # Iterate over each row in the batch and its corresponding generated outputs
    for batch_row, gen_row in zip(batch_df.itertuples(index=False), generated_codes):
        row_dict = {
            "corpus_id": batch_row.corpus_id,
            "query_id": batch_row.query_id,
            "Original_Code": batch_row.code,
        }
        
        # Add each explanation output as a dictionary
        for j in range(col_no):  # 4 descriptions per row
            variant_outputs = gen_row[j]  # list of 3 prompt completions
            for k in range(num_backward_passes):
                row_dict[f"Generated_Code_{model}_{j+1}_code{k+1}"] = variant_outputs[k]

        
        flattened_rows.append(row_dict)

    # Create DataFrame from all rows in this batch
    batch_result_df = pd.DataFrame(flattened_rows)
    result_list.append(batch_result_df)


    #Add identifiers (Corpus ID, query_id, and Original Code)
    # batch_result_df["Original_Code"] = batch_df["code"].values
    # batch_result_df["corpus_id"] = batch_df["corpus_id"].values
    # batch_result_df["query_id"] = batch_df["query_id"].values

    # result_list.append(batch_result_df)

    #Memory Optimization: Clear Cache after Each Batch
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    # gc.collect()

#Delete Model After Processing (Frees Up VRAM)
# del codegenerator
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
# gc.collect()

#Store processed model results
results = pd.concat(result_list, ignore_index=True)


#Merge DeepSeek & Granite Results on Corpus_ID & Doc_ID
# merged_result_df = results["deepseek"].merge(
#     results["granite"],
#     on=["corpus_id", "query_id", "Original_Code"],  # Ensure correct merging
#     how="inner"  # Only keep rows that exist in both
# )


# Apply similarity computation for both DeepSeek & Granite
# for model in model_path_dict:
#     merged_result_df = merged_result_df.apply(lambda row: compute_metrics(row, model), axis=1)


# # Apply common RTC & Pass@1 calculation
# merged_result_df = merged_result_df.apply(lambda row: compute_common_scores(row, model_path_dict), axis=1)
# columns_order = ["corpus_id", "query_id", "Original_Code", 
#                 "Generated_Code_deepseek_1", "Generated_Code_deepseek_2", "Generated_Code_deepseek_3", "Generated_Code_deepseek_4",
#                 "Generated_Code_granite_1", "Generated_Code_granite_2", "Generated_Code_granite_3", "Generated_Code_granite_4",
#                 "Sim_Code_deepseek_1", "Exact_Match_deepseek_1", 
#                 "Sim_Code_deepseek_2", "Exact_Match_deepseek_2",
#                 "Sim_Code_deepseek_3", "Exact_Match_deepseek_3",
#                 "Sim_Code_deepseek_4", "Exact_Match_deepseek_4",
#                 "RTC_deepseek", "Pass@1_deepseek",
#                 "Sim_Code_granite_1", "Exact_Match_granite_1",
#                 "Sim_Code_granite_2", "Exact_Match_granite_2",
#                 "Sim_Code_granite_3", "Exact_Match_granite_3",
#                 "Sim_Code_granite_4", "Exact_Match_granite_4",
#                 "RTC_granite", "Pass@1_granite",
#                 "RTC_Common", "Pass@1_Common"]
# merged_result_df=merged_result_df[columns_order]

# merged_result_df.to_csv(output_csv, index=False)
# logger.info(f"Results saved to {output_csv}")
results.to_csv(output_csv, index=False)
logger.info(f"Results saved to {output_csv}")


