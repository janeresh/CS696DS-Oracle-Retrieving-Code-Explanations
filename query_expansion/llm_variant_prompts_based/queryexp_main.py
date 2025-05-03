import sys
import pandas as pd

from explanation_generation2 import QueryExpander


MODEL = sys.argv[1]
INPUT_CSV = sys.argv[2]
OUTPUT_CSV = sys.argv[3]

num_generations = 5
query_batch_size = 4

MODEL_PATHS = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

# === MAIN ===
if __name__ == "__main__":
    expander = QueryExpander(MODEL_PATHS, OUTPUT_CSV)
    expander.generate_expansions(MODEL, INPUT_CSV, num_generations=num_generations, query_batch_size=query_batch_size)

    df_ckpt = pd.read_csv(OUTPUT_CSV)
    df_ckpt["generated_explanation_cleaned"] = df_ckpt["generated_explanation"].apply(expander.remove_ast_code)

    df_ckpt.drop("generated_explanation", axis=1, inplace=True)

    df_ckpt.rename(columns={"generated_explanation_cleaned": "generated_explanation"}, inplace=True)

    print("All Quries Expansion Generation Completed.")
