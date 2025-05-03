import sys
import pandas as pd

from explanation_generation import QueryExpander
from ranker import MMRRanker, TFIDFRanker


MODEL = sys.argv[1]
INPUT_CSV = sys.argv[2]
OUTPUT_TFIDF = sys.argv[3]
OUTPUT_MMR = sys.argv[4]

num_generations = 10
query_batch_size = 4

checkpoint_csv = OUTPUT_TFIDF.replace(".csv", ".checkpoint.csv")
MODEL_PATHS = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

# === MAIN ===
if __name__ == "__main__":
    expander = QueryExpander(MODEL_PATHS, checkpoint_csv)
    expander.generate_expansions(MODEL, INPUT_CSV, num_generations=num_generations, query_batch_size=query_batch_size)

    df_ckpt = pd.read_csv(checkpoint_csv)

    tfidf_ranker = TFIDFRanker()
    mmr_ranker = MMRRanker()

    tfidf_results = tfidf_ranker.rank(df_ckpt)
    mmr_results = mmr_ranker.rank(df_ckpt)

    pd.DataFrame(tfidf_results).to_csv(OUTPUT_TFIDF, index=False)
    pd.DataFrame(mmr_results).to_csv(OUTPUT_MMR, index=False)

    print("Ranking complete. Results saved.")
