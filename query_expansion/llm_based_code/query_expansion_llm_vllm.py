import os
import sys
import pandas as pd
from vllm import LLM, SamplingParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATHS = {
    "deepseek": "/datasets/ai/deepseek/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/530ca3e1ad39d440e182c2e4317aa40f012512fa",
    "granite": "/datasets/ai/ibm-granite/hub/models--ibm-granite--granite-3.0-2b-instruct/snapshots/69e41fe735f54cec1792de2ac4f124b6cc84638f"
}

model_name = sys.argv[1]
input_csv = sys.argv[2]
output_csv = sys.argv[3]
checkpoint_csv = output_csv.replace(".csv", ".checkpoint.txt")
num_expansions = 10

model_path = MODEL_PATHS[model_name]
print(f"Loading model: {model_name} from {model_path}")
llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)
sampling_params = SamplingParams(temperature=0, max_tokens=256)


def clean_lines(text):
    return [
        line.strip().lstrip("0123456789).:- ").strip('" ')
        for line in text.splitlines() if line.strip()
    ]


def rank_queries(expanded, original, top_k=5):
    corpus = [original] + expanded
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    return sorted(zip(expanded, sims), key=lambda x: x[1], reverse=True)[:top_k]


def process_batch(df_batch, processed_ids):
    prompts = []
    SYSTEM_MSG = (
        "You are a query expansion engine specialized in programming queries. "
        "Return a list of clean, short, semantically similar variants of the input query. "
        "Each variant must be standalone, concise, and not include numbers or bullet points."
    )

    rows_to_process = []
    for _, row in df_batch.iterrows():
        if row["query_id"] in processed_ids:
            continue
        user_prompt = (
            f"Expand this query into {num_expansions} semantically similar variants:\n"
            f"{row['doc']}\n\n"
            "Only return the expanded queries, one per line, no bullets or numbers."
        )
        prompt = f"<|system|>\n{SYSTEM_MSG}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        prompts.append(prompt)
        rows_to_process.append(row)

    if not prompts:
        return []

    outputs = llm.generate(prompts, sampling_params)
    results = []

    for i, output in enumerate(outputs):
        row = rows_to_process[i]
        original_query = row["doc"]
        query_id = row["query_id"]
        expanded = clean_lines(output.outputs[0].text)
        ranked = rank_queries(expanded, original_query)

        for rank, (exp_q, score) in enumerate(ranked, 1):
            results.append({
                "id": query_id,
                "original_query": original_query,
                "rank": rank,
                "expanded_query": exp_q,
                "similarity": round(score, 4)
            })
    return results


# === Main with checkpointing ===
try:
    df = pd.read_csv(input_csv)

    if os.path.exists(checkpoint_csv):
        df_checkpoint = pd.read_csv(checkpoint_csv)
        processed_ids = set(df_checkpoint["id"].unique())
        print(f"Found checkpoint: {len(processed_ids)} queries already processed.")
    else:
        processed_ids = set()

    batch_results = process_batch(df, processed_ids)

    if batch_results:
        df_out = pd.DataFrame(batch_results)
        mode = 'a' if os.path.exists(checkpoint_csv) else 'w'
        df_out.to_csv(checkpoint_csv, mode=mode, header=(mode == 'w'), index=False)
        print(f"Saved checkpoint to {checkpoint_csv}")

    # Write final merged output
    pd.read_csv(checkpoint_csv).to_csv(output_csv, index=False)
    print(f"Final output saved to {output_csv}")

except Exception as e:
    print(f"Error processing {input_csv}: {e}")
