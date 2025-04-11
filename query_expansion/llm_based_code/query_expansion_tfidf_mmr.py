import os
import sys
import pandas as pd
import numpy as np
import torch
import psutil
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", trust_remote_code=True)
embedding_model = AutoModel.from_pretrained("intfloat/e5-base-v2", trust_remote_code=True).to("cuda")
embedding_model.eval()


class QueryExpander:
    def __init__(self, model_paths, output_tfidf):
        self.model_paths = model_paths
        self.output_tfidf = output_tfidf

    def generate_expansions(self, model_name, input_csv, num_generations=10, query_batch_size=4):
        print("Generating Expansions..")
        model_path = self.model_paths[model_name]
        llm = LLM(model=model_path, dtype="float16", trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0.5, max_tokens=256)

        df = pd.read_csv(input_csv)
        all_query_ids = set(df["query_id"].unique())

        if os.path.exists(checkpoint_csv):
            df_checkpoint = pd.read_csv(checkpoint_csv)
            processed_ids = set(df_checkpoint["id"].unique())
            print(f"[Checkpoint] Resuming from {len(processed_ids)} queries")

            if all_query_ids.issubset(processed_ids):
                print("[Checkpoint] All queries already processed. Skipping generation.")
                return
        else:
            processed_ids = set()

        example_1 = (
            "Query: \npython code to write bool value 1",
            "Explanation: \nThe user wants to write a boolean value to a file in Python..."
        )
        example_2 = (
            "Query: \npython create directory using relative path",
            "Explanation: \nThe user wants Python code that creates a directory using a relative path..."
        )
        SYSTEM_MSG = (
            "You are an expert technical writer specialized in generating detailed, natural language explanations for programming and computer science queries. "
            "Your responses must be clear, logically structured, and contain no code, pseudocode, or bullet points."
        )

        for i in range(0, len(df), query_batch_size):
            print("Query Index: ", i)
            batch_rows = df.iloc[i:i + query_batch_size]
            prompts = []
            metadata = []
            checkpoint_rows = []

            for _, row in batch_rows.iterrows():
                query_id = row["query_id"]
                original_query = row["doc"]
                if query_id in processed_ids:
                    continue

                prompt_base = (
                    f"{SYSTEM_MSG}\n\n"
                    "Below are two examples...\n\n"
                    f"{example_1}\n\n"
                    f"{example_2}\n\n"
                    "Expand the following programming-related query into a detailed and informative explanation..."
                    f"\n\nQuery: {original_query}\n\nExpanded Explanation:"
                )

                for _ in range(num_generations):
                    prompts.append(prompt_base)
                    metadata.append((query_id, original_query))

            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params)

            for meta, output in zip(metadata, outputs):
                query_id, original_query = meta
                raw_text = output.outputs[0].text
                raw_text = raw_text.replace("</think>", "")
                checkpoint_rows.append({
                    "id": query_id,
                    "original_query": original_query,
                    "generated_explanation": raw_text.strip()
                })

            if checkpoint_rows:
                pd.DataFrame(checkpoint_rows).to_csv(checkpoint_csv, mode='a', header=not os.path.exists(checkpoint_csv), index=False)
                print(f"[Checkpoint] Saved {len(checkpoint_rows)} new entries")

        print("Checkpointing complete.")


class TFIDFRanker:
    def __init__(self):
        pass

    def rank(self, df_ckpt):
        grouped = df_ckpt.groupby("id")
        results = []

        for query_id, group in grouped:
            original_query = group.iloc[0]["original_query"]
            explanations = group["generated_explanation"].dropna().tolist()
            if not explanations:
                continue

            corpus = [original_query] + explanations
            tfidf = TfidfVectorizer().fit_transform(corpus)
            if tfidf.shape[0] < 2:
                continue

            sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            ranked = sorted(zip(explanations, sims), key=lambda x: x[1], reverse=True)[:5]

            for rank, (exp_q, score) in enumerate(ranked, 1):
                results.append({
                    "id": query_id,
                    "original_query": original_query,
                    "rank": rank,
                    "ranked_explanations": exp_q,
                    "rank-similarity": round(score, 4)
                })

        return results


class MMRRanker:
    def __init__(self):
        pass

    def print_memory_usage(self, tag):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        print(f"[MEMORY] {tag}: {mem:.2f} MB")

    def get_embeddings(self, texts, device='cuda', batch_size=4):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = embedding_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = embedding_model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1)
                embeddings.extend(mean_pooled.cpu().numpy())
            self.print_memory_usage(f"Batch {i // batch_size + 1}")
        torch.cuda.empty_cache()
        return np.vstack(embeddings)

    def mmr(self, query, documents, top_k=5, lambda_param=0.5):
        embeddings = self.get_embeddings([query] + documents, device='cuda')
        query_embedding = embeddings[0:1]
        doc_embeddings = embeddings[1:]

        selected = []
        remaining = list(range(len(documents)))
        sim_to_query = cosine_similarity(query_embedding, doc_embeddings)[0]

        while len(selected) < top_k and remaining:
            mmr_scores = []
            for idx in remaining:
                diversity = 0 if not selected else max(
                    cosine_similarity([doc_embeddings[idx]], [doc_embeddings[i] for i in selected])[0]
                )
                score = lambda_param * sim_to_query[idx] - (1 - lambda_param) * diversity
                mmr_scores.append((idx, score))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [documents[i] for i in selected]

    def rank(self, df_ckpt):
        grouped = df_ckpt.groupby("id")
        results = []

        for query_id, group in grouped:
            original_query = group.iloc[0]["original_query"]
            explanations = group["generated_explanation"].dropna().tolist()
            if not explanations:
                continue

            mmr_ranked = self.mmr(original_query, explanations)

            for rank, exp_q in enumerate(mmr_ranked, 1):
                results.append({
                    "id": query_id,
                    "original_query": original_query,
                    "mmr_rank": rank,
                    "mmr_explanation": exp_q
                })

        return results


# === MAIN ===
if __name__ == "__main__":
    expander = QueryExpander(MODEL_PATHS, OUTPUT_TFIDF)
    expander.generate_expansions(MODEL, INPUT_CSV, num_generations=num_generations, query_batch_size=query_batch_size)

    df_ckpt = pd.read_csv(checkpoint_csv)

    tfidf_ranker = TFIDFRanker()
    mmr_ranker = MMRRanker()

    tfidf_results = tfidf_ranker.rank(df_ckpt)
    mmr_results = mmr_ranker.rank(df_ckpt)

    pd.DataFrame(tfidf_results).to_csv(OUTPUT_TFIDF, index=False)
    pd.DataFrame(mmr_results).to_csv(OUTPUT_MMR, index=False)

    print("Ranking complete. Results saved.")
