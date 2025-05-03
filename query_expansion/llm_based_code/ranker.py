import numpy as np
import torch
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import os

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

        self.embedding_tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2", trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained("intfloat/e5-base-v2", trust_remote_code=True).to("cuda")
        self.embedding_model.eval()
        pass

    def print_memory_usage(self, tag):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        print(f"[MEMORY] {tag}: {mem:.2f} MB")

    def get_embeddings(self, texts, device='cuda', batch_size=4):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.embedding_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
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