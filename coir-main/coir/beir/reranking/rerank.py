import logging
from typing import Dict, List

logger = logging.getLogger(__name__)
import numpy as np

#Parent class for any reranking model
class Rerank1:
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.rerank_results = {}

    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])

        #### Starting to Rerank using cross-attention
        logging.info("Starting To Rerank Top-{}....".format(top_k))
        rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]

        #### Reranking results
        self.rerank_results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.rerank_results[query_id][doc_id] = score

        return self.rerank_results 
      
class Rerank:
    
    def __init__(self, model, batch_size: int = 128, alpha: float = 0.5):
        self.cross_encoder = model
        self.batch_size = batch_size
        self.alpha = alpha  # Weight for retriever + reranker scores
        self.rerank_results = {}
        
    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids, initial_scores = [], [], []
        
        for query_id in results:
            sorted_docs = sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]
            for doc_id, init_score in sorted_docs:
                pair_ids.append([query_id, doc_id])
                corpus_text = corpus[doc_id].get("text", "").strip()
                sentence_pairs.append([queries[query_id], corpus_text])
                initial_scores.append(init_score)  # Store initial retriever scores

        #### Rerank using cross-encoder
        logging.info(f"Starting To Rerank Top-{top_k}....")
        rerank_scores = [float(score) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]
        
        #### Normalize cross-encoder scores
        min_score, max_score = min(rerank_scores), max(rerank_scores)
        if max_score > min_score:  # Avoid division by zero
            rerank_scores = [(s - min_score) / (max_score - min_score) for s in rerank_scores]

        #### Combine retriever & reranker scores
        final_scores = [(self.alpha * init) + ((1 - self.alpha) * rerank) 
                        for init, rerank in zip(initial_scores, rerank_scores)]

        #### Store new rankings
        self.rerank_results = {query_id: {} for query_id in results}
        for (query_id, doc_id), final_score in zip(pair_ids, final_scores):
            self.rerank_results[query_id][doc_id] = final_score

        return self.rerank_results

