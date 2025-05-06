from .. import BaseSearch
from .util import cos_sim, dot_score
import logging
import torch
from typing import Dict
import heapq
from rank_bm25 import BM25Okapi
import spacy
nlp = spacy.blank("en")
import heapq
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import os, pickle
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import faiss
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def tokenize(text):
    return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]


logger = logging.getLogger(__name__)

# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, alpha=0.7, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.alpha = alpha
        self.bm25 = None
        self.results = {}
        self.ce_tokenizer = AutoTokenizer.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.ce_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").half().to("cuda")
    
    def build_bm25(self, corpus, cache_path="bm25_tokens.pkl"):
        """
        Build BM25 index once and cache the tokenised corpus to disk.
        Next run: reload tokens and rebuild BM25 in milliseconds.
        """
        self.corpus_ids = list(corpus.keys())

        # -------- check cache --------
        if os.path.exists(cache_path):
            logger.info(f"Loading BM25 tokens from {cache_path}")
            with open(cache_path, "rb") as f:
                corpus_tokens = pickle.load(f)
        else:
            logger.info("Tokenising corpus for BM25 …")
            corpus_tokens = [self.tokenize(doc["text"]) for doc in corpus.values()]
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(corpus_tokens, f)

        # (Re)create the BM25Okapi object
        self.bm25 = BM25Okapi(corpus_tokens)

    @staticmethod
    def tokenize(text: str):
        return text.lower().split()
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids

        print('in exact_search.py')

        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]

        print('len of queries: ', len(query_ids))

        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]
        
        print('len of corpus: ', len(corpus_ids))

        print("Encoding Corpus in batches... Warning: This might take a while!")
        print("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
        self.corpus_chunk_size = 10000
        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            print("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
                )
            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()
            
            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]                  
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        print('exact_search finished')
        return self.results 

    def search_reranker(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids

        print('in exact_search reranker.py')

        lambda_blend = 0.5
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        query_ids   = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        print('encoding queries')
        q_emb = self.model.encode_queries(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor)

        corpus_ids_sorted = sorted(
            corpus, key=lambda k: len(corpus[k].get("title", "") +
                                    corpus[k].get("text", "")), reverse=True)
        corpus_sorted = [corpus[cid] for cid in corpus_ids_sorted]

        result_heaps = {qid: [] for qid in query_ids}
        print('encoding corpus')
        for start in range(0, len(corpus_sorted), self.corpus_chunk_size):
            end = min(start + self.corpus_chunk_size, len(corpus_sorted))
            c_emb = self.model.encode_corpus(
                corpus_sorted[start:end],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor)

            sims = self.score_functions[score_function](q_emb, c_emb)
            sims[torch.isnan(sims)] = -1

            vals, idxs = torch.topk(
                sims, min(top_k + 1, c_emb.shape[0]), dim=1,
                largest=True, sorted=return_sorted)
            vals, idxs = vals.cpu().tolist(), idxs.cpu().tolist()

            for qi, qid in enumerate(query_ids):
                heap = result_heaps[qid]
                for idx, dscore in zip(idxs[qi], vals[qi]):
                    doc_id = corpus_ids_sorted[start + idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (dscore, doc_id))
                    else:
                        heapq.heappushpop(heap, (dscore, doc_id))

        self.results = {}
        print('using cross encoder')
        for qid, qtext in zip(query_ids, query_texts):
            dense_hits = heapq.nlargest(top_k, result_heaps[qid])  # [(dscore, doc)]
            docs = [corpus[doc_id]["text"] for _, doc_id in dense_hits]

            enc = self.ce_tokenizer([qtext]*len(docs),
                                    docs,
                                    padding=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to("cuda")
            with torch.no_grad():
                ce_logits = self.ce_model(**enc).logits.squeeze()   # (len(docs),)

            ce_scores = ce_logits.cpu().tolist()
            fused = []
            for (dscore, doc_id), ce in zip(dense_hits, ce_scores):
                fused_score = lambda_blend * ce + (1 - lambda_blend) * dscore
                fused.append((fused_score, doc_id))

            fused.sort(reverse=True)
            self.results[qid] = {doc: s for s, doc in fused[:top_k]}

        return self.results 

    def search_dres_sparse(
            self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            top_k: int,
            score_function: str = "cos_sim",
            cache_path: str = "bm25_tokens.pkl",
            return_sorted: bool = False,
            **kwargs) -> Dict[str, Dict[str, float]]:

        print("in exact_search.py  dres+sparse (RRF)")

        if self.bm25 is None:
            self.build_bm25(corpus, cache_path+"bm25_tokens.pkl")    
        id2row = {cid: i for i, cid in enumerate(self.corpus_ids)}

        if score_function not in self.score_functions:
            raise ValueError("score_function must be 'cos_sim' or 'dot'")

        query_ids   = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        self.results = {qid: {} for qid in query_ids}

        q_emb = self.model.encode_queries(
            query_texts, batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor)

        corpus_sorted_ids = sorted(
            corpus, key=lambda k: len(corpus[k].get("title", "") +
                                    corpus[k].get("text", "")), reverse=True)
        corpus_sorted = [corpus[cid] for cid in corpus_sorted_ids]

        result_heaps = {qid: [] for qid in query_ids}
        for start in range(0, len(corpus_sorted), self.corpus_chunk_size):
            end = min(start + self.corpus_chunk_size, len(corpus_sorted))
            sub_emb = self.model.encode_corpus(
                corpus_sorted[start:end], batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor)

            sims = self.score_functions[score_function](q_emb, sub_emb)
            sims[torch.isnan(sims)] = -1

            vals, idxs = torch.topk(
                sims, min(top_k+1, sub_emb.shape[0]), dim=1,
                largest=True, sorted=return_sorted)
            vals, idxs = vals.cpu().tolist(), idxs.cpu().tolist()

            for qi, qid in enumerate(query_ids):
                heap = result_heaps[qid]
                for idx, dscore in zip(idxs[qi], vals[qi]):
                    doc_id = corpus_sorted_ids[start + idx]
                    if len(heap) < top_k:
                        heapq.heappush(heap, (dscore, doc_id))
                    else:
                        heapq.heappushpop(heap, (dscore, doc_id))

        k_rrf = 60     

        for qid, qtext in queries.items():
            q_tok      = self.tokenize(qtext)
            bm25_vec   = self.bm25.get_scores(q_tok)            
            bm25_top   = np.argpartition(bm25_vec, -top_k)[-top_k:]
            bm25_hits  = [(bm25_vec[i], self.corpus_ids[i]) for i in bm25_top]

            dense_hits = list(result_heaps[qid])                 

            all_docs = {doc for _, doc in dense_hits} 

            dense_rank = {doc: r for r, (_, doc) in
                        enumerate(sorted(dense_hits, key=lambda x: x[0], reverse=True), 1)}
            bm25_rank  = {doc: r for r, doc in
                        enumerate(sorted(all_docs,
                                key=lambda d: bm25_vec[id2row[d]], reverse=True), 1)}
            sentinel = 10**9

            fused = []
            for doc in all_docs:
                r_dense = dense_rank.get(doc, sentinel)
                r_bm25  = bm25_rank[doc]
                rrf     = 1/(k_rrf + r_dense) + 1/(k_rrf + r_bm25)
                fused.append((rrf, doc))

            fused.sort(reverse=True)                  # high → low
            self.results[qid] = {doc: s for s, doc in fused[:top_k]}

        return self.results

    def search_bm25(self, 
                corpus: Dict[str, Dict[str, str]], 
                queries: Dict[str, str], 
                top_k: int, 
                score_function: str, 
                return_sorted: bool = False, 
                **kwargs) -> Dict[str, Dict[str, float]]:
        print("Starting BM25-only search...")

        # Tokenize corpus for BM25
        corpus_ids = list(corpus.keys())
        corpus_tokens = [tokenize(doc["text"]) for doc in corpus.values()]

        print("Initializing BM25...")
        bm25 = BM25Okapi(corpus_tokens)

        self.results = {qid: {} for qid in queries.keys()}
        result_data = []  # For storing results in a CSV

        print("Retrieving top-K documents using BM25...")
        for qid, query in queries.items():
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)

            # Get top-K BM25 results
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            top_docs = [(corpus_ids[idx], scores[idx]) for idx in top_k_indices]

            # Store results
            for doc_id, score in top_docs:
                self.results[qid][doc_id] = score
                result_data.append([qid, doc_id, score])

            # Log BM25 retrieval statistics
            print(f"Query {qid}: Retrieved {len(top_docs)} documents.")
            print(f"BM25 score distribution (min: {min(scores)}, max: {max(scores)}, mean: {np.mean(scores):.4f})")

        # # Save results in the required format
        # filename = "/work/pi_wenlongzhao_umass_edu/27/vaishnavisha/CS696DS-Oracle-Retrieving-Code-Explanations/coir-main/results/intfloat/bm25_results.csv"
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score"])
        # df.to_csv(filename, index=False)
        # print(f"BM25-only results saved to {filename}")

        return self.results
    
    def search_bm25PreFilter(self, 
           corpus: Dict[str, Dict[str, str]], 
           queries: Dict[str, str], 
           top_k: int, 
           score_function: str,
           return_sorted: bool = False, 
           **kwargs) -> Dict[str, Dict[str, float]]:
        print("Starting search...")

        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        print("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)

        print("Tokenizing corpus for BM25...")
        corpus_ids = list(corpus.keys())
        corpus_texts = [doc["text"] for doc in corpus.values()]
        corpus_tokens = [tokenize(doc["text"]) for doc in corpus.values()]

        print("Initializing BM25...")
        bm25 = BM25Okapi(corpus_tokens)

        print("Filtering corpus using BM25 top-k results...")
        filtered_corpus = {}
        for qid, query in zip(query_ids, query_texts):
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)
            top_k_indices = np.argsort(scores)[::-1][:top_k]  # Get top-K BM25 results
            filtered_corpus[qid] = {corpus_ids[idx]: corpus[corpus_ids[idx]] for idx in top_k_indices}

        print("Encoding Corpus in batches... Warning: This might take a while!")
        print("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        result_heaps = {qid: [] for qid in query_ids}
        for qid in query_ids:
            top_corpus = filtered_corpus[qid]
            top_corpus_ids = list(top_corpus.keys())
            top_corpus_docs = list(top_corpus.values())

            itr = range(0, len(top_corpus_docs), self.corpus_chunk_size)
            for batch_num, corpus_start_idx in enumerate(itr):
                print(f"Encoding Batch {batch_num + 1}/{len(itr)} for Query {qid}...")
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(top_corpus_docs))

                sub_corpus_embeddings = self.model.encode_corpus(
                    top_corpus_docs[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor=self.convert_to_tensor
                )

                cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
                cos_scores[torch.isnan(cos_scores)] = -1

                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores, min(top_k + 1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted
                )
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]                  
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = top_corpus_ids[sub_corpus_id]
                        if corpus_id != query_id:
                            if len(result_heaps[query_id]) < top_k:
                                heapq.heappush(result_heaps[query_id], (score, corpus_id))
                            else:
                                heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
                
        print("Finalizing results...")
        result_data = []
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
                result_data.append([qid, corpus_id, score])

        df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score"])
        filename = "results/intfloat/retrieval_results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)

        print(f"Retrieval evaluation results saved to {filename}")

        print("Search complete!")
        return self.results
    
    def search_combinedScores(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
               return_sorted: bool = False, 
               **kwargs) -> Dict[str, Dict[str, float]]:

        print("Starting search... --combined_scores")

        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product")

        print("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)

        print("Tokenizing corpus for BM25...")
        corpus_ids = list(corpus.keys())
        corpus_texts = [doc["text"] for doc in corpus.values()]
        corpus_tokens = [tokenize(doc["text"]) for doc in corpus.values()]

        print("Initializing BM25...")
        bm25 = BM25Okapi(corpus_tokens)

        print("Filtering corpus using BM25 top-k results...")
        filtered_corpus = {}
        bm25_scores = {}
        for qid, query in zip(query_ids, query_texts):
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)
            top_k_indices = np.argsort(scores)[::-1][:top_k]  # Get top-K BM25 results
            filtered_corpus[qid] = {corpus_ids[idx]: corpus[corpus_ids[idx]] for idx in top_k_indices}
            bm25_scores[qid] = {corpus_ids[idx]: scores[idx] for idx in top_k_indices}

        print("BM25 Filtering Summary:")
        avg_recall = np.mean([len(docs) for docs in filtered_corpus.values()])
        print(f"Avg. number of docs retrieved per query: {avg_recall}")

        print("Encoding Corpus in batches...")
        result_heaps = {qid: [] for qid in query_ids}
        for qid in query_ids:
            top_corpus = filtered_corpus[qid]
            top_corpus_ids = list(top_corpus.keys())
            top_corpus_docs = list(top_corpus.values())

            itr = range(0, len(top_corpus_docs), self.corpus_chunk_size)
            for batch_num, corpus_start_idx in enumerate(itr):
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(top_corpus_docs))

                sub_corpus_embeddings = self.model.encode_corpus(
                    top_corpus_docs[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor=self.convert_to_tensor
                )

                cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
                cos_scores[torch.isnan(cos_scores)] = -1

                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores, min(top_k + 1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted
                )

                print("Calculating combined scores...")
                # Normalize BM25 and cosine similarity scores
                bm25_score_list = np.array([bm25_scores[qid].get(doc_id, 0) for doc_id in top_corpus_ids])
                cos_score_list = cos_scores.cpu().numpy()

                scaler = MinMaxScaler()
                cos_score_list = scaler.fit_transform(cos_score_list)
                bm25_score_list = scaler.fit_transform(bm25_score_list.reshape(-1, 1)).flatten()

                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]
                    for sub_corpus_id, cos_sim_score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = top_corpus_ids[sub_corpus_id]
                        if corpus_id != query_id:
                            # Combined score with weighted normalization
                            bm25_score = bm25_score_list[sub_corpus_id]
                            combined_score = 0.85 * cos_score_list[query_itr, sub_corpus_id] + 0.25 * bm25_score

                            if len(result_heaps[query_id]) < top_k:
                                heapq.heappush(result_heaps[query_id], (combined_score, corpus_id))
                            else:
                                heapq.heappushpop(result_heaps[query_id], (combined_score, corpus_id))

        print("Finalizing results...")
        result_data = []
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
                result_data.append([qid, corpus_id, score])

        df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score"])
        filename = "results/intfloat/retrieval_results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)

        print(f"Retrieval evaluation results saved to {filename}")
        return self.results
    
    def search_combinedScoresF(self, 
                   corpus: Dict[str, Dict[str, str]], 
                   queries: Dict[str, str], 
                   top_k: int, 
                   score_function: str,
                   return_sorted: bool = False, 
                   **kwargs) -> Dict[str, Dict[str, float]]:

        # Move model to GPU
        self.model.to('cuda')

        print("Starting search... --combined_scores F")

        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product")

        print("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in query_ids]

        # Encode queries in batches and move to GPU
        query_embeddings = self.model.encode_queries(query_texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=False)

        # Convert numpy array to tensor and move to GPU
        query_embeddings_tensor = torch.from_numpy(query_embeddings)
        query_embeddings_on_gpu = query_embeddings_tensor.to('cuda')

        print("Tokenizing corpus for BM25...")
        corpus_ids = list(corpus.keys())
        corpus_texts = [doc["text"] for doc in corpus.values()]
        corpus_tokens = [tokenize(doc["text"]) for doc in corpus.values()]
        bm25 = BM25Okapi(corpus_tokens)

        # Parallelize BM25 filtering
        def filter_corpus(qid, query, bm25, corpus_ids, corpus):
            query_tokens = query.split()
            scores = bm25.get_scores(query_tokens)
            top_k_indices = np.argsort(scores)[::-1][:top_k]
            filtered_corpus = {corpus_ids[idx]: corpus[corpus_ids[idx]] for idx in top_k_indices}
            bm25_scores = {corpus_ids[idx]: scores[idx] for idx in top_k_indices}
            return qid, filtered_corpus, bm25_scores

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(filter_corpus, query_ids, query_texts, [bm25]*len(query_ids), [corpus_ids]*len(query_ids), [corpus]*len(query_ids)))

        filtered_corpus = {}
        bm25_scores = {}
        for qid, filtered, scores in results:
            filtered_corpus[qid] = filtered
            bm25_scores[qid] = scores

        print("BM25 Filtering Summary:")
        avg_recall = np.mean([len(docs) for docs in filtered_corpus.values()])
        print(f"Avg. number of docs retrieved per query: {avg_recall}")

        print("Encoding Corpus in batches...")
        result_heaps = {qid: [] for qid in query_ids}
        for qid in query_ids:
            top_corpus = filtered_corpus[qid]
            top_corpus_ids = list(top_corpus.keys())
            top_corpus_docs = list(top_corpus.values())

            itr = range(0, len(top_corpus_docs), self.corpus_chunk_size)
            for batch_num, corpus_start_idx in enumerate(itr):
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(top_corpus_docs))

                # Encode corpus batch
                sub_corpus_embeddings = self.model.encode_corpus(
                    top_corpus_docs[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=True
                )

                # If it's still a numpy array, convert it to a tensor and move to GPU
                if isinstance(sub_corpus_embeddings, np.ndarray):
                    sub_corpus_embeddings_tensor = torch.from_numpy(sub_corpus_embeddings)
                    sub_corpus_embeddings_on_gpu = sub_corpus_embeddings_tensor.to('cuda')
                else:
                    sub_corpus_embeddings_on_gpu = sub_corpus_embeddings.to('cuda')

                # Compute scores on GPU
                cos_scores = self.score_functions[score_function](query_embeddings_on_gpu, sub_corpus_embeddings_on_gpu)

                cos_scores[torch.isnan(cos_scores)] = -1

                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                    cos_scores, min(top_k + 1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted
                )

                print("Calculating combined scores...")
                # Normalize BM25 and cosine similarity scores
                bm25_score_list = np.array([bm25_scores[qid].get(doc_id, 0) for doc_id in top_corpus_ids])
                cos_score_list = cos_scores.cpu().numpy()

                scaler = MinMaxScaler()
                cos_score_list = scaler.fit_transform(cos_score_list)
                bm25_score_list = scaler.fit_transform(bm25_score_list.reshape(-1, 1)).flatten()

                for query_itr in range(len(query_embeddings)):
                    query_id = query_ids[query_itr]
                    for sub_corpus_id, cos_sim_score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        corpus_id = top_corpus_ids[sub_corpus_id]
                        if corpus_id != query_id:
                            # Combined score with weighted normalization
                            bm25_score = bm25_score_list[sub_corpus_id]
                            combined_score = 0.85 * cos_score_list[query_itr, sub_corpus_id] + 0.25 * bm25_score

                            if len(result_heaps[query_id]) < top_k:
                                heapq.heappush(result_heaps[query_id], (combined_score, corpus_id))
                            else:
                                heapq.heappushpop(result_heaps[query_id], (combined_score, corpus_id))

        print("Finalizing results...")
        result_data = []
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
                result_data.append([qid, corpus_id, score])

        df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score"])
        filename = "results/intfloat/retrieval_results.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)

        print(f"Retrieval evaluation results saved to {filename}")
        return self.results

    def search_bm25PostFilter(self, 
                          corpus: Dict[str, Dict[str, str]], 
                          queries: Dict[str, str], 
                          top_k: int, 
                          score_function: str,
                          return_sorted: bool = False, 
                          **kwargs) -> Dict[str, Dict[str, float]]:
        print("Starting search...")

        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))

        print("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in query_ids]

        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor
        )

        print("Tokenizing corpus for BM25...")
        corpus_ids = list(corpus.keys())
        corpus_texts = [doc["text"] for doc in corpus.values()]
        corpus_tokens = [tokenize(doc["text"]) for doc in corpus.values()]

        print("Initializing BM25...")
        bm25 = BM25Okapi(corpus_tokens)

        print("Encoding Entire Corpus for Cosine Similarity...")
        corpus_embeddings = self.model.encode_corpus(
            corpus_texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor
        )

        print("Computing Cosine Similarity Scores...")
        cos_scores = self.score_functions[score_function](query_embeddings, corpus_embeddings)
        cos_scores[torch.isnan(cos_scores)] = -1  # Handle NaN values

        print("Computing BM25 Scores...")
        bm25_scores = {qid: bm25.get_scores(query.split()) for qid, query in zip(query_ids, query_texts)}

        print("Normalizing BM25 and Cosine Similarity Scores...")
        for qid in query_ids:
            # Convert BM25 scores to a dictionary format {doc_id: score}
            bm25_scores_dict = {cid: bm25_scores[qid][i] for i, cid in enumerate(corpus_ids)}

            # Normalize BM25 scores to range [0,1]
            min_bm25, max_bm25 = min(bm25_scores_dict.values()), max(bm25_scores_dict.values())
            normalized_bm25_scores = {cid: (bm25_scores_dict[cid] - min_bm25) / (max_bm25 - min_bm25 + 1e-8) for cid in corpus_ids}

            # Normalize Cosine Similarity scores to range [0,1]
            query_idx = query_ids.index(qid)  # Get index of query in query_ids
            cos_values = cos_scores[query_idx].cpu().numpy()

            min_cos, max_cos = cos_values.min(), cos_values.max()
            if max_cos - min_cos > 0:  # Avoid division by zero
                normalized_cos_scores = {cid: (cos_values[i] - min_cos) / (max_cos - min_cos + 1e-8) for i, cid in enumerate(corpus_ids)}
            else:
                normalized_cos_scores = {cid: 0.5 for cid in corpus_ids}  # If all scores are the same, assign neutral value

            # Combine Scores with Weights
            combined_scores = {cid: 0.85 * normalized_cos_scores[cid] + 0.15 * normalized_bm25_scores[cid] for cid in corpus_ids}

            # Retrieve Top-K based on combined score
            top_k_results = heapq.nlargest(top_k, combined_scores.items(), key=lambda x: x[1])

            # Store results
            self.results[qid] = {cid: score for cid, score in top_k_results}

        print("Search complete!")
        return self.results
