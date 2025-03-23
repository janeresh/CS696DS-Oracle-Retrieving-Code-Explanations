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

def tokenize(text):
    return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]


logger = logging.getLogger(__name__)

# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
    
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
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
          
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)
        
        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
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
        
        return self.results 
    

    def search_bm25(self, 
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
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        print("Search complete!")
        return self.results


  