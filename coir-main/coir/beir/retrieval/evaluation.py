import pytrec_eval
import logging
from typing import List, Dict, Tuple
from .search.base import BaseSearch
from .custom_metrics import mrr, recall_cap, hole, top_k_accuracy
from collections import defaultdict
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger(__name__)

class EvaluateRetrieval:
    
    def __init__(self, retriever: BaseSearch = None, k_values: List[int] = [1,3,5,10,100,1000], score_function: str = "cos_sim"):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function
        #self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        print('in beir/retrieval/evaluation.py: loading up search\n')
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)
    
    def rerank(self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int) -> Dict[str, Dict[str, float]]:
    
        new_corpus = {}
    
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]
                    
        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    def rerank_rrf(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int,
               rrf_k: int = 60) -> Dict[str, Dict[str, float]]:
        """
        Enhanced reranking using Reciprocal Rank Fusion (RRF).

        rrf_k: Controls the impact of lower-ranked documents. Higher means lower-ranked docs contribute less.
        """

        reranked_results = defaultdict(dict)

        for query_id, doc_scores in results.items():
            # Sort docs by original score and take top-k
            top_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

            # RRF score aggregation
            rrf_scores = defaultdict(float)
            for rank, (doc_id, _) in enumerate(top_docs, start=1):
                # Reciprocal Rank Fusion formula: RRF = 1 / (rank + rrf_k)
                rrf_scores[doc_id] += 1 / (rank + rrf_k)

            # Sort by final RRF scores
            reranked_results[query_id] = {doc_id: score for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)}

        return reranked_results
    
#     def rerank_cross_encoder(self, 
#                          corpus: Dict[str, Dict[str, str]], 
#                          queries: Dict[str, str],
#                          results: Dict[str, Dict[str, float]],
#                          top_k: int) -> Dict[str, Dict[str, float]]:

#         reranked_results = {}

#         for query_id, doc_scores in results.items():
#             # Get the top-k docs from initial retrieval
#             top_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

#             pairs = [(queries[query_id], corpus[doc_id]['text']) for doc_id, _ in top_docs]

#             # Batch predict scores for efficiency
#             scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)

#             reranked = sorted(zip([doc_id for doc_id, _ in top_docs], scores), key=lambda x: x[1], reverse=True)
#             reranked_results[query_id] = {doc_id: score for doc_id, score in reranked}

#         return reranked_results



    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        print('in beir/retrieval/evaluation.py: evaluating now\n')
        if ignore_identical_ids:
            logger.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision
    
    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
        
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        
        elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)
