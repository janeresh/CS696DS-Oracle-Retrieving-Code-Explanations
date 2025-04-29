import pytrec_eval
import logging
from typing import List, Dict, Tuple
from .search.base import BaseSearch
from .custom_metrics import mrr, recall_cap, hole, top_k_accuracy
import json
from collections import defaultdict
from sentence_transformers.cross_encoder import CrossEncoder
import pandas as pd
from statistics import median
import os 

logger = logging.getLogger(__name__)

def combine_scores_by_average(results, top_k=1000):
    print('in combine scores by average')
    grouped = defaultdict(lambda: defaultdict(list))  # {q_prefix: {corpus_id: [scores]}}

    for qid, corpus_scores in results.items():
        q_prefix = qid.split('.')[0]
        for cid, score in corpus_scores.items():
            grouped[q_prefix][cid].append(score)

    final_results = {}

    for q_prefix, cid_scores in grouped.items():
        freq_counter = {cid: len(scores) for cid, scores in cid_scores.items()}
        averaged = {cid: sum(scores)/len(scores) for cid, scores in cid_scores.items()}
        top_k_cids = sorted(freq_counter.items(), key=lambda x: (-x[1], -averaged[x[0]]))[:top_k]

        # Formatted as {q_prefix: {cid: score, ...}}
        final_results[q_prefix] = {cid: round(averaged[cid], 4) for cid, _ in top_k_cids}

    return final_results

def combine_scores_by_median(results, top_k=1000):
    print('in combine scores by median')
    grouped = defaultdict(lambda: defaultdict(list))  # {q_prefix: {corpus_id: [scores]}}

    for qid, corpus_scores in results.items():
        q_prefix = qid.split('.')[0]
        for cid, score in corpus_scores.items():
            grouped[q_prefix][cid].append(score)

    final_results = {}

    for q_prefix, cid_scores in grouped.items():
        freq_counter = {cid: len(scores) for cid, scores in cid_scores.items()}
        median_scores = {cid: median(scores) for cid, scores in cid_scores.items()}
        top_k_cids = sorted(freq_counter.items(), key=lambda x: (-x[1], -median_scores[x[0]]))[:top_k]

        # Formatted as {q_prefix: {cid: score, ...}}
        final_results[q_prefix] = {cid: round(median_scores[cid], 4) for cid, _ in top_k_cids}

    return final_results

class EvaluateRetrieval:
    
    def __init__(self, retriever: BaseSearch = None, k_values: List[int] = [1,3,5,10,100,1000], score_function: str = "cos_sim"):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function

        #self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str],fileName:str, **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        print('in beir/retrieval/evaluation.py: loading up search\n')
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)
        #return self.retriever.search_dres_sparse(corpus, queries, self.top_k, self.score_function, fileName, **kwargs)
    
 
    @staticmethod
    def evaluate_eff(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool = True,
                 save_results: bool = True,
                 filename: str = "/work/pi_wenlongzhao_umass_edu/27/vaishnavisha/CS696DS-Oracle-Retrieving-Code-Explanations/coir-main/results/retrieval_evaluation.csv") -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        import numpy as np
        from collections import defaultdict

        print('in beir/retrieval/evaluation.py: evaluating now\n')

        if ignore_identical_ids:
            logger.info('Ignoring identical query and document IDs...')
            for qid, rels in results.items():
                results[qid] = {pid: score for pid, score in rels.items() if qid != pid}
        
        results = combine_scores_by_average(results, max(k_values))

        ndcg, _map, recall, precision = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

        map_string = "map_cut." + ",".join(map(str, k_values))
        ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
        recall_string = "recall." + ",".join(map(str, k_values))
        precision_string = "P." + ",".join(map(str, k_values))

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        result_data = [
            [qid, doc_id, score, qrels.get(qid, {}).get(doc_id, 0)]
            for qid in scores.keys()
            for doc_id, score in sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        ]

        total_queries = len(scores)
        for k in k_values:
            for qid in scores.keys():
                ndcg[f"NDCG@{k}"] += scores[qid].get(f"ndcg_cut_{k}", 0.0)
                _map[f"MAP@{k}"] += scores[qid].get(f"map_cut_{k}", 0.0)
                recall[f"Recall@{k}"] += scores[qid].get(f"recall_{k}", 0.0)
                precision[f"P@{k}"] += scores[qid].get(f"P_{k}", 0.0)

            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / total_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / total_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / total_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / total_queries, 5)

        if save_results:
            df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score", "ground_truth_relevance"])
            os.makedirs(filename, exist_ok=True)
            df.to_csv(os.path.join(filename, "retrieval_evaluation.csv"), index=False)
            print(f"Retrieval evaluation results saved to {filename}")

        return dict(ndcg), dict(_map), dict(recall), dict(precision)

    
    
    @staticmethod

    def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool = True,
                 save_results: bool = True,
                 filename: str = "/work/pi_wenlongzhao_umass_edu/27/vaishnavisha/CS696DS-Oracle-Retrieving-Code-Explanations/coir-main/results/retrieval_evaluation.csv") -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

        print('in beir/retrieval/evaluation.py: evaluating now\n')

        if ignore_identical_ids:
            logger.info('Ignoring identical query and document IDs...')
            for qid, rels in results.items():
                results[qid] = {pid: score for pid, score in rels.items() if qid != pid}
        results = combine_scores_by_average(results, max(k_values))
        
        # os.makedirs(filename, exist_ok=True)
        # path = filename + '/results.json'
        # with open(path, 'w') as f:
        #     json.dump(results, f, indent=2)
        
        ndcg, _map, recall, precision = {}, {}, {}, {}

        for k in k_values:
            ndcg[f"NDCG@{k}"], _map[f"MAP@{k}"], recall[f"Recall@{k}"], precision[f"P@{k}"] = 0.0, 0.0, 0.0, 0.0

        map_string = "map_cut." + ",".join(map(str, k_values))
        ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
        recall_string = "recall." + ",".join(map(str, k_values))
        precision_string = "P." + ",".join(map(str, k_values))

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        # os.makedirs(filename, exist_ok=True)
        # path = filename + '/scores.json'
        # with open(path, 'w') as f:
        #     json.dump(scores, f, indent=2)
        result_data = []  

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
                _map[f"MAP@{k}"] += scores[query_id][f"map_cut_{k}"]
                recall[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
                precision[f"P@{k}"] += scores[query_id][f"P_{k}"]

            for doc_id, score in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True):
                relevance = qrels.get(query_id, {}).get(doc_id, 0)  # Get ground truth relevance (1 or 0)
                result_data.append([query_id, doc_id, score, relevance])

        total_queries = len(scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / total_queries, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / total_queries, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / total_queries, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / total_queries, 5)

        # Save results for further analysis
        if save_results:
            df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score", "ground_truth_relevance"])
            os.makedirs(filename, exist_ok=True)
            df.to_csv(os.path.join(filename, "retrieval_evaluation.csv"), index=False)
            print(f"Retrieval evaluation results saved to {filename}")

        return ndcg, _map, recall, precision
    
    
    @staticmethod
    
    def evaluate_grouped(qrels: Dict[str, Dict[str, int]], 
                     results: Dict[str, Dict[str, float]], 
                     k_values: List[int],
                     ignore_identical_ids: bool = True,
                     save_results: bool = True,
                     filename: str = "/work/pi_wenlongzhao_umass_edu/27/vaishnavisha/CS696DS-Oracle-Retrieving-Code-Explanations/coir-main/results/retrieval_evaluation.csv"
                    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    
        print('Grouped evaluation based on query prefixes...\n')

        from collections import defaultdict
        import os
        import json
        import pandas as pd
        import pytrec_eval

        if ignore_identical_ids:
            print('Ignoring identical query and document IDs...')
            for qid, rels in results.items():
                results[qid] = {pid: score for pid, score in rels.items() if qid != pid}

        # Evaluation strings
        map_string = "map_cut." + ",".join(map(str, k_values))
        ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
        recall_string = "recall." + ",".join(map(str, k_values))
        precision_string = "P." + ",".join(map(str, k_values))

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        raw_scores = evaluator.evaluate(results)

        # Save raw scores
        os.makedirs(filename, exist_ok=True)
        raw_score_path = filename + '/retrieval_scores.json'
        with open(raw_score_path, 'w') as f:
            json.dump(raw_scores, f, indent=2)

        # Group by query prefix
        prefix_scores = defaultdict(lambda: defaultdict(float))
        prefix_counts = defaultdict(int)

        for qid, metrics in raw_scores.items():
            prefix = qid.split('.')[0]
            for metric_name, value in metrics.items():
                prefix_scores[prefix][metric_name] += value
            prefix_counts[prefix] += 1

        # Compute average metrics per prefix
        averaged_scores = {}
        for prefix, metrics in prefix_scores.items():
            count = prefix_counts[prefix]
            averaged_scores[prefix] = {metric: round(value / count, 6) for metric, value in metrics.items()}

        # Save averaged scores
        os.makedirs(filename, exist_ok=True)
        averaged_score_path = filename + '/retrieval_averaged_scores.json'
        with open(averaged_score_path, 'w') as f:
            json.dump(averaged_scores, f, indent=2)

        # Initialize metrics
        ndcg, _map, recall, precision = {}, {}, {}, {}
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        # Use averaged scores for metrics
        for prefix, metrics in averaged_scores.items():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += metrics.get(f"ndcg_cut_{k}", 0.0)
                _map[f"MAP@{k}"] += metrics.get(f"map_cut_{k}", 0.0)
                recall[f"Recall@{k}"] += metrics.get(f"recall_{k}", 0.0)
                precision[f"P@{k}"] += metrics.get(f"P_{k}", 0.0)

        total_prefixes = len(averaged_scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / total_prefixes, 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / total_prefixes, 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / total_prefixes, 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / total_prefixes, 5)

        # Build result_data from raw scores (preserve original qid + doc_ids)
        result_data = []
        for qid in results:
            for doc_id, score in sorted(results[qid].items(), key=lambda x: x[1], reverse=True):
                relevance = qrels.get(qid, {}).get(doc_id, 0)
                result_data.append([qid, doc_id, score, relevance])

        # Save CSV
        if save_results:
            df = pd.DataFrame(result_data, columns=["query_id", "retrieved_doc_id", "score", "ground_truth_relevance"])
            os.makedirs(filename, exist_ok=True)
            df.to_csv(filename+'/retrieval_evaluation.csv', index=False)
            print(f"Retrieval evaluation results saved to {filename}")
        
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
    
    def rerank_cross_encoder(self, 
                         corpus: Dict[str, Dict[str, str]], 
                         queries: Dict[str, str],
                         results: Dict[str, Dict[str, float]],
                         top_k: int) -> Dict[str, Dict[str, float]]:

        reranked_results = {}

        for query_id, doc_scores in results.items():
            top_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

            pairs = [(queries[query_id], corpus[doc_id]['text']) for doc_id, _ in top_docs]

            scores = self.cross_encoder.predict(pairs, batch_size=self.batch_size)

            reranked = sorted(zip([doc_id for doc_id, _ in top_docs], scores), key=lambda x: x[1], reverse=True)
            reranked_results[query_id] = {doc_id: score for doc_id, score in reranked}

        return reranked_results


   