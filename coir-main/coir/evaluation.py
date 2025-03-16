import os
import json
import logging
from coir.beir.retrieval.evaluation import EvaluateRetrieval
from coir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


logger = logging.getLogger(__name__)


class COIR:
    def __init__(self, tasks, batch_size):
        self.tasks = tasks
        self.batch_size = batch_size
        print('COIR init')

    def run(self, model, output_folder: str):
        results = {}
        for task_name, task_data in self.tasks.items():
            output_file = os.path.join(output_folder, f"{task_name}.json")

            # Check if the output file already exists
            if os.path.exists(output_file):
                print(f"Results for {task_name} already exist. Skipping task.")
                continue

            corpus, queries, qrels = task_data

            # Initialize custom model
            print('in evaluation.py: loading up dres\n')
            custom_model = DRES(model, batch_size=self.batch_size)
            print('in evaluation.py: loading up retriever\n')
            retriever = EvaluateRetrieval(custom_model, score_function="cos_sim")
            print(f"Batch size: {self.batch_size}")
            # Retrieve results
            # print('in evaluation.py: retrieving\n')
            # task_results = retriever.retrieve(corpus, queries)
            
            print('in evaluation.py: retrieving\n')
            initial_results = retriever.retrieve(corpus, queries)

            # Rerank the results
            # print('in evaluation.py: reranking\n')
            # reranked_results = retriever.rerank_rrf(corpus, queries, initial_results, top_k=10)


            # Evaluate results
            print('in evaluation.py: evaluating\n')
            ndcg, map, recall, precision = retriever.evaluate(qrels, initial_results, retriever.k_values)
            metrics = {
                "NDCG": ndcg,
                "MAP": map,
                "Recall": recall,
                "Precision": precision
            }

            # Save results
            os.makedirs(output_folder, exist_ok=True)
            with open(output_file, 'w') as json_file:
                json.dump({"metrics": metrics}, json_file, indent=4)

            logger.info(f"Results for {task_name} saved to {output_folder}")
            results[task_name] = metrics

        return results