import json
import logging
from io import StringIO
from typing import Dict, Tuple
import csv
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

class InMemoryDataLoader:
    def __init__(self, corpus_data, query_data, qrels_data):
        self.corpus_file = StringIO('\n'.join(json.dumps(doc) for doc in corpus_data))
        self.query_file = StringIO('\n'.join(json.dumps(query) for query in query_data))
        self.qrels_file = StringIO('\n'.join(f"{qrel['query_id']}\t{qrel['corpus_id']}\t{qrel['score']}" for qrel in qrels_data))
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        print('data loader init')

    def load_custom(self, task_name) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        logger.info("Loading Corpus...")
        self._load_corpus()
        logger.info("Loaded %d Documents.", len(self.corpus))
        logger.info("Doc Example: %s", list(self.corpus.values())[0])

        logger.info("Loading Queries...")
        self._load_queries()

        self._load_qrels()

        self.queries = {qid: self.queries[qid] for qid in self.qrels}
        logger.info("Loaded %d Queries.", len(self.queries))
        logger.info("Query Example: %s", list(self.queries.values())[0])
        #print('4. ',len(self.corpus), len(self.queries), len(self.qrels))

        return self.corpus, self.queries, self.qrels

    def _load_corpus(self):
        self.corpus_file.seek(0)  # Reset the StringIO object to the beginning
        for line in tqdm(self.corpus_file):
            doc = json.loads(line)
            self.corpus[doc["_id"]] = {
                "text": doc.get("text"),
                "title": doc.get("title")
            }

    def _load_queries(self):
        self.query_file.seek(0)  # Reset the StringIO object to the beginning
        for line in self.query_file:
            query = json.loads(line)
            self.queries[query["_id"]] = query.get("text")

    def _load_qrels(self):
        self.qrels_file.seek(0)  # Reset the StringIO object to the beginning
        reader = csv.reader(self.qrels_file, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score

def swap(corpus_data, query_data, qrels_data):
    corpus_list = corpus_data.to_list()
    query_list = query_data.to_list()

    corpus_id_to_idx = {doc['_id']: idx for idx, doc in enumerate(corpus_list)}
    query_id_to_idx = {qry['_id']: idx for idx, qry in enumerate(query_list)}

    for example in qrels_data:
        orig_query_id = example["query_id"]
        orig_corpus_id = example["corpus_id"]

        if orig_query_id in query_id_to_idx and orig_corpus_id in corpus_id_to_idx:
            q_idx = query_id_to_idx[orig_query_id]
            c_idx = corpus_id_to_idx[orig_corpus_id]

            query_list[q_idx]['text'], corpus_list[c_idx]['text'] = (
                corpus_list[c_idx]['text'], query_list[q_idx]['text']
            )

    swapped_corpus_data = Dataset.from_list(corpus_list).cast(corpus_data.features)
    swapped_query_data = Dataset.from_list(query_list).cast(query_data.features)

    return swapped_corpus_data, swapped_query_data

def update_cosqa_qrels_one2Many(corpus_data, query_data, qrels_data):
    code_to_corpus_ids = defaultdict(list)
    for row in corpus_data:
        code_to_corpus_ids[row['text']].append(row['_id'])

    duplicate_codes = {code: ids for code, ids in code_to_corpus_ids.items() if len(ids) > 1}

    corpus_id_to_code = {corpus_id: code for code, corpus_ids in duplicate_codes.items() for corpus_id in corpus_ids}

    corpus_id_to_query_ids = defaultdict(set)
    for row in qrels_data:
        corpus_id = row['corpus_id']
        query_id = row['query_id']
        if corpus_id in corpus_id_to_code:
            corpus_id_to_query_ids[corpus_id].add(query_id)

    final_results = []
    for code, corpus_ids in duplicate_codes.items():
        all_query_ids = set()
        for cid in corpus_ids:
            all_query_ids.update(corpus_id_to_query_ids.get(cid, []))
        final_results.append({
            'code_snippet': code,
            'corpus_ids': corpus_ids,
            'query_ids': list(all_query_ids)
        })

    final_results = sorted(final_results, key=lambda x: len(x['corpus_ids']), reverse=True)

    final_results_df = pd.DataFrame(final_results)

    code_to_corpus_ids = final_results_df.explode('corpus_ids').groupby('code_snippet')['corpus_ids'].apply(set).to_dict()

    new_qrels = []

    for _, row in final_results_df.iterrows():
        query_ids = row['query_ids']         
        code = row['code_snippet']                   
        corpus_ids_with_same_code = code_to_corpus_ids[code]

        for qid in query_ids:
            for cid in corpus_ids_with_same_code:
                new_qrels.append({'query_id': qid, 'corpus_id': cid, 'score': 1})

    new_qrels_dataset = Dataset.from_list(new_qrels)

    updated_qrels = concatenate_datasets([qrels_data, new_qrels_dataset])

    seen_pairs = set()
    filtered_rows = []

    for row in updated_qrels:
        pair = (row['query_id'], row['corpus_id'])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            filtered_rows.append(row)

    updated_qrels = Dataset.from_list(filtered_rows)
    return updated_qrels

def deduplicate_qrels_by_code(corpus_data, qrels_data):
    print('in deduplicate_qrels_by_code')
    code_to_corpus_ids = defaultdict(list)
    for row in corpus_data:
        code_to_corpus_ids[row['text']].append(row['_id'])

    duplicate_codes = {code: ids for code, ids in code_to_corpus_ids.items() if len(ids) > 1}

    corpus_id_replacements = {}
    for code, ids in duplicate_codes.items():
        canonical_id = ids[0] # first occuring as the cid
        for cid in ids:
            corpus_id_replacements[cid] = canonical_id

    updated_qrels = []
    seen = set()
    for row in qrels_data:
        query_id = row['query_id']
        corpus_id = row['corpus_id']

        canonical_id = corpus_id_replacements.get(corpus_id, corpus_id)

        pair = (query_id, canonical_id)
        if pair not in seen:
            seen.add(pair)
            updated_qrels.append({
                'query_id': query_id,
                'corpus_id': canonical_id,
                'score': 1  
            })

    return Dataset.from_list(updated_qrels)

def load_data_from_hf(task_name):
    try:
        queries_corpus_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-queries-corpus")
        qrels_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-qrels")

        print('fetched data from hf')
        corpus_data = queries_corpus_dataset['corpus']
        query_data = queries_corpus_dataset['queries']
        qrels_data_test = qrels_dataset['test']
        if task_name == 'cosqa':
            qrels_data_test = deduplicate_qrels_by_code(corpus_data, qrels_data_test)
        if task_name == "CodeSearchNet-python":            
            qrels_data_train = qrels_dataset['train']
            qrels_data_valid = qrels_dataset['valid']
            qrels_data = concatenate_datasets([qrels_data_train, qrels_data_valid, qrels_data_test])
            corpus_data, query_data = swap(corpus_data, query_data, qrels_data)
        data_loader = InMemoryDataLoader(corpus_data, query_data, qrels_data_test)
        return data_loader.load_custom(task_name)
    except Exception as e:
        logger.error(f"Failed to load data for task {task_name}: {e}")
        return None
    
def load_data_from_hf_train(task_name):
    try:
        queries_corpus_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-queries-corpus")
        qrels_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-qrels")
        print('fetched data from hf train')
        corpus_data = queries_corpus_dataset['corpus']
        query_data = queries_corpus_dataset['queries']
        qrels_data = qrels_dataset['train']

        data_loader = InMemoryDataLoader(corpus_data, query_data, qrels_data)
        return data_loader.load_custom()
    except Exception as e:
        logger.error(f"Failed to load data for task {task_name}: {e}")
        return None

def load_data_from_hf_valid(task_name):
    try:
        queries_corpus_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-queries-corpus")
        qrels_dataset = load_dataset(f"CoIR-Retrieval/{task_name}-qrels")
        print('fetched data from hf valid')
        corpus_data = queries_corpus_dataset['corpus']
        query_data = queries_corpus_dataset['queries']
        qrels_data = qrels_dataset['valid']


        data_loader = InMemoryDataLoader(corpus_data, query_data, qrels_data)
        return data_loader.load_custom()
    except Exception as e:
        logger.error(f"Failed to load data for task {task_name}: {e}")
        return None


def get_tasks(tasks: list, segment="test"):
    all_tasks = {}
    print('in tasks ')

    # Define sub-tasks for special cases
    special_tasks = {
        "codesearchnet": [
            "CodeSearchNet-go", "CodeSearchNet-java", "CodeSearchNet-javascript",
            "CodeSearchNet-ruby", "CodeSearchNet-python", "CodeSearchNet-php"
        ],
        "codesearchnet-ccr": [
            "CodeSearchNet-ccr-go", "CodeSearchNet-ccr-java", "CodeSearchNet-ccr-javascript",
            "CodeSearchNet-ccr-ruby", "CodeSearchNet-ccr-python", "CodeSearchNet-ccr-php"
        ]
    }


    # for task in tasks:
    #     if task in special_tasks:
    #         for sub_task in special_tasks[task]:
    #             task_data = load_data_from_hf(sub_task)
    #             if task_data is not None:
    #                 all_tasks[sub_task] = task_data
    #     else:
    #         task_data = load_data_from_hf(task)
    #         if task_data is not None:
    #             all_tasks[task] = task_data
    
    for task in tasks:
        if segment == "test":
            task_data = load_data_from_hf(task)
            if task_data is not None:
                all_tasks[task] = task_data
        elif segment == "train": 
            task_data = load_data_from_hf_train(task)
            if task_data is not None:
                all_tasks[task] = task_data
        else:
            task_data = load_data_from_hf_valid(task)
            if task_data is not None:
                all_tasks[task] = task_data

    return all_tasks