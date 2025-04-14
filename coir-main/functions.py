from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import pandas as pd

def update_corpus_with_cleaned_code(corpus: dict, df: pd.DataFrame, id_col='corpus_id', code_col='cleaned_code') -> dict:
    updated_corpus = corpus.copy()
    
    for _, row in df.iterrows():
        doc_id = row[id_col]
        new_text = row[code_col]

        if pd.notna(new_text) and doc_id in updated_corpus:
            updated_corpus[doc_id]['text'] = new_text

    return updated_corpus

def load_data(dataset_name):
    tasks = get_tasks(tasks=[dataset_name])
    return tasks

def add_expl(tasks, dataset_name, explanation_df_path, col_name='explanation_deepseek_1_cleaned'):
    corpus, queries, qrels = tasks[dataset_name]
    expl_df = pd.read_csv(explanation_df_path)
    expl_df.rename(columns={"query_id": "query-id", "corpus_id": "corpus-id"}, inplace=True)
    for _, row in expl_df.iterrows():
        corpus_id = row['corpus-id']
        explanation = row[col_name]

        if corpus_id in corpus and explanation and explanation.strip():
            corpus[corpus_id]['text'] = explanation  

    corpus = {doc_id: doc for doc_id, doc in corpus.items() if 'text' in doc and doc['text'].strip()}
    print(corpus['d1'])
    tasks[dataset_name] = (corpus, queries, qrels)

    print(f"Total docs in corpus after replacement: {len(tasks[dataset_name][0])}")
    missing_text = sum(1 for doc in tasks[dataset_name][0].values() if not doc.get('text'))
    print(f"Number of docs still missing 'text': {missing_text}")
    
def add_expl_2queries(tasks, dataset_name, explanation_df_path, col_name='explanation_deepseek_1_cleaned'):
    corpus, queries, qrels = tasks[dataset_name]
    expl_df = pd.read_csv(explanation_df_path)
    expl_df.rename(columns={"query_id": "query-id", "corpus_id": "corpus-id"}, inplace=True)
    for _, row in expl_df.iterrows():
        query_id = row['query-id']
        explanation = row[col_name]

        if query_id in queries and explanation and explanation.strip():
            queries[query_id] = explanation  

    queries = {doc_id: doc for doc_id, doc in queries.items()}
    print(queries['q265734'])
    tasks[dataset_name] = (corpus, queries, qrels)

    print(f"Total docs in q after replacement: {len(tasks[dataset_name][1])}")
    
def run(model_name, tasks, llm_name, retrieval_name, dataset_name):
    model = YourCustomDEModel(model_name=model_name)

    evaluation = COIR(tasks=tasks,batch_size=256)
    if(retrieval_name == "bm25"):
        results = evaluation.run(model, output_folder=f"results/{dataset_name}/{llm_name}/{retrieval_name}")
    else:
        results = evaluation.run(model, output_folder=f"results/{dataset_name}/{llm_name}/{retrieval_name}/{model_name}")
    print(results)