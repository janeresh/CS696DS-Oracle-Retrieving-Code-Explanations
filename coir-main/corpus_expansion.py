import pandas as pd

def update_corpus(corpus: dict, df: pd.DataFrame, id_col='corpus_id', expl_col='explanation') -> dict:
    updated_corpus = {}

    for _, row in df.iterrows():
        corpus_id, explanation = row[id_col], row[expl_col]
        updated_corpus[corpus_id] = {
            'text': explanation,
            'title': corpus.get(corpus_id.split('.')[0], {}).get('title', '')
        }

    return updated_corpus

def expand_corpus(corpus, df):
    # explanation_cols = [
    #     'explanation_granite_1_cleaned',
    #     'explanation_granite_2_cleaned',
    #     'explanation_granite_3_cleaned',
    #     'explanation_granite_4_cleaned',
    #     'explanation_granite_5_cleaned'
    # ]

    rows = []

    for i in range(1, 6): 
        for idx, row in df.iterrows():
            corpus_id = f"{row['corpus_id']}.{i}"
            explanation = row[f'explanation_granite_{i}_cleaned']
            rows.append({"corpus_id": corpus_id, "explanation": explanation})


    df_final = pd.DataFrame(rows)
    df_final = df_final.sort_values(by=['corpus_id'])
    df_final.reset_index(inplace=True, drop=True)
    updated_corpus = update_corpus(corpus, df_final)
    return updated_corpus

