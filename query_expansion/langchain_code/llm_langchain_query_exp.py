import os
import sys
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Parse CLI Arguments ===
model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
port = sys.argv[2] if len(sys.argv) > 2 else "8000"
API_BASE = f"http://localhost:{port}/v1"

print(f"Using model: {model_name}, API: {API_BASE}")

# === LangChain ChatOpenAI Wrapper ===
llm = ChatOpenAI(
    model_name=f"{model_name}-chat",
    openai_api_base=API_BASE,
    openai_api_key="dummy-key",
    temperature=0.4
)

def clean_lines(lines):
    return [
        line.strip().lstrip("0123456789).:- ").strip('" ')
        for line in lines if line.strip()
    ]

def expand_query_with_deepseek(query, llm, num_expansions=5):
    system_msg = SystemMessage(content=(
        "You are a query expansion engine specialized in programming queries. "
        "Return a list of clean, short, semantically similar variants of the input query. "
        "Do not include any explanation, preambles, or list numbers. "
        "Keep programming terms like 'python', 'list', 'json', etc. unchanged. "
        "Each expansion must be a standalone query."
    ))

    user_msg = HumanMessage(content=(
        f"Expand this query into {num_expansions} semantically similar variants:\n"
        f"{query}\n\n"
        "Only return the expanded queries, one per line, no numbering or bullet points."
    ))

    try:
        response = llm([system_msg, user_msg])
        return clean_lines(response.content.splitlines())
    except Exception as e:
        print(f"Error expanding query '{query}': {e}")
        return []

def rank_queries(expanded_queries, original_query, top_k=5):
    corpus = [original_query] + expanded_queries
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
    ranked = sorted(zip(expanded_queries, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# === File Paths ===
input_csv = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/data/CoSQA_explanations_query_code.csv"
output_csv = f"/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/code_queries_{model_name}_expanded.csv"

df = pd.read_csv(input_csv).iloc[:10]
results = []

# === Expand and Rank ===
for _, row in df.iterrows():
    qid = row['query_id']
    query = row['doc']

    try:
        expanded = expand_query_with_deepseek(query, llm, num_expansions=10)
        ranked = rank_queries(expanded, query, top_k=5)

        for rank, (exp_q, score) in enumerate(ranked, 1):
            results.append({
                "id": qid,
                "original_query": query,
                "rank": rank,
                "expanded_query": exp_q,
                "similarity": round(score, 4)
            })
    except Exception as e:
        print(f"Error expanding query ID {qid}: {e}")

# === Save ===
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Done! Results saved to {output_csv}")
