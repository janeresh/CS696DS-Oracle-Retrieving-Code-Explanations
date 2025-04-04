import os
import pandas as pd
import string
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Setup ===
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Programming terms to keep intact
tech_terms = {"python", "java", "list", "array", "dict", "json", "csv", "sql", "string", "int", "file", "loop", "variable", "function", "pandas", "numpy", "query", "dataframe"}

def get_synonyms(word, limit=3):
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name != word:
                syns.add(name)
            if len(syns) >= limit:
                return list(syns)
    return list(syns)

def expand_query(query, max_syns=3):
    query = query.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(query)
    filtered = [t for t in tokens if t not in stop_words]

    expansions = set([query])
    for word in filtered:
        if word in tech_terms:
            continue  # preserve technical terms
        synonyms = get_synonyms(word, max_syns)
        for syn in synonyms:
            new_query = query.replace(word, syn)
            expansions.add(new_query)
    return list(expansions)

def rank_queries(expanded_queries, original_query, top_k=5):
    corpus = [original_query] + expanded_queries
    tfidf = TfidfVectorizer().fit_transform(corpus)
    orig_vec = tfidf[0]
    sims = cosine_similarity(orig_vec, tfidf[1:])[0]
    ranked = sorted(zip(expanded_queries, sims), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# === Main ===
input_file = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/data/CoSQA_explanations_query_code.csv"
output_file = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/code_queries_expanded2.csv"
df = pd.read_csv(input_file)
results = []

for _, row in df.iterrows():
    qid = row['query_id']
    query = row['doc']
    expanded = expand_query(query)
    ranked = rank_queries(expanded, query, top_k=5)

    for rank, (q, score) in enumerate(ranked, 1):
        results.append({
            "id": qid,
            "original_query": query,
            "rank": rank,
            "expanded_query": q,
            "similarity": round(score, 4)
        })

# Save to CSV
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"✅ Saved top 5 expanded queries per input to: {output_file}")
