import os
import string
import nltk
import pandas as pd
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize

# === Set custom NLTK data path ===
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# === Download necessary resources to that path ===
# nltk.download('punkt', download_dir=nltk_data_path)
# nltk.download('stopwords', download_dir=nltk_data_path)
# nltk.download('wordnet', download_dir=nltk_data_path)

# === Load stopwords ===
stop_words = set(stopwords.words("english"))

# === Load input CSV ===
input_path = "/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/CoSQA_explanations_query_code.csv"  # Update this if needed
df = pd.read_csv(input_path)

# === Function to get WordNet synonyms (limit 3 per word) ===
def get_synonyms(word, limit=3):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name != word:
                synonyms.add(name)
            if len(synonyms) >= limit:
                return list(synonyms)
    return list(synonyms)

# === Expand queries ===
expanded_data = []

for _, row in df.iterrows():
    query_id = row['query_id']
    query = row['doc'].lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(query)
    words = [w for w in words if w not in stop_words]

    synonyms = []
    for word in words:
        synonyms += get_synonyms(word, limit=3)

    expanded_data.append({
        "id": query_id,
        "original_query": row['doc'],
        "expanded_query": ' '.join(synonyms)
    })

# === Save to output CSV ===
output_path = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/queries_expanded_nltk.csv"
pd.DataFrame(expanded_data).to_csv(output_path, index=False)
print(f"✅ Expanded queries saved to {output_path}")
