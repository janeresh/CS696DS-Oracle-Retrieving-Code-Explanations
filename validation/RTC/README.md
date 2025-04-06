# Validation using RTC

This repository provides simple yet powerful scripts for expanding natural language queries using **NLTK** and **WordNet**. Query expansion is essential for improving search quality, information retrieval, and downstream NLP tasks.


## Install Requirements

```bash
pip install -r validate_requirements.txt
```

## NLTK

### Wordnet UnFiltered
Run the wordnet based query expansion file
```bash
python nltk_unfiltered_query_exp.py
```

### Wordnet Filtered
Run the wordnet based query expansion file and applying filter for code related words.
```bash
python nltk_filtered_query_exp.py
```

### LangChain + LLM
Run the LangChain + LLM query expansion file.
```bash
python llm_langchain_query_exp.py
```

We can run this as a SBatch Job by using the below file

```bash
python langchain_sbatch.sh
```