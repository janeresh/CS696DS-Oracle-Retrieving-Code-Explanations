# Query Expansion
## Install Requirements

```bash
pip install -r expansion_requirements.txt
```

## Query Expansion with Wordnet UnFiltered
Expand queries using synonyms from WordNet without filtering:
```bash
python nltk_code/nltk_unfiltered_query_exp.py
```

## Query Expansion with Wordnet Filtered
Expand queries using WordNet and filter out unrelated non-code terms:
```bash
python nltk_code/nltk_filtered_query_exp.py
```

## Query Expansion with LangChain
Use an LLM via LangChain to generate query expansions:
```bash
python langchain_code/llm_langchain_query_exp.py
```
### Running on SLURM
To run the LangChain+LLM job as a SLURM batch job:
```bash
sbatch langchain_code/langchain_sbatch.sh
```

## Query Expansion with Local LLM (vLLM)
This method uses a locally hosted large language model via vLLM to expand queries without requiring any cloud APIs or LangChain.

### Prerequisites
- A compatible GPU (e.g., A100 / L40S with ≥40GB VRAM)
- A local model path (e.g., DeepSeek, Granite)
- Python 3.8+

### Run LLM-Based Query Expansion for a Small Dataset
Run the query expansion script directly using a local LLM:

```bash
python llm_based_code/query_expansion_llm_vllm.py <model_name> <input_csv> <output_csv>
```

### Run LLM-Based Query Expansion for a Large Dataset
#### Step 1: Split the Input CSV
If your dataset is large, split it into smaller CSV shards (e.g., 5000 rows per file):
**Note:** Add the input_csv, output_csv and shard_no in the script.

```bash
python ../utils/split_csv.py 
```

#### Step 2: Batch Job with SLURM 
To run the expansion in parallel across multiple shards/subcsv and GPUs:
**Note:** Modify the input CSV, output CSV and model name in the script.

```bash
sbatch llm_based_code/qa_expand.sh 
```

#### Step 3: Merge All Result CSVs
After all shards are processed, merge the outputs into one file: 
**Note:** Add the input_csv and output_csv in the script.

```bash
python ../utils/merge_csv.py <input_csv> <output_csv>
```
