import pandas as pd
from vllm import LLM
from vllm.sampling_params import SamplingParams


df = pd.read_csv('/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/CS696DS-Oracle-Retrieving-Code-Explanations/Explanation_Generation/Cosqa/raw_data/cosqa_queries_code_corpus.csv')

# 2. Initialize vLLM client once
client = LLM(model="/datasets/ai/llama3/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6", dtype="half")

# 3. Define deterministic sampling
params = SamplingParams(
    max_tokens=2,
    temperature=0.0,
    top_p=1.0
    )

# 4. Row-wise function that builds the prompt, calls vLLM, and returns “yes”/“no”
def impl_check(row):
    prompt = f"""
    Docstring:
    {row['doc']}

    Code:
    {row['code']}

    Question: Does the code implement the behavior described in the docstring? Answer “yes” or “no” only.
    Answer:\n
    """

    response = client.generate([{"prompt": prompt}], sampling_params=params)
    # grab the single-token reply, strip whitespace, force lowercase
    return response[0].outputs[0].text.strip().lower()

# 5. Apply it across your DataFrame
df['validation_flag'] = df.apply(impl_check, axis=1)

df.to_csv('/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/CS696DS-Oracle-Retrieving-Code-Explanations/Explanation_Generation/Cosqa/pre_processing_COSQA/COSQA_validation_query_code.csv')