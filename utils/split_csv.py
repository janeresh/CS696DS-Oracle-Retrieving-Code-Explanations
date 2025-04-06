import pandas as pd
import os

input_csv = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/data/CoSQA_explanations_query_code.csv"
output_dir = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/cosqa_queries_expanded/"
rows_per_shard = 5000

os.makedirs(output_dir, exist_ok=True)

df_iter = pd.read_csv(input_csv, chunksize=rows_per_shard)

for i, chunk in enumerate(df_iter):
    out_path = os.path.join(output_dir, f"part_{i}.csv")
    chunk.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

print("\nCSV splitting completed.")

