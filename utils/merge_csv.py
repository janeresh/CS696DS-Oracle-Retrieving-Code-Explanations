import pandas as pd
import glob
import os



input_dir = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/cosqa_queries_expanded_granite/"
output_file = "/work/pi_wenlongzhao_umass_edu/27/janet/query_expansion/results/cosqa_queries_expanded_granite.csv"

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

#  Check if CSV files exist
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_csv(output_file, index=False)
