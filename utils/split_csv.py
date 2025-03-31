import pandas as pd
import os

input_csv = "/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/results/exps_generated_code_results.csv"
output_dir = "/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/results/exps_generated_code_results/"

os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(input_csv)
total_rows = len(df)

# Number of parts to split into
num_splits = 4
rows_per_file = total_rows // num_splits

# Split and save parts
for i in range(num_splits):
    start_idx = i * rows_per_file
    end_idx = start_idx + rows_per_file if i < num_splits - 1 else total_rows  # Ensure last file gets all remaining rows
    df_split = df.iloc[start_idx:end_idx]
    output_file = os.path.join(output_dir, f"split_part_{i}.csv")
    df_split.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

print("\nCSV splitting completed.")
