import pandas as pd
import glob
import os



input_dir = "/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/metrics/deepseek_struct_metrics_results_dir"
output_file = "/work/pi_wenlongzhao_umass_edu/27/janet/validation_tool/RTC/results/explanations_4_codes_3/metrics/deepseek_struct_metrics_results.csv"

csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

#  Check if CSV files exist
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_csv(output_file, index=False)
