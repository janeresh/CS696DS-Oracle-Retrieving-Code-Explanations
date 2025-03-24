import re
import pandas as pd
from evaluation import compute_metrics,compute_common_scores
import sys

def clean_to_function_only(text):
    if not isinstance(text, str):
        return ""
    pattern = re.compile(r"^(def .+?:\n(?:\s{4}.*\n?)*)", re.MULTILINE)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else ""


input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]

df=pd.read_csv(input_csv)
result=[]
for i, row in df.iterrows():
    row[f"Generated_Code_{model}_1"]=clean_to_function_only(row[f"Generated_Code_{model}_1"])
    row[f"Generated_Code_{model}_2"]=clean_to_function_only(row[f"Generated_Code_{model}_2"])
    row[f"Generated_Code_{model}_3"]=clean_to_function_only(row[f"Generated_Code_{model}_3"])
    row[f"Generated_Code_{model}_4"]=clean_to_function_only(row[f"Generated_Code_{model}_4"])
    row1 = compute_metrics(row, model)
    result.append(row1)

pd.DataFrame(result).to_csv(output_csv)
