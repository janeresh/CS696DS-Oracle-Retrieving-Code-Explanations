import re
import pandas as pd
import sys
import ast
import textwrap

def clean_to_function_or_class(text):
    if not isinstance(text, str):
        return ""

    text = textwrap.dedent(text).strip()

    def extract_method_signature(text):
        pattern = re.compile(r'\b(\w+\.\w+\s*\([^)]*\))')
        matches = pattern.findall(text)
        return matches[-1].strip() if matches else ""

    def remove_empty_docstring(code):
        lines = code.splitlines()
        cleaned = []
        skip_next = False
        for i, line in enumerate(lines):
            if i < len(lines) - 1 and re.match(r'^\s*"""\s*"""\s*$', line) and re.match(r'^\s*"""\s*"""\s*$', lines[i+1]):
                skip_next = True
                continue
            elif skip_next:
                skip_next = False
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    try:
        tree = ast.parse(text)
        nodes = [node for node in reversed(tree.body) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        if nodes:
            segment = ast.get_source_segment(text, nodes[0])
            return remove_empty_docstring(segment)
    except (SyntaxError, MemoryError):
        pass

    func_pattern = re.compile(r"^(def\s+\w+\(.*?\):\s*\n(?:[ \t]+.+\n?)*)", re.MULTILINE)
    class_pattern = re.compile(r"^(class\s+\w+.*?:\s*\n(?:[ \t]+.+\n?)*)", re.MULTILINE)

    matches = func_pattern.findall(text)
    if not matches:
        matches = class_pattern.findall(text)

    if matches:
        return remove_empty_docstring(matches[-1].strip())

    return extract_method_signature(text)

input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]

col_no = sys.argv[4]
num_backward_passes = sys.argv[5]

df = pd.read_csv(input_csv)

# Clean each generated column
for row_idx in range(len(df)):
    for i in range(1, col_no + 1):
        for k in range(1, num_backward_passes + 1):
            col = f"Generated_Code_{model}_{i}_code{k}"
            if col in df.columns:
                print(f"Cleaning row {row_idx+1}: {col}")
                try:
                    df.at[row_idx, col] = clean_to_function_or_class(str(df.at[row_idx, col]))
                except Exception as e:
                    print(f"Error cleaning {col} in row {row_idx+1}: {e}")
                    df.at[row_idx, col] = ""

# Save cleaned CSV
df.to_csv(output_csv, index=False)
print(f"\nCleaned file saved to: {output_csv}")
