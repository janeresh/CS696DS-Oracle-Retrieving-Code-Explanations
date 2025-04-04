import ast
import tokenize
from io import StringIO
import textwrap
import pandas as pd
import sys

def extract_docstrings(source):
    docstrings = []
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], source

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = ast.get_docstring(node, clean=False)
            if doc and isinstance(node.body[0], ast.Expr):
                first_line = node.body[0].lineno - 1
                last_line = node.body[0].end_lineno
                docstrings.append('\n'.join(lines[first_line:last_line]))
                for i in range(first_line, last_line):
                    lines[i] = ''
    cleaned_code = '\n'.join(line for line in lines if line.strip())
    return docstrings, cleaned_code

def extract_top_level_comments(source):
    comments = []
    tokens = tokenize.generate_tokens(StringIO(source).readline)
    for token_type, token_string, start, _, _ in tokens:
        if token_type == tokenize.COMMENT:
            line_no, _ = start
            if line_no <= 3 or token_string.strip().startswith("# This"):
                comments.append(token_string)
    return comments

def split_code_and_comments(source_code):
    source_code = textwrap.dedent(source_code)
    docstrings, code_without_doc = extract_docstrings(source_code)
    top_level_comments = extract_top_level_comments(source_code)
    all_comments = '\n'.join(docstrings + top_level_comments)
    return all_comments, code_without_doc

# === MAIN ===
input_csv = sys.argv[1]
output_csv = sys.argv[2]
model = sys.argv[3]
col_no = int(sys.argv[4])
num_backward_passes = int(sys.argv[5])

df = pd.read_csv(input_csv)

for row_idx in range(len(df)):
    orig_col = "Original_Code"
    if orig_col in df.columns:
        orig_comments, orig_code = split_code_and_comments(str(df.at[row_idx, orig_col]))
        df.at[row_idx, f"{orig_col}"] = orig_code
        df.at[row_idx, f"{orig_col}_comments"] = orig_comments

    for i in range(1, col_no + 1):
        for k in range(1, num_backward_passes + 1):
            col = f"Generated_Code_{model}_{i}_code{k}"
            if col in df.columns:
                print(f"[{row_idx+1}/{len(df)}] Cleaning: {col}")
                try:
                    comments, code = split_code_and_comments(str(df.at[row_idx, col]))
                    df.at[row_idx, f"{col}_comments"] = comments
                    df.at[row_idx, f"{col}"] = code
                except Exception as e:
                    print(f"⚠️ Error cleaning {col} in row {row_idx+1}: {e}")
                    df.at[row_idx, f"{col}_comments"] = ""
                    df.at[row_idx, f"{col}"] = ""

df.to_csv(output_csv, index=False)
print(f"\nCleaned file saved to: {output_csv}")
