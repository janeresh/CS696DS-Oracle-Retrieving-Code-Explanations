import pandas as pd
import ast
import tokenize
from io import StringIO
import textwrap
import re

# --- Patch legacy Python 2 exceptions ---
def fix_legacy_exception_syntax(code: str) -> str:
    return re.sub(r'except\s+(\w+)\s*,\s*(\w+):', r'except \1 as \2:', code)

# --- Fallback: Regex-based docstring remover ---
def remove_docstrings_fallback(code: str) -> str:
    return re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)

# --- Robust AST-based docstring remover with fallback ---
def remove_docstrings(source_code: str) -> str:
    source_code = fix_legacy_exception_syntax(source_code)
    lines = source_code.splitlines()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return remove_docstrings_fallback(source_code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if not node.body:
                continue
            first_stmt = node.body[0]
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                start = first_stmt.lineno - 1
                end = getattr(first_stmt, 'end_lineno', start + len(first_stmt.value.value.splitlines()))
                for i in range(start, end):
                    if 0 <= i < len(lines):
                        lines[i] = ''
    return '\n'.join(lines)

# --- Remove all `#` comments ---
def remove_all_comments(code: str) -> str:
    io_obj = StringIO(code)
    output = ""
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string, start, end, _ = tok
        if token_type == tokenize.COMMENT:
            continue
        sline, scol = start
        eline, ecol = end
        if sline > last_lineno:
            output += "\n" * (sline - last_lineno - 1)
            last_col = 0
        if scol > last_col:
            output += " " * (scol - last_col)
        output += token_string
        last_lineno = eline
        last_col = ecol
    return output

# --- Remove extra blank lines ---
def remove_extra_blank_lines(code: str) -> str:
    cleaned_lines = []
    previous_blank = False
    for line in code.splitlines():
        if line.strip() == "":
            if not previous_blank:
                cleaned_lines.append("")
                previous_blank = True
        else:
            cleaned_lines.append(line.rstrip())
            previous_blank = False
    return "\n".join(cleaned_lines)

# --- Full pipeline ---
def split_code_and_comments(source_code: str) -> str:
    source_code = textwrap.dedent(source_code)
    no_docstrings = remove_docstrings(source_code)
    no_comments = remove_all_comments(no_docstrings)
    final_code = remove_extra_blank_lines(no_comments)
    return final_code

def clean_code_column(input_csv: str, code_column: str = "code"):
    df = pd.read_csv(input_csv)
    df["cleaned_code"] = df[code_column].apply(split_code_and_comments)
    return df

# --- Step 6: Run it ---
if __name__ == "__main__":
    input_path = "/work/pi_wenlongzhao_umass_edu/27/anamikaghosh/pre_processing_CSN/CodeSearchNet_Python_valid_cleaned.csv"
    df = clean_code_column(input_path)
    df.to_csv("Cleaned_code.csv")
