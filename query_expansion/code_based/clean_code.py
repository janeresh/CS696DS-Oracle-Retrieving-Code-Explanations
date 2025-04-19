import ast
import tokenize
import textwrap
from io import StringIO
import re

class CodeCleaner:
    def __init__(self):
        pass

    def remove_docstrings(self, source_code):
        source_code = self.fix_legacy_exception_syntax(source_code)
        lines = source_code.splitlines()
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self.remove_docstrings_fallback(source_code)

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

    def remove_docstrings_fallback(self, code):
        return re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)

    def fix_legacy_exception_syntax(self, code: str) -> str:
        return re.sub(r'except\s+(\w+)\s*,\s*(\w+):', r'except \1 as \2:', code)

    def remove_comments(self, code: str) -> str:
        io_obj = StringIO(code)
        output = ""
        last_lineno = -1
        last_col = 0
        try:
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
        except tokenize.TokenError:
            return code  # fallback on malformed tokens
        return output

    def remove_extra_blank_lines(self, code: str) -> str:
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

    def clean_code(self, code: str) -> str:
        if not isinstance(code, str):
            return ""
        code = textwrap.dedent(code)
        code = self.remove_docstrings(code)
        code = self.remove_comments(code)
        code = self.remove_extra_blank_lines(code)
        return code.strip()
