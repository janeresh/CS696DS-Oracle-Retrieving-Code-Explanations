import ast
import tokenize
from io import StringIO
import textwrap
import re
import autopep8


class CodeCleaner:
    def __init__(self):
        pass

    def fix_legacy_exception_syntax(self, code: str) -> str:
        return re.sub(r'except\s+(\w+)\s*,\s*(\w+):', r'except \1 as \2:', code)

    def remove_docstrings_fallback(self, code: str) -> str:
        return re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)
    
    def extract_code_blocks(self, text: str) -> list:
        # Looks for lines that begin with def or are indented blocks
        lines = text.splitlines()
        code_lines = []
        capture = False
        for line in lines:
            if line.strip().startswith("def "):
                capture = True
            if capture:
                code_lines.append(line)
            if capture and line.strip() == "":
                break
        return "\n".join(code_lines).strip()

    def remove_docstrings(self, source_code: str) -> str:
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

    def remove_all_comments(self, code: str) -> str:
        attempts = 0
        max_attempts = 1
        while attempts < max_attempts:
            try:
                io_obj = StringIO(code)
                tokens = list(tokenize.generate_tokens(io_obj.readline))
                break
            except IndentationError:
                attempts += 1
                code = autopep8.fix_code(code)
            except tokenize.TokenError as e:
                # Fallback to extracting valid-looking code blocks
                print(f"[Fallback] TokenError: {e}. Using extract_code_blocks.")
                return self.extract_code_blocks(code)
        else:
            return code

        output = ""
        last_lineno = -1
        last_col = 0
        for tok in tokens:
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

    def fix_indentation(self, code: str) -> str:
        try:
            return autopep8.fix_code(code)
        except Exception:
            print("[Warning] autopep8 failed. Falling back to extract_code_blocks.")
            return self.extract_code_blocks(code)


    def split_code_and_comments(self, source_code: str) -> str:
        source_code = self.fix_indentation(source_code)
        source_code = textwrap.dedent(source_code)
        no_docstrings = self.remove_docstrings(source_code)
        no_comments = self.remove_all_comments(no_docstrings)
        final_code = self.remove_extra_blank_lines(no_comments)
        return final_code
