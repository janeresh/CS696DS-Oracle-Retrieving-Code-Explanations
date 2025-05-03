import numpy as np
import code_bert_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tree_sitter import Parser, Language
import tree_sitter_python as tspython
import os
import subprocess
import tempfile

class CodeBERTScorer:
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def compute_codebertscore(self, original_code, generated_code):
        P, R, F1, _ = code_bert_score.score(cands=[generated_code], refs=[original_code], lang='python')
        return F1.mean().item()

    def evaluate_generated_code(self, original_code, generated_code):
        exact_match = original_code.strip() == generated_code.strip()
        score = self.compute_codebertscore(original_code, generated_code)
        return exact_match, score

    def compute_rtc(self, sim_scores):
        Nf, Nb = len(sim_scores), len(sim_scores[0])
        return sum(sum(row) for row in sim_scores) / (Nf * Nb)

    def pass_at_k(self, n, c, k):
        return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class CodeStructureScorer:
    def __init__(self):
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize_code, lowercase=False)
        self.preserved_tokens = {'True', 'False', 'None', '0', '1', 'if', 'else', 'for', 'while', 'return'}

    def normalize_code(self, code_str):
        self.identifier_counter = 1
        self.scope_stack = [{}]
        tree = self.parser.parse(bytes(code_str, 'utf8'))
        normalized = self._normalize_node(tree.root_node)
        return self._structure_to_text(normalized)

    def _normalize_node(self, node):
        normalized = {'type': node.type}
        if node.type == 'identifier':
            return {'type': 'identifier', 'name': self._get_normalized_name(node)}
        if node.type in ('function_definition', 'class_definition', 'block'):
            self.scope_stack.append({})
        normalized['children'] = [self._normalize_node(child) for child in node.children if child]
        if node.type in ('function_definition', 'class_definition', 'block'):
            self.scope_stack.pop()
        return normalized

    def _get_normalized_name(self, node):
        text = node.text.decode('utf8')
        for scope in reversed(self.scope_stack):
            if text in scope:
                return scope[text]
        prefix = 'VAR'
        norm_name = f"{prefix}_{self.identifier_counter}"
        self.identifier_counter += 1
        self.scope_stack[-1][text] = norm_name
        return norm_name

    def _structure_to_text(self, node):
        if node['type'] == 'identifier':
            return node['name']
        parts = [node['type']]
        for child in node.get('children', []):
            parts.append(self._structure_to_text(child))
        return ' '.join(parts)

    def _tokenize_code(self, code_text):
        return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S', code_text)

    def compute_cosine_similarity(self, code1, code2):
        try:
            norm1 = self.normalize_code(code1)
            norm2 = self.normalize_code(code2)
            if not norm1 or not norm2:
                return 0.0
            vectors = self.vectorizer.fit_transform([norm1, norm2])
            return cosine_similarity(vectors[0], vectors[1])[0][0]
        except Exception:
            return 0.0

    def compute_rtc(self, sim_scores):
        Nf, Nb = len(sim_scores), len(sim_scores[0]) if sim_scores else 0
        denom = Nf * Nb if Nf and Nb else 1
        return round(sum(sum(row) for row in sim_scores) / denom, 4)

    def pass_at_k(self, n, c, k):
        return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class CodeBLEUEvaluator:
    def __init__(self, script_path='../CodeBLEU/calc_code_bleu.py', lang='python', params='0.25,0.25,0.25,0.25'):
        self.script_path = script_path
        self.lang = lang
        self.params = params

    def compute_rtc(self, sim_scores):
        Nf, Nb = len(sim_scores), len(sim_scores[0])
        denom = Nf * Nb
        return sum(sum(row) for row in sim_scores) / denom

    def pass_at_k(self, n, c, k):
        return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def run_codebleu_on_strings(self, reference_codes, generated_code):
        ref_paths = []
        # Flatten multi-line code into single line
        
        try:
            # Write reference code(s)
            for ref_code in reference_codes:
                flattened_ref = " ".join(line.strip() for line in ref_code.strip().splitlines() if line.strip())

                tmp_ref = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
                tmp_ref.write(flattened_ref + "\n")
                tmp_ref.flush()
                ref_paths.append(tmp_ref.name)
                tmp_ref.close()

            # Write hypothesis (generated code)
            flattened_hyp = " ".join(line.strip() for line in generated_code.strip().splitlines() if line.strip())
            tmp_hyp = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
            tmp_hyp.write(flattened_hyp + "\n")
            tmp_hyp.flush()
            hyp_path = tmp_hyp.name
            tmp_hyp.close()

            # Run subprocess
            cmd = ["python", self.script_path, "--refs", *ref_paths, "--hyp", hyp_path,
                "--lang", self.lang, "--params", self.params]
            result = subprocess.run(cmd, capture_output=True, text=True)

            print("=== CodeBLEU OUTPUT ===")
            print(result.stdout)
            print("=== STDERR ===")
            print(result.stderr)

            score_line = [line for line in result.stdout.splitlines() if 'CodeBLEU score:' in line]
            score = float(score_line[0].split()[-1]) if score_line else None

        finally:
            for path in ref_paths:
                os.remove(path)
            if 'hyp_path' in locals():
                os.remove(hyp_path)

        return score
    