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

    def compute_metrics(self, row, model_name, col_no, num_backward_passes):
        original_code = row.get("cleaned_code", "")
        codebert_scores = []

        for i in range(col_no):
            sub_scores = []
            for j in range(num_backward_passes):
                col = f"generated_code_{i+1}_code{j+1}"
                score_col = f"CodeBERT_Score_{model_name}_{i+1}_code{j+1}"
                match_col = f"Exact_Match_{model_name}_{i+1}_code{j+1}"
                gen_code = row.get(col, "")
                if isinstance(gen_code, str) and isinstance(original_code, str):
                    exact_match, score = self.evaluate_generated_code(original_code, gen_code)
                else:
                    exact_match, score = False, 0.0
                row[score_col] = score
                row[match_col] = exact_match
                sub_scores.append(score)
            codebert_scores.append(sub_scores)

        row[f"RTC_{model_name}_CodeBERT_Score"] = self.compute_rtc(codebert_scores)
        pass1 = sum(score > self.threshold for scores in codebert_scores for score in scores)
        row[f"Pass@1_{model_name}_CodeBERT_Score"] = self.pass_at_k(col_no * num_backward_passes, pass1, 1)
        return row

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

    def compute_structural_scores(self, row, model, col_no, num_backward_passes):
        original_code = row.get("cleaned_code", "")
        sim_scores = []

        for i in range(col_no):
            row_scores = []
            for j in range(num_backward_passes):
                col_name = f"generated_code_{i+1}_code{j+1}"
                score_col = f"Structural_Score_{i+1}_code_{j+1}"
                sim = 0.0
                gen_code = row.get(col_name, "")
                if isinstance(gen_code, str) and isinstance(original_code, str):
                    sim = self.compute_cosine_similarity(original_code, gen_code)
                row[score_col] = sim
                row_scores.append(sim)
            sim_scores.append(row_scores)

        row[f"RTC_Struct_Score"] = self.compute_rtc(sim_scores)
        true_positive = sum(score > 0.8 for scores in sim_scores for score in scores)
        row[f"Pass@1_Struct_Score"] = round(self.pass_at_k(col_no * num_backward_passes, true_positive, 1), 4)
        return row


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

    def run_codebleu_on_strings(self, reference_codes, generated_code, lang='python',
                                params='0.25,0.25,0.25,0.25', script_path='CodeBLEU/calc_code_bleu.py'):
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
            cmd = ["python", script_path, "--refs", *ref_paths, "--hyp", hyp_path,
                "--lang", lang, "--params", params]
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
    
    def compute_bleu_metrics(self, row, model, col_no, num_backward_passes):
        code_bleu_scores = []
        original_code = row.get("cleaned_code", "")

        for i in range(col_no):
            code_bleu_scores_sub = []
            for j in range(num_backward_passes):
                gen_col = f"generated_code_{i+1}_code{j+1}"
                score_col = f"CodeBLEU_Score_{i+1}_code{j+1}"
                
                gen_code = row.get(gen_col, "")
                
                if isinstance(gen_code, str) and isinstance(original_code, str):
                    code_bleu_score = self.run_codebleu_on_strings([original_code], gen_code)
                    row[score_col] = code_bleu_score
                    code_bleu_scores_sub.append(code_bleu_score)
                else:
                    row[score_col] = 0.0
                    code_bleu_scores_sub.append(0.0)
            
            code_bleu_scores.append(code_bleu_scores_sub)

        row[f"RTC_CodeBLEU_Score"] = self.compute_rtc(code_bleu_scores)
        pass1_true_count = sum(score > 0.8 for scores in code_bleu_scores for score in scores)
        row[f"Pass@1_CodeBLEU_Score"] = self.pass_at_k(col_no * num_backward_passes, pass1_true_count, 1)

        return row
