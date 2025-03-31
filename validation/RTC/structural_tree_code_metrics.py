from tree_sitter import Parser, Language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import re
import tree_sitter_python as tspython
import pandas as pd
import numpy as np
import sys

class CodeSimilarityCalculator:
    def __init__(self, language='python'):
        # Initialize parser
        self.language = Language(tspython.language())
        self.parser = Parser(self.language)
        
        # For normalization
        self.identifier_counter = 1
        self.identifier_map = {}
        self.scope_stack = [{}]
        
        # Special tokens to preserve
        self.preserved_tokens = {'True', 'False', 'None', '0', '1', 'if', 'else', 'for', 'while', 'return'}
        
        # For vectorization
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize_code, lowercase=False)

    @staticmethod
    def compute_rtc(sim_scores):
        Nf, Nb = len(sim_scores), len(sim_scores[0]) if sim_scores else 0
        denom = Nf * Nb if Nf and Nb else 1
        rtc_score = sum(sum(row) for row in sim_scores) / denom
        return round(rtc_score, 4)

    @staticmethod
    def pass_at_k(n, c, k):
        return 1.0 if n - c < k else 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def normalize_code(self, code_str):
        self.identifier_counter = 1
        self.identifier_map = {}
        self.scope_stack = [{}]
        
        tree = self.parser.parse(bytes(code_str, 'utf8'))
        normalized = self._normalize_node(tree.root_node)
        return self._structure_to_text(normalized)
    
    def _normalize_node(self, node):
        normalized = {'type': node.type}
        
        if node.type == 'identifier':
            text = node.text.decode('utf8')
            if text in self.preserved_tokens:
                return {'type': 'literal', 'value': text}
            return {'type': 'identifier', 'name': self._get_normalized_name(node)}
        
        if node.type in ('function_definition', 'class_definition', 'block'):
            self.scope_stack.append({})
            
        normalized['children'] = []
        for child in node.children:
            norm_child = self._normalize_node(child)
            if norm_child:
                normalized['children'].append(norm_child)
        
        if node.type in ('function_definition', 'class_definition', 'block'):
            self.scope_stack.pop()
            
        return normalized

    def _get_normalized_name(self, node):
        text = node.text.decode('utf8')
        parent = node.parent
        
        for scope in reversed(self.scope_stack):
            if text in scope:
                return scope[text]
        
        if parent.type == 'function_definition' and node == parent.child_by_field_name('name'):
            prefix = 'FUNC'
        elif parent.type == 'class_definition' and node == parent.child_by_field_name('name'):
            prefix = 'CLASS'
        elif parent.type in ('parameters', 'lambda_parameters'):
            prefix = 'PARAM'
        elif text.isupper():
            prefix = 'CONST'
        else:
            prefix = 'VAR'
            
        norm_name = f"{prefix}_{self.identifier_counter}"
        self.identifier_counter += 1
        self.scope_stack[-1][text] = norm_name
        return norm_name

    def _structure_to_text(self, node):
        if node['type'] == 'identifier':
            return node['name']
        elif node['type'] == 'literal':
            return node['value']
        
        parts = [node['type']]
        for child in node.get('children', []):
            parts.append(self._structure_to_text(child))
        return ' '.join(parts)

    def _tokenize_code(self, code_text):
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S', code_text)
        return [t for t in tokens if t.strip()]

    def compute_cosine_similarity(self, code1, code2):
        try:
            norm1 = self.normalize_code(code1)
            norm2 = self.normalize_code(code2)
            
            if not norm1 or not norm2:
                return 0.0
                
            vectors = self.vectorizer.fit_transform([norm1, norm2])
            similarity = sklearn_cosine_similarity(vectors[0], vectors[1])[0][0]
            return round(similarity, 4)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0


# === Main Execution ===
if __name__ == "__main__":
    comparator = CodeSimilarityCalculator()
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    model = sys.argv[3]

    col_no = int(sys.argv[4])
    num_backward_passes = int(sys.argv[5])  # number of generations per sample
    results = []

    input_df = pd.read_csv(input_csv)
    for row_iter, row in input_df.iterrows():
        print(f"\nProcessing row {row_iter + 1}/{len(input_df)}...")
        original_code = row["Original_Code"]
        new_row = row.copy()
        sim_scores = []

        for i in range(col_no):
            row_scores = []
            for j in range(num_backward_passes):
                col_name = f"Generated_Code_{model}_{i+1}_code{j+1}"
                score_col = f"Structural_Score_{model}_{i+1}_code_{j+1}"
                similarity = 0.0
                if col_name in row:
                    gen_code = row[col_name]
                    if gen_code and not pd.isna(gen_code):
                        similarity = comparator.compute_cosine_similarity(original_code, gen_code)
                        new_row[score_col] = similarity
                    else:
                        similarity = 0.0
                        new_row[score_col] = 0.0
                else:
                    print("Column not found: ", col_name)
                row_scores.append(similarity)
            sim_scores.append(row_scores)
        
        # Compute RTC and Pass@1
        rtc_score = CodeSimilarityCalculator.compute_rtc(sim_scores)
        pass1_true_count = sum(score > 0.8 for scores in sim_scores for score in scores)
        pass1_score = CodeSimilarityCalculator.pass_at_k(col_no * num_backward_passes, pass1_true_count, 1)

        new_row[f"RTC_{model}_Struct_Score"] = rtc_score
        new_row[f"Pass@1_{model}_Struct_Score"] = round(pass1_score, 4)
        
        results.append(new_row)

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully processed and saved results to {output_csv}")
