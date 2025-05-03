class PromptVariants:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_prompt_intent(self, code: str, exp: str) -> str:
        example_1 = (
            "Code:\nfor i in range(len(items)):\n    print(items[i])\n\n"
            "Explanation:\nThe developer iterates through each element in a list by index to display each item, "
            "likely for debugging or logging list contents."
        )
        example_2 = (
            "Code:\ndata = {'a': 1, 'b': 2}\nwith open('output.json', 'w') as f:\n    json.dump(data, f)\n\n"
            "Explanation:\nThis code serializes an in-memory dictionary into JSON and writes it to a file, "
            "probably to save configuration or intermediate results."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that explains code by focusing on the author’s intent and motivation. "
                    "Do not describe syntax or API usage. Focus on what the code is meant to achieve and why, in no more than 3 sentences. "
                    "Here are two examples for guidance:\n"
                    f"{example_1}\n{example_2}\n"
                    f"Don't generate explanation similar to before: {exp}"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Code snippet:\n{code}\n"
                    "Instruction: Explain the intent and motivation behind this code in up to 3 sentences. "
                    "Answer:\n"                
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt

    def build_prompt_technical(self, code: str, exp: str) -> str:
        example_1 = (
            "Code:\nfor i in range(len(items)):\n    print(items[i])\n\n"
            "Explanation:\nFirst, calculate the list length, then iterate over each index, retrieve the element, "
            "and output it; this requires loop control and index access."
        )
        example_2 = (
            "Code:\ndata = {'a': 1, 'b': 2}\nwith open('output.json', 'w') as f:\n    json.dump(data, f)\n\n"
            "Explanation:\nIt builds a dictionary, opens a file handle in write mode via a context manager, "
            "serializes the dictionary into JSON, and writes it to disk."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a technical assistant who expands code snippets into concise, step-by-step descriptions. "
                    "Mention loop mechanics, data flow, and key operations without showing code. "
                    "Here are examples:\n"
                    f"{example_1}\n{example_2}\n"
                    f"Don't generate explanation similar to before: {exp}"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Code snippet:\n{code}\n"
                    "Instruction: Provide a concise, step-by-step technical explanation of this code in up to 3 sentences. "
                    "Answer:\n"                
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt

    def build_prompt_general(self, code: str, exp: str) -> str:
        example_1 = (
            "Code:\nfor i in range(len(items)):\n    print(items[i])\n\n"
            "Explanation:\nThis goes through each thing in a list and prints it one by one."
        )
        example_2 = (
            "Code:\ndata = {'a': 1, 'b': 2}\nwith open('output.json', 'w') as f:\n    json.dump(data, f)\n\n"
            "Explanation:\nIt takes a small table of information and saves it in a file in a simple text format for later use."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly assistant that explains code in simple, beginner-friendly language. "
                    "Avoid jargon. Keep explanations under 3 sentences. "
                    "Examples:\n"
                    f"{example_1}\n{example_2}\n"
                    f"Don't generate explanation similar to before: {exp}"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Code snippet:\n{code}\n"
                    "Instruction: Summarize what this code does in plain language. "
                    "Answer:\n"              
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt

    def build_prompt_semitechnical(self, code: str, exp: str) -> str:
        example_1 = (
            "Code:\nfor i in range(len(items)):\n    print(items[i])\n\n"
            "Explanation:\nThis snippet loops by index through a list to display each element, "
            "using basic loop control and indexing."
        )
        example_2 = (
            "Code:\ndata = {'a': 1, 'b': 2}\nwith open('output.json', 'w') as f:\n    json.dump(data, f)\n\n"
            "Explanation:\nThe code creates a dictionary, opens a file safely, and writes the dictionary as JSON."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a semi-technical assistant who explains code with moderate detail. "
                    "Mention concepts like data structures or I/O without syntax. "
                    "Examples:\n"
                    f"{example_1}\n{example_2}\n"
                    f"Don't generate explanation similar to before: {exp}"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Code snippet:\n{code}\n"
                    "Instruction: Explain this code in moderate detail in up to 3 sentences. "
                    "Answer:\n"            
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt

    def build_prompt_highlevel(self, code: str, exp: str) -> str:
        example_1 = (
            "Code:\nfor i in range(len(items)):\n    print(items[i])\n\n"
            "Explanation:\nThis loop supports a data inspection workflow for debugging or monitoring list contents."
        )
        example_2 = (
            "Code:\ndata = {'a': 1, 'b': 2}\nwith open('output.json', 'w') as f:\n    json.dump(data, f)\n\n"
            "Explanation:\nThis step integrates serialized data into a persistence layer by exporting runtime state as JSON."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software architect interpreting code at a system level. "
                    "Describe how this snippet fits into broader workflows like logging or persistence. "
                    "Examples:\n"
                    f"{example_1}\n{example_2}\n"
                    f"Don't generate explanation similar to before: {exp}"
                )
            },
            {
                "role": "user", 
                "content": (
                    f"Code snippet:\n{code}\n"
                    "Instruction: Interpret this code’s high-level role in an application in up to 3 sentences. "
                    "Answer:\n"           
                )
            }
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt
    
