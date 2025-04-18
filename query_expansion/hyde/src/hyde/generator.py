# import time
# import openai
# import cohere
from vllm import LLM, SamplingParams


class Generator:
    def __init__(self, model_name, api_key=""):
        self.model_name = model_name
        self.api_key = api_key
    
    def generate(self):
        return ""


# class OpenAIGenerator(Generator):
#     def __init__(self, model_name, api_key, base_url=None, n=8, max_tokens=512, temperature=0.7, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
#         super().__init__(model_name, api_key)
#         self.n = n
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.frequency_penalty = frequency_penalty
#         self.presence_penalty = presence_penalty
#         self.stop = stop
#         self.wait_till_success = wait_till_success
#         self._client_init()
#         self.base_url = base_url
    
#     @staticmethod
#     def parse_response(response):
#         to_return = []
#         for _, g in enumerate(response['choices']):
#             text = g['text']
#             logprob = sum(g['logprobs']['token_logprobs'])
#             to_return.append((text, logprob))
#         texts = [r[0] for r in sorted(to_return, key=lambda tup: tup[1], reverse=True)]
#         return texts
    
#     def _client_init(self):
#         self.client = openai.OpenAI(
#             base_url=self.base_url,
#             api_key=self.api_key,
#         )
#         self.client

#     def generate(self, prompt):
#         get_results = False
#         while not get_results:
#             try:
#                 result = self.client.chat.completions.create(
#                     messages=[{"role":"user", "content": prompt}]
#                     model=self.model_name,
#                     max_completion_tokens=self.max_tokens,
#                     temperature=self.temperature,
#                     frequency_penalty=self.frequency_penalty,
#                     presence_penalty=self.presence_penalty,
#                     top_p=self.top_p,
#                     n=self.n, # some models only support n=1
#                     stop=self.stop,
#                     logprobs=1 # some models are not compatible with this setting
#                 )
#                 get_results = True
#             except Exception as e:
#                 if self.wait_till_success:
#                     time.sleep(1)
#                 else:
#                     raise e
#         return self.parse_response(result)


# class CohereGenerator(Generator):
#     def __init__(self, model_name, api_key, n=8, max_tokens=512, temperature=0.7, p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n\n'], wait_till_success=False):
#         super().__init__(model_name, api_key)
#         self.cohere = cohere.Cohere(self.api_key)
#         self.n = n
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.p = p
#         self.frequency_penalty = frequency_penalty
#         self.presence_penalty = presence_penalty
#         self.stop = stop
#         self.wait_till_success = wait_till_success

    
#     @staticmethod
#     def parse_response(response):
#         text = response.generations[0].text
#         return text
    
#     def generate(self, prompt):
#         texts = []
#         for _ in range(self.n):
#             get_result = False
#             while not get_result:
#                 try:
#                     result = self.cohere.generate(
#                         prompt=prompt,
#                         model=self.model_name,
#                         max_tokens=self.max_tokens,
#                         temperature=self.temperature,
#                         frequency_penalty=self.frequency_penalty,
#                         presence_penalty=self.presence_penalty,
#                         p=self.p,
#                         k=0,
#                         stop=self.stop,
#                     )
#                     get_result = True
#                 except Exception as e:
#                     if self.wait_till_success:
#                         time.sleep(1)
#                     else:
#                         raise e
#             text = self.parse_response(result)
#             texts.append(text)
#         return texts

class VLLMGenerator(Generator):
    def __init__(self, model_name, model_path, api_key='', n=1, max_tokens=512, temperature=0.7, top_p=1.0):
        super().__init__(model_name, api_key)
        self.model_path = model_path
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        print(f"Loading vLLM model from {model_path}")
        self.llm = LLM(model=model_path, trust_remote_code=True, dtype="float16")
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=self.n
        )

    def generate(self, prompt):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]