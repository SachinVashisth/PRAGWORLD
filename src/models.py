import time
import openai
from typing import *

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ModuleNotFoundError: pass


API_KEY = "PUT_YOUR_API_KEY"

openai.api_type = "PUT_API_TYPE"
openai.api_version = "PUT_API_VERSION"

class LLMWrapper:
    def __call__(self, messages):
        raise NotImplementedError

class ChatGPTWrapper(LLMWrapper):
    def __init__(self, model_id: str="PUT_ENGINE", checkpoint_path=None,
                 delay: int=2, quantization=None, quantization_type=None):
        self.delay = delay
        self.api_key = API_KEY
        self.model_ckpt = model_id
        openai.api_base = "PUT_API_BASE"
        openai.api_key = "PUT_YOUR_API_KEY"
        self.model_id = model_id
    
    def __call__(self, prompt: str) -> str:
        time.sleep(self.delay)
        response = openai.ChatCompletion.create(
            engine = "PUT_ENGINE",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop='----'
        )
        generated_text = response['choices'][0]['message']['content'].strip()

        return generated_text

class LLMWrapper:
    def __call__(self, messages):
        raise NotImplementedError

class HFWrapper(LLMWrapper):
    def __init__(self, model_id, checkpoint_path: Union[str, None]=None, 
                 quantization: bool=True, quantization_type: int=4,
                 do_sample: bool=False, top_p: float=0.9):
        self.model_id = model_id
        self.model_path = checkpoint_path if checkpoint_path not in [None, "", "NONE"] else model_id
        if model_id in ["WizardLMTeam/WizardCoder-15B-V1.0"]:
            print("doing left padding")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        else: self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.do_sample = do_sample
        self.top_p = top_p
        if quantization:
            if quantization_type == 4:
                self.quantization_config = quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True,
                    quantization_config=quantization_config, 
                    device_map="auto",    
                )
            if quantization_type == 8:
                self.quantization_config = quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, trust_remote_code=True,
                    quantization_config=quantization_config, 
                    device_map="auto"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map="auto",
                trust_remote_code=True,
            )
            # if torch.cuda.is_available(): self.model.cuda()
    
    def __call__(self, messages):
        # print("LLM call")
        # start = time.time()
        # try:
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # except ValueError: # no default chat template set.
        #     tokenized_chat = self.tokenizer.tokenize(messages["content"], return_tensors="pt")
        # tokenized_chat = tokenized_chat 
        outputs = self.model.generate(
            tokenized_chat.to(self.model.device), max_new_tokens=500, 
            do_sample=self.do_sample, top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        # print(f"LLM call took: {time.time()-start}")
        return self.tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)

def get_llm_wrapper(model_id, **kwargs) -> LLMWrapper:
    if model_id in ["chatgpt", "gpt-4"] or model_id.startswith("gpt"):
        return ChatGPTWrapper(model_id, **kwargs)
    else:
        return HFWrapper(model_id, **kwargs)
