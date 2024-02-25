'''
    HF text decoder models
'''
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

HF_MODEL_URLS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
    "gemma-2b": "google/gemma-2b-it",
}

class HFBase():
    '''
        Load HF model
    '''
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_URLS[model_name])
        # self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name], load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name])
        self.model.to(device)
        self.device = device

    def predict(self, prompt, max_new_tokens=512):
        inputs = self.tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # top_k=top_k,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        output_tokens = output[0]
        output_tokens = output_tokens[inputs["input_ids"].shape[1] :]
        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()
        return output_text


class GemmaModel():
    '''
        Load HF model
    '''
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_URLS[model_name])

        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name], quantization_config=quantization_config)
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name], torch_dtype=torch.bfloat16)
        self.model.to(device)
        self.device = device

    def predict(self, prompt, max_new_tokens=512):

        chat = [{ "role": "user", "content": prompt}]
        modified_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(modified_prompt, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        output_tokens = output[0]
        output_tokens = output_tokens[inputs["input_ids"].shape[1] :]
        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()
        return output_text