import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLAMA_MODEL:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda", attn_implementation="flash_attention_2", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.lm, self.tokenizer = self._load_model()
        self.lm.eval()
        self.lm.to(self.device).to(self.dtype)

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(self.model_name, attn_implementation=self.attn_implementation, dtype=self.dtype)
        return lm, tokenizer

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str):
        chat_data = [{"role": "system", "content": sys_prompt}]
        chat_data += chat_history
        chat_data += [{"role": "assistant", "content": recommend_item}]
        chat_template = self.tokenizer.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)
        return chat_template

    def response_generation(self, sys_prompt: str, chat_history: list, recommend_item: str,max_new_tokens=512, response_format=None):
        chat_history = self._format_chat_history(sys_prompt, chat_history, recommend_item)
        token_inputs = self.tokenizer(chat_history, return_tensors="pt")
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            outputs = self.lm.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        generated_text = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)[0]
        return generated_text

    def batch_response_generation(self, sys_prompts: list[str], chat_histories: list[list], recommend_items: list[str], max_new_tokens=64):
        """Generate responses for multiple inputs in batch.

        Args:
            sys_prompts: List of system prompts.
            chat_histories: List of chat history lists.
            recommend_items: List of recommended items.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated response texts.
        """
        # Format all chat histories
        formatted_chats = [
            self._format_chat_history(sys_prompt, chat_history, recommend_item)
            for sys_prompt, chat_history, recommend_item in zip(sys_prompts, chat_histories, recommend_items)
        ]

        # Tokenize with padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        token_inputs = self.tokenizer(formatted_chats, return_tensors="pt", padding=True, truncation=True)
        input_ids = token_inputs.input_ids.to(self.device)
        attention_mask = token_inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only the newly generated tokens
        generated_texts = self.tokenizer.batch_decode(outputs[:,input_ids.shape[1]:], skip_special_tokens=True)
        return generated_texts
