from .llama import LLAMA_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    if lm_type == "meta-llama/Llama-3.2-1B-Instruct":
        return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    else:
        raise ValueError(f"Unsupported LM type: {lm_type}")
