from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import torch
import random
from StaICC.util import experimentor

def load_hf_model(
    name: str, 
    device: str = "cuda", 
    huggingface_token = None, 
    quantized = False, 
    forcedownload = False, 
    revision = None
):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if quantized else None
    if huggingface_token is not None:
        model = AutoModelForCausalLM.from_pretrained(name, token = huggingface_token, quantization_config = quantization_config, force_download = forcedownload, revision = revision, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config = quantization_config, force_download = forcedownload, revision = revision, trust_remote_code=True)
    if not quantized:
        model.to(device)
    model.eval()
    if huggingface_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(name, token = huggingface_token, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model, tokenizer