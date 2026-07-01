# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# TODO: where we will load models for the algorithm

model_info = {
    'qwen': {
        'dir': 'models--Qwen--Qwen1.5-0.5B',
        'dtype': torch.Tensor.bfloat16
    }
}

# model = AutoModelForCausalLM.from_pretrained(model_paths['qwen'])  # enable trust_remote_code for safetensors
# tokenizer = AutoTokenizer.from_pretrained(model_paths['qwen'])

def load_model(model_name:str, model_path: None | str=None):
    model_dir = model_info[model_name]['dir']
    model_dtype = model_info[model_name]['dtype']
    if not model_path:
        model_path = os.path.join('.cache/huggingface/hub', model_dir)

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=model_dtype)
    tokenizer = AutoTokenizer(model_path)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f'Running on {device}')
    model.eval()
    model.to_device(device)

    return model, tokenizer