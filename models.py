# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# TODO: where we will load models for the algorithm

model_info = {
    'qwen': 'Qwen/Qwen1.5-0.5B'
}

# model = AutoModelForCausalLM.from_pretrained(model_paths['qwen'])  # enable trust_remote_code for safetensors
# tokenizer = AutoTokenizer.from_pretrained(model_paths['qwen'])

def load_model(model_name:str):
    """
    Loads model weights and tokenizer from local HuggingFace directory.
    :param model_name: name of model to load
    :return: model, tokenizer
    """

    model_repo_id = model_info[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device}')
    model.eval()
    model.to(device)

    return model, tokenizer