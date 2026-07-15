# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

model_info = {
    'qwen': 'Qwen/Qwen1.5-0.5B',
    'qwen_ft': 'kulia-moon/Lily-Qwen1.5-0.5B',
    'bitnet': 'microsoft/bitnet-b1.58-2B-4T-bf16',
    'llama': 'meta-llama/Llama-3.2-1B-Instruct',
    'stablelm': 'stabilityai/stablelm-3b-4e1t'
}

def load_model(model_name:str):
    """
    Loads model weights and tokenizer from local HuggingFace directory.

    :param model_name: name of model to load
    :return: model, tokenizer, device
    """

    model_repo_id = model_info[model_name]
    model = AutoModelForCausalLM.from_pretrained(model_repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # get device: GPU if available else CPU
    print(f'------ Running on {device} ------')

    model.eval()  # set model in evaluation mode
    model = model.to(device)  # run model on device
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:  # todo: check if all models decoder only, if so leave this otherwise will need to dynamically switch
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device