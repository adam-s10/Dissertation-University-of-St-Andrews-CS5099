from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# todo: make this reusable...

models = {
    'qwen': {
        'repo_id': 'Qwen/Qwen1.5-0.5B',
        'files': [
            'config.json',
            'generation_config.json',
            'merges.txt',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json'
        ]
    },
    'qwen_ft': {
        'repo_id': 'kulia-moon/Lily-Qwen1.5-0.5B',
        'files': [
            'added_tokens.json',
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.json'
        ]
    }
}

for file in models['qwen_ft']['files']:  # make reusable and turn into a function
    download_location = hf_hub_download(
        repo_id=models['qwen_ft']['repo_id'],
        filename=file,
        token=hf_token
    )
    print(f'File downloaded to: {download_location}')  # /Users/adam/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B