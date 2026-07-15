from dotenv import load_dotenv
import os
import argparse

from huggingface_hub import hf_hub_download

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

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
    },
    'llama': {
        'repo_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'files': [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]

    },
    'bitnet': {
        'repo_id': 'microsoft/bitnet-b1.58-2B-4T-bf16',
        'files': [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
    },
    'stablelm': {
        'repo_id': 'stabilityai/stablelm-3b-4e1t',
        'files': [
            'config.json',
            'configuration_stablelm.py',
            'generation_config.json',
            'model.safetensors',
            'modeling_stablelm.py',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
    }
}

def download_all_models():
    for key in models.keys():
        for file in models[key]['files']:
            download_location = hf_hub_download(
                repo_id=models[key]['repo_id'],
                filename=file,
                token=hf_token
            )
            print(f'File downloaded to: {download_location}')


def download_individual_model(model_name: str):
    for file in models[model_name]['files']:
        download_location = hf_hub_download(
            repo_id=models[args.model]['repo_id'],
            filename=file,
            token=hf_token
        )
        print(f'File downloaded to: {download_location}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Hugging Face Models')
    parser.add_argument('--all', type=bool, default=False, required=False)
    parser.add_argument('--model', type=str, default='qwen', required=False)
    args = parser.parse_args()

    if args.model not in models:
        raise RuntimeError(
            f'Provided model "{args.model}" is not available for download, select one of the following: {', '.join(models.keys())}'
        )

    if args.all:
        download_all_models()

    if args.model:
        download_individual_model(args.model)
