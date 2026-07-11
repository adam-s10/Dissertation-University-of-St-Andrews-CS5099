from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import argparse

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
    },
    'llama': {
        'repo_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'files': [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map',
            'tokenizer.json',
            'tokenizer_config.json'
        ]

    },
    'bitnet_2b': {
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
    'bitnet': {  # this will not run locally, todo: remove...
        'repo_id': 'microsoft/bitnet-b1.58-2B-4T',
        'files': [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
    }
}

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
        for key in models.keys():
            for file in models[key]['files']:
                download_location = hf_hub_download(
                    repo_id=models[key]['repo_id'],
                    filename=file,
                    token=hf_token
                )


    if args.model:
        for file in models[args.model]['files']:
            download_location = hf_hub_download(
                repo_id=models[args.model]['repo_id'],
                filename=file,
                token=hf_token
            )
            print(f'File downloaded to: {download_location}')


# for file in models['qwen_ft']['files']:  # make reusable and turn into a function
#     download_location = hf_hub_download(
#         repo_id=models['qwen_ft']['repo_id'],
#         filename=file,
#         token=hf_token
#     )
#     print(f'File downloaded to: {download_location}')  # /Users/adam/.cache/huggingface/hub/models--Qwen--Qwen1.5-0.5B