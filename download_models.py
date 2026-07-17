from dotenv import load_dotenv
import os
import argparse

from huggingface_hub import hf_hub_download

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

config_json = 'config.json'
generation_config_json = 'generation_config.json'
model_safetensors = 'model.safetensors'
tokenizer_json = 'tokenizer.json'
tokenizer_config_json = 'tokenizer_config.json'
special_tokens_map_json = 'special_tokens_map.json'

models = {
    'qwen': {
        'repo_id': 'Qwen/Qwen1.5-0.5B',
        'files': [
            config_json,
            generation_config_json,
            'merges.txt',
            model_safetensors,
            tokenizer_json,
            tokenizer_config_json,
            'vocab.json'
        ]
    },
    'qwen_ft': {
        'repo_id': 'kulia-moon/Lily-Qwen1.5-0.5B',
        'files': [
            'added_tokens.json',
            config_json,
            generation_config_json,
            model_safetensors,
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json,
            'vocab.json'
        ]
    },
    'llama': {
        'repo_id': 'meta-llama/Llama-3.2-1B-Instruct',
        'files': [
            config_json,
            generation_config_json,
            model_safetensors,
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json
        ]

    },
    'bitnet': {
        'repo_id': 'microsoft/bitnet-b1.58-2B-4T-bf16',
        'files': [
            config_json,
            generation_config_json,
            model_safetensors,
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json
        ]
    },
    'stablelm': {
        'repo_id': 'stabilityai/stablelm-3b-4e1t',
        'files': [
            config_json,
            'configuration_stablelm.py',
            generation_config_json,
            model_safetensors,
            'modeling_stablelm.py',
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json
        ]
    },
    'rocket': {
        'repo_id': 'pansophic/rocket-3B',
        'files': [
            'added_tokens.json',
            config_json,
            generation_config_json,
            model_safetensors,
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json
        ]
    },
    'nvidia': {
        'repo_id': 'nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8',
        'files': [
config_json,
            'configureation_momtron_h.py',
            generation_config_json,
            'hf_quant_config.json',
            'model-00001-of-00010.saftensors',
            'model-00002-of-00010.saftensors',
            'model-00003-of-00010.saftensors',
            'model-00004-of-00010.saftensors',
            'model-00005-of-00010.saftensors',
            'model-00006-of-00010.saftensors',
            'model-00007-of-00010.saftensors',
            'model-00008-of-00010.saftensors',
            'model-00009-of-00010.saftensors',
            'model-00010-of-00010.saftensors',
            'model-00001-of-00010.json',
            'model-00002-of-00010.json',
            'model-00003-of-00010.json',
            'model-00004-of-00010.json',
            'model-00005-of-00010.json',
            'model-00006-of-00010.json',
            'model-00007-of-00010.json',
            'model-00008-of-00010.json',
            'model-00009-of-00010.json',
            'model-00010-of-00010.json',
            'model.safetensors.index.json',
            'modeling_nemotron_h.py',
            'nemotron_toolcall_parser_no_streaming.py',
            'nemotron_toolcall_parser_streaming.py',
            special_tokens_map_json,
            tokenizer_json,
            tokenizer_config_json
        ]
    },
    'google': {
        'repo_id': 'google/gemma-2-2b-it',
        'files': [
            'chat_template.jinja',
            config_json,
            generation_config_json,
            model_safetensors,
            'processor_config.json',
            tokenizer_json,
            tokenizer_config_json
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
