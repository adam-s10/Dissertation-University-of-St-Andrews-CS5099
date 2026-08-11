import os
import argparse
from dotenv import load_dotenv

from huggingface_hub import hf_hub_download

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
REPO_TYPE_DATASET = 'dataset'

datasets = {
    'xsum': {  # https://huggingface.co/datasets/EdinburghNLP/xsum
        'repo_id': 'EdinburghNLP/xsum',
        'files': ['data/test-00000-of-00001.parquet', 'data/train-00000-of-00001.parquet', 'data/validation-00000-of-00001.parquet'],
    },
    'squad': {  # https://huggingface.co/datasets/rajpurkar/squad
        'repo_id': 'rajpurkar/squad',
        'files': ['plain_text/train-00000-of-00001.parquet', 'plain_text/validation-00000-of-00001.parquet'],
    },
    'writingprompts': {  # https://huggingface.co/datasets/euclaise/writingprompts
        'repo_id': 'euclaise/writingprompts',
        'files': [
            'data/test-00000-of-00001-16503b0c26ed00c6.parquet', 'data/train-00000-of-00002-105e07cb0d199464.parquet',
            'data/train-00001-of-00002-4fdb982c11056472.parquet', 'data/validation-00000-of-00001-137b93e1e979d138.parquet'
        ],
    },
    'wmt16': {  # https://huggingface.co/datasets/wmt/wmt16
        'repo_id': 'wmt/wmt16',
        'files': [
            'de-en/test-00000-of-00001.parquet', 'de-en/train-00000-of-00003.parquet',
            'de-en/train-00001-of-00003.parquet', 'de-en/train-00002-of-00003.parquet',
            'de-en/validation-00000-of-00001.parquet'
        ],
    },
    'pubmedqa': {  # https://huggingface.co/datasets/qiaojin/PubMedQA
        'repo_id': 'qiaojin/PubMedQA',
        'files': ['pqa_labeled/train-00000-of-00001.parquet'],
    }
}

def download_data(download_location: str = 'Raw Datasets'):
    os.makedirs(download_location, exist_ok=True)
    for dataset_name in datasets.keys():
        dataset = datasets[dataset_name]
        for file in dataset['files']:
            hf_hub_download(
                repo_id=dataset['repo_id'],
                repo_type=REPO_TYPE_DATASET,
                filename=file,
                token=hf_token,
                local_dir=f'{download_location}/{dataset_name}'
            )

if __name__ == '__main__':

    download_data()