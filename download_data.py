import pandas as pd
from huggingface_hub import hf_hub_download

from os import getenv
from dotenv import load_dotenv

load_dotenv()
hf_token = getenv('HF_TOKEN')
download_location = '/Raw Datasets'
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
    # !-------- not marked as safe so will wait on this one --------!
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

def download_data():
    for dataset in datasets.keys():
        for file in dataset['files']:
            hf_hub_download(
                repo_id=dataset['repo_id'],
                repo_type=REPO_TYPE_DATASET,
                filename=file,
                token=hf_token,
                local_dir=f'{download_location}/{dataset}'
            )

def download_xsum():
    xsum_dataset = datasets['xsum']
    for file in xsum_dataset['files']:
        hf_hub_download(
            repo_id=xsum_dataset['repo_id'],
            repo_type=REPO_TYPE_DATASET,
            filename=file,
            token=hf_token,
            local_dir='/Users/adam/PycharmProjects/Dissertation-University-of-St-Andrews-CS5099/Raw Datasets/xsum'
        )

download_xsum()