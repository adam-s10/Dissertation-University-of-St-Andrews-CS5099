# imports
import os
import shutil
import pandas as pd
import torch

from models import load_model
import time
from datetime import timedelta
import argparse


# Methodology:
# "sample 150 to 500 human-written examples per dataset as negative
# samples and produce equal numbers of positive samples by prompting the source model with the
# first 30 tokens of the human-written text"

raw_datasets_dir = 'Raw Datasets'
sampled_datasets_dir = 'Sampled Datasets'
generated_datasets_dir = 'Generated Datasets'

number_of_samples = 150
dataset_dict = {}


def generate_sampled_datasets(remove_raw_datasets:bool=False) -> None:
    """
    Generates a randomly sampled .parquet file without replacement for each dataset in Raw Datasets.

    :param remove_raw_datasets: permanently deletes the Raw Datasets directory after creating the sampled dataset.
    :return: None
    """

    for dataset in os.listdir(raw_datasets_dir):
        # combine train, validation, test parquet files for all datasets
        data_dir = f'{raw_datasets_dir}/{dataset}/data'
        dataset_dict[dataset] = os.listdir(data_dir)  # gets a list of all files in the data directory
        dfs = [pd.read_parquet(f'{data_dir}/{file}') for file in dataset_dict[dataset]]
        combined_df = pd.concat(dfs)

        # randomly sample from combined DataFrame
        sampled = combined_df.sample(
            n=number_of_samples,
            axis=0,
            replace=False,  # no duplicate samples
            random_state=42
        )

        # Save sampled file to Sampled Datasets
        os.makedirs(sampled_datasets_dir, exist_ok=True)
        sampled.to_parquet(path=f'{sampled_datasets_dir}/{dataset}_sampled.parquet', index=False)

    if remove_raw_datasets:
        shutil.rmtree(raw_datasets_dir)


def generate_positive_samples(model_name:str, dataset_name:str, batch_size:int=20) -> None:
    """
    Adds positive examples of machine-generated content to the datasets.

    *Tokens can be calculated differently across models,
    some models first 30 tokens may have more or less characters when compared against others.*

    :param model_name: model to generate samples
    :param dataset_name: dataset to have samples generated for
    :param batch_size: batch size
    :return: None
    """

    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # make use of reserved memory to avoid torch.OutOfMemoryError
    data = pd.read_parquet(f'{sampled_datasets_dir}/{dataset_name}_sampled.parquet')
    data['is_machine'] = 0
    data = data[['document', 'is_machine']]
    passages = data.document.to_list()
    model, tokenizer, device = load_model(model_name)

    tokens = tokenizer(
        passages,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)

    first_thirty_tokens = [
        token_id[mask.bool()][:30] for token_id, mask in zip(tokens['input_ids'], tokens['attention_mask'])  # extracts the first 30 tokens that are not padded (attention_mask index != 0)
    ]
    
    outputs = []
    start = time.time()
    epochs = 0
    for i in range(0, len(first_thirty_tokens), batch_size):
        batch = first_thirty_tokens[i:i+batch_size]
        # use known structure of tokenizer return to avoid decode and re-tokenize pipeline
        input_ids = torch.stack(batch).to(device)  # tensor of size k x n where k is the batch size and n is number of tokens (30)
        attention_mask = torch.ones_like(input_ids).to(device)  # all tensors are the same size so no padding is used so attention mask will be all 1's
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        passage_lengths = tokens['attention_mask'][i:i+batch_size].sum(dim=1)
        max_tokens = passage_lengths.max().item()

        batch_output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
        
        batch_outputs = tokenizer.batch_decode(batch_output_tokens, skip_special_tokens=True)
        outputs.extend(batch_outputs)
        torch.cuda.empty_cache()
        epochs += batch_size
        if epochs % (batch_size * 5) == 0:
            print(f'------Epoch {epochs} complete------')
    end = time.time()
    print(f'Took {timedelta(seconds=end - start)}')
    df = pd.DataFrame({
        'document': outputs,
        'is_machine': 1
    })
    
    combined_df = pd.concat([data, df])
    combined_df.to_parquet()

    os.makedirs(generated_datasets_dir, exist_ok=True)
    combined_df.to_parquet(
        path=f'{generated_datasets_dir}/{model_name}_{dataset_name}_generated.parquet', index=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset sampler and generator')
    parser.add_argument('--mode', type=str, default='sample', required=True)
    parser.add_argument('--model', type=str, default='qwen', required=False)
    parser.add_argument('--dataset', type=str, default='xsum',  required=False)
    parser.add_argument('--remove', type=bool, default=False, required=False)
    parser.add_argument('--batch-size', type=int, default=20, required=False)

    args = parser.parse_args()

    modes = ['sample', 'generate']
    if args.mode not in modes:
        raise RuntimeError(f'Mode must use one of these two modes: {', '.join(modes)}')

    if args.mode == 'sample':
        generate_sampled_datasets(remove_raw_datasets=args.remove)

    else:
        generate_positive_samples(model_name=args.model, dataset_name=args.dataset, batch_size=args.batch_size)
