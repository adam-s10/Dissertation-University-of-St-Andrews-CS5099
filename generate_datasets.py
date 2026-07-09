# imports
import os
import shutil
import pandas as pd
import torch
import numpy as np

from models import load_model
import time
from datetime import timedelta


# Methodology:
# "sample 150 to 500 human-written examples per dataset as negative
# samples and produce equal numbers of positive samples by prompting the source model with the
# first 30 tokens of the human-written text"
raw_datasets_dir = 'Raw Datasets'
sampled_datasets_dir = 'Sampled Datasets'
generated_datasets_dir = 'Generated Datasets'
number_of_samples = 150

dict_ = {}

def generate_sampled_datasets(remove_raw_datasets:bool=False) -> None:
    """
    Generates a randomly sampled .parquet file without replacement for each dataset in Raw Datasets.

    :param remove_raw_datasets: permanently deletes the Raw Datasets directory after creating the sampled dataset.
    :return: None
    """

    for dataset in os.listdir(raw_datasets_dir):
        # combine train, validation, test parquet files for all datasets
        data_dir = f'{raw_datasets_dir}/{dataset}/data'
        dict_[dataset] = os.listdir(data_dir)  # gets a list of all files in the data directory
        dfs = [pd.read_parquet(f'{data_dir}/{file}') for file in dict_[dataset]]
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


def generate_positive_samples(model_name:str, dataset_name:str) -> None:
    """
    Adds positive examples of machine-generated content to the datasets.

    *Tokens can be calculated differently across models,
    for the sake of reproducibility, we will class the first 30 tokens as the first 30 words in the passsage.*

    :return: None
    """
    # todo
    # ----------Skeleton Code----------
    
    # todo: add switch statement to handle other dataset column names in future
    # add column "is_machine"
    # tokenize input
    # pass input to model (maybe include prompt?)
    # decode output
    # check minimum number of tokens generated (how should this be calculated and checked?)
    # append to dataset as column "generated_text"

    # ----------Skeleton Code----------


    def func(passage, model_name_:str) -> str:
        pass

        model, tokenizer, device = load_model(model_name_)
        passage_length = len(passage)
        min_tokens = round(np.quantile(passage_length, q=.25))

        print(passage)

        inputs = tokenizer([passage], return_tensors='pt').to(device)
        output = ''

        while len(output) < min_tokens:
            output_tokens = model.generate(**inputs, min_new_tokens=min_tokens, max_new_tokens=passage_length)
            output = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            print(output)
        


    
    data = pd.read_parquet(f'{sampled_datasets_dir}/{dataset_name}_sampled.parquet')
    data['is_machine'] = 0

    data = data[['document', 'is_machine']]

    passages = data['document'].to_list()
    passages_first_thirty_tokens = [passage[:30] for passage in passages]

    min_tokens, max_tokens = np.quantile(data['document'].str.len(), q=[.25, .75])
    min_tokens, max_tokens = round(min_tokens), round(max_tokens)

    model, tokenizer, device = load_model(model_name)
    
    inputs = tokenizer(
        passages_first_thirty_tokens,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    output_tokens = model.generate(
        **inputs,
        min_new_tokens=min_tokens,
        max_new_tokens=max_tokens
    )
    df = pd.DataFrame({
        'document': tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    })
    df['is_machine'] = 1
    
    combined_df = pd.concat([data, df])
    combined_df.info()

    print(combined_df.loc[200]['document'])
    
    # model, tokenizer = load_model(model_name)
    
    
    # pass

# generate_sampled_datasets()

# generate_positive_samples('qwen', 'xsum')

def toy_example(model_name:str, dataset_name:str) -> None:
    data = pd.read_parquet(f"{sampled_datasets_dir}/{dataset_name}_sampled.parquet").iloc[0, :]
    data["is_machine"] = 0

    data = data[["document", "is_machine"]]

    passages = data["document"].to_list()
    # print(passages)
    # passages_first_thirty_tokens = [passage[:30] for passage in passages]
    # print(passages_first_thirty_tokens)
    # print(data["document"].str.len())



    min_tokens, max_tokens = np.quantile(data["document"].str.len(), q=[.25, .75])
    min_tokens, max_tokens = round(min_tokens), round(max_tokens)

    model, tokenizer, device = load_model(model_name)

    start = time.time()
    inputs = tokenizer(
        passages,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    output_tokens = model.generate(
        **inputs,
        min_new_tokens=min_tokens,
        max_new_tokens=max_tokens
    )
    end = time.time()
    df = pd.DataFrame(
        {"document": tokenizer.batch_decode(output_tokens, skip_special_tokens=True)}
    )
    df["is_machine"] = 1

    for i in df['document']:
        print('Generated passage:', i)

    combined_df = pd.concat([data, df])
    combined_df.info()
    print(combined_df.head())



    print(f'Ran on 5 rows of data. Started at: {start}, ended at: {end}, took: {end - start}')

# toy_example('qwen', 'xsum')

def toy_example_2(model_name:str, dataset_name:str) -> None:
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

    # use known structure of tokenizer return to avoid decode and re-tokenize pipeline
    input_ids = torch.stack(first_thirty_tokens)  # tensor of size k x n where k is the batch size and n is number of tokens (30)
    attention_mask = torch.ones_like(input_ids)  # all tensors are the same size so no padding is used so attention mask will be all 1's
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    passage_lengths = tokens['attention_mask'].sum(dim=1)
    print(passage_lengths.max())
    max_tokens = passage_lengths.max().item()
    print(max_tokens)

    start = time.time()
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        do_sample=True,
        top_p=.9,
        temperature=.9
    )
    end = time.time()
    print(f'Took {timedelta(seconds=end - start)}')
    
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
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
    

toy_example_2('qwen_ft', 'xsum')