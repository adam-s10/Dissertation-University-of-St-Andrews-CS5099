# imports
import os
import shutil

import pandas as pd


# Methodology:
# "sample 150 to 500 human-written examples per dataset as negative
# samples and produce equal numbers of positive samples by prompting the source model with the
# first 30 tokens of the human-written text"
raw_datasets_dir = 'Raw Datasets'
sampled_datasets_dir = 'Sampled Datasets'
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
        dict_[dataset] = os.listdir(data_dir)
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

    if remove_raw_datasets: shutil.rmtree(raw_datasets_dir)


def generate_positive_samples() -> None:
    """
    Adds positive examples of machine-generated content to the datasets.

    *Tokens can be calculated differently across models,
    for the sake of reproducibility, we will class the first 30 tokens as the first 30 words in the passsage.*

    :return: None
    """
    # todo
    pass

generate_sampled_datasets()