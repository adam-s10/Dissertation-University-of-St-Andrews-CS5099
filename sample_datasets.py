import argparse
import os
import shutil

from sklearn.model_selection import train_test_split
import pandas as pd


raw_datasets_dir = 'Raw Datasets'
sampled_datasets_dir = 'Sampled Datasets'

def generate_sampled_datasets(remove_raw_datasets: bool = False, number_of_samples: int = 1000) -> None:
    """
    Generates a randomly sampled .parquet file without replacement for each dataset in Raw Datasets.

    :param remove_raw_datasets: permanently deletes the Raw Datasets directory after creating the sampled dataset.
    :param number_of_samples: number of samples to be taken total from combined dataset
    :return: None
    """

    dataset_dict = {}
    for dataset in os.listdir(raw_datasets_dir):
        print(f'------- Starting sampling for file {dataset} ------')
        # combine train, validation, test parquet files for all datasets
        individual_dataset_root = f'{raw_datasets_dir}/{dataset}'  # Raw Datasets/"Dataset Name" e.g. Raw Datasets/wmt16
        individual_dataset_data_dir = [dirs for dirs in os.listdir(individual_dataset_root) if dirs != '.cache'][0]  # Raw Datasets/"Dataset Name"/"Data for dataset dir" e.g. Raw Datasets/wmt16/de-en

        data_dir = f'{individual_dataset_root}/{individual_dataset_data_dir}'
        dataset_dict[dataset] = os.listdir(data_dir)  # gets a list of all files in the data directory
        dfs = [pd.read_parquet(f'{data_dir}/{file}') for file in dataset_dict[dataset]]
        combined_df = pd.concat(dfs)

        # randomly sample from combined DataFrame
        sampled_df = combined_df.sample(
            n=number_of_samples,
            axis=0,
            replace=False,  # no duplicate samples
            random_state=42,
        )

        # split dataset into train and test
        train, test = train_test_split(
            sampled_df,
            shuffle=True,
            test_size=0.2,
            random_state=42,
        )

        # Save train and test files to directory
        os.makedirs(sampled_datasets_dir, exist_ok=True)
        train.to_parquet(path=f'{sampled_datasets_dir}/train_{dataset}_sampled.parquet', index=False)
        test.to_parquet(path=f'{sampled_datasets_dir}/test_{dataset}_sampled.parquet', index=False)

    if remove_raw_datasets:
        shutil.rmtree(raw_datasets_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sample Datasets')
    parser.add_argument('--delete-dir', type=bool, default=False, required=False)
    parser.add_argument('--num-samples', type=int, default=1000, required=False)

    args = parser.parse_args()

    generate_sampled_datasets(remove_raw_datasets=args.delete_dir, number_of_samples=args.num_samples)