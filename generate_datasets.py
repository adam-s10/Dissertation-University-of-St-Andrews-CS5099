# imports
import os
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


generated_datasets_dir = 'Generated Datasets'


def generate_positive_samples(model_name: str, dataset_dir: str = 'Sampled Datasets', batch_size: int = 20) -> None:
    """
    Adds positive examples of machine-generated content to the datasets.

    *Tokens can be calculated differently across models,
    some models first 30 tokens may have more or less characters when compared against others.*

    :param model_name: model to generate samples
    :param dataset_dir: dataset directory to read and generate samples for
    :param batch_size: batch size
    :return: None
    """
    
    files = os.listdir(dataset_dir)
    print(f'------ Found {', '.join(files)} in provided directory {dataset_dir} ------')

    # todo: make this run sequentially for all datasets not one at a time
    model, tokenizer, device = load_model(model_name)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # make use of reserved memory to avoid torch.OutOfMemoryError
    for file in files:
        print(f'------ Starting generation for file {file} ------')

        data = pd.read_parquet(f'{dataset_dir}/{file}')
        data['is_machine'] = 0
        data = data[['document', 'is_machine']]
        passages = data.document.to_list()
    
        tokens = tokenizer(
            passages,
            return_tensors='pt',
            truncation=True,
            padding=True
        ).to(device)
    
        # todo: add argument for q and a dataset to use the question instead
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
            if epochs % (batch_size * 20) == 0:
                print(f'------ Epoch {epochs} complete ------')
        print(f'------ Took {timedelta(seconds=time.time() - start)} ------')
        
        df = pd.DataFrame({
            'document': outputs,
            'is_machine': 1
        })
        
        combined_df = pd.concat([data, df])
        os.makedirs(generated_datasets_dir, exist_ok=True)
        combined_df.to_parquet(path=f'{generated_datasets_dir}/{model_name}_{file}_generated.parquet', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset-dir', type=str, default='Sampled Datasets',  required=False)
    parser.add_argument('--batch-size', type=int, default=20, required=False)

    args = parser.parse_args()

    start = time.time()
    generate_positive_samples(model_name=args.model, dataset_dir=args.dataset_dir, batch_size=args.batch_size)

    print(f'###### Sample generation for files in {args.dataset_dir} took {timedelta(seconds=time.time() - start)} ######')
