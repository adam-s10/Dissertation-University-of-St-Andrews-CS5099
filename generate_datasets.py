# imports
import os
import time
from datetime import timedelta
import argparse

import pandas as pd
import torch

from models import load_model


# Methodology:
# "sample 150 to 500 human-written examples per dataset as negative
# samples and produce equal numbers of positive samples by prompting the source model with the
# first 30 tokens of the human-written text"

generated_datasets_dir = 'Generated Datasets'


def generate_samples(
        model,
        tokenizer,
        device: str,
        model_name: str,
        file: str,
        dataset_dir: str,
        batch_size: int = 20
) -> None:
    """
    Adds positive examples of machine-generated content to for the datasets using the first 30 tokens of
    each row for xsum or the question along with the context for the others.

    *Tokens can be calculated differently across models, some models first 30 tokens may have more or less characters
    when compared against others.*

    :param model: loaded model
    :param tokenizer: loaded tokenizer
    :param device: device to load tensors to
    :param model_name: model name
    :param file: file
    :param dataset_dir: directory where file located
    :param batch_size: batch size

    :return: None
    """
    
    data = pd.read_parquet(f'{dataset_dir}/{file}')
    
    if 'xsum' in file:
        passages = data['document'].to_list()
        q_and_a_dataset = False
    else:
        passages = data['context'].to_list()
        q_and_a_dataset = True

    # supress warnings by adjusting settings correctly for model tokenization and generation
    tokenizer.padding_side = 'left'
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        passages,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)
    
    # extract the first 30 tokens of each passage
    if 'xsum' in file:
        generation_tokens = []
        for ids, mask in zip(tokens['input_ids'], tokens['attention_mask']):
            real_tokens = ids[mask.bool()][:30].to(device)

            if len(real_tokens) < 30:
                padding = torch.full(
                    (30-len(real_tokens),),
                    tokenizer.pad_token_id
                ).to(device)
                real_tokens = torch.cat([padding, real_tokens]).to(device)

            generation_tokens.append(real_tokens)
    
    else:  # or passes the context directly
        generation_tokens = tokens

    outputs = []
    epoch = 1
    total_epochs = len(data) // batch_size
    for i in range(0, len(data), batch_size):

        if 'xsum' in file:
            # extract the tokens for dataset rows according to batch size
            batch = generation_tokens[i:i+batch_size]
            input_ids = torch.stack(batch).to(device)
            # creates boolean tensor for if a token at index i is the pad token then converts to model.generate expected dtype
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        else:
            input_ids = tokens['input_ids'][i:i+batch_size].to(device)
            attention_mask = tokens['attention_mask'][i:i+batch_size].to(device)

        passage_lengths = (tokens['attention_mask'][i:i+batch_size].sum(dim=1).float())
        max_tokens = int(passage_lengths.quantile(0.75).item())  # set max tokens to be the third quantile of the current batch

        batch_output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.9
        )

        if q_and_a_dataset:
            # strip the prompts that are included for Q and A datasets, leaving just the output
            batch_output_tokens = [
                out[mask.sum() :] for out, mask in zip(batch_output_tokens, attention_mask)
            ]

        batch_outputs = tokenizer.batch_decode(
            batch_output_tokens, skip_special_tokens=True
        )
        outputs.extend(batch_outputs)
        torch.cuda.empty_cache()

        print(f'    --- Finished epoch {epoch:04d} of {total_epochs:04d} ---  ')
        epoch += 1

    df = pd.DataFrame({
        'document': outputs,
        'is_machine': 1
    })

    combined_df = pd.concat([data, df])
    combined_df = combined_df[['document', 'is_machine']]
    combined_df['generation_model'] = model_name
    
    os.makedirs(generated_datasets_dir, exist_ok=True)
    combined_df.to_parquet(path=f'{generated_datasets_dir}/{model_name}_{file.split('_sampled')[0]}_generated.parquet', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Generator')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=20, required=False)

    args = parser.parse_args()

    dataset_dir = 'Sampled Datasets'

    files = os.listdir(dataset_dir)
    print(f'\n------ Found {', '.join(files)} in provided directory "{dataset_dir}" ------\n')

    model, tokenizer, device = load_model(args.model)
    start = time.time()
    for file in files:
        print(f'\n------ Starting generation for file {file} ------\n')
        generate_samples(
            model,
            tokenizer,
            device,
            model_name=args.model,
            file=file,
            dataset_dir=dataset_dir,
            batch_size=args.batch_size
        )

    print(f'###### Sample generation for files in {dataset_dir} took {timedelta(seconds=time.time() - start)} ######')
