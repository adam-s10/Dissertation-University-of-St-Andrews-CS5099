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


def generate_samples_deprecated(
    model_name: str, file_: str, dataset_dir: str, batch_size: int = 20
) -> None:
    """
    Adds positive examples of machine-generated content to the datasets for the pubmed dataset. Because pubmed is a
    q and a dataset, we use the whole unedited question.

    :param model_name: model
    :param file_: file
    :param dataset_dir: directory where file located
    :param batch_size: batch size
    :return: None
    """

    model, tokenizer, device = load_model(model_name)

    # todo: edit this to handle the different types of datasets
    data = pd.read_parquet(f"{dataset_dir}/{file_}")
    # handled in process_datasets.py
    # data['is_machine'] = 0
    # data = data[['document', 'is_machine']]
    passages = data["document"].to_list()

    # pubmed dataset does not
    tokens = tokenizer(passages, return_tensors="pt", truncation=True, padding=True).to(
        device
    )

    outputs = []
    total_epochs = len(data)
    epoch = 0
    for i in range(0, total_epochs, batch_size):
        # extract the tokens for dataset rows according to batch size
        input_ids = tokens["input_ids"][i : i + batch_size].to(device)
        attention_mask = tokens["attention_mask"][i : i + batch_size].to(device)

        passage_lengths = (
            tokens["attention_mask"][i : i + batch_size].sum(dim=1).float()
        )
        max_tokens = int(
            passage_lengths.quantile(0.75).item()
        )  # set max tokens to be the third quantile of the current batch

        batch_output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

        batch_outputs = tokenizer.batch_decode(
            batch_output_tokens, skip_special_tokens=True
        )
        outputs.extend(batch_outputs)
        torch.cuda.empty_cache()
        epoch += 1

        if epoch % 10 == 0:
            print(
                f"------ Finished epoch {epoch} of {total_epochs} for file {file_} ------"
            )


# def generate_positive_samples(model_name: str, path_to_file: str, batch_size: int = 20) -> None:
#     """
#     Adds positive examples of machine-generated content to the datasets.
#
#     *Tokens can be calculated differently across models,
#     some models first 30 tokens may have more or less characters when compared against others.*
#
#     :param model_name: model to generate samples
#     :param path_to_file: path to file to generate samples for
#     :param batch_size: batch size
#     :return: None
#     """
#
#     model, tokenizer, device = load_model(model_name)
#     # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # make use of reserved memory to avoid torch.OutOfMemoryError
#
#     data = pd.read_parquet(f'{dataset_dir}/{file}')
#     data['is_machine'] = 0
#     data = data[['document', 'is_machine']]
#     passages = data['document'].to_list()
#
#     tokens = tokenizer(
#         passages,
#         return_tensors='pt',
#         truncation=True,
#         padding=True
#     ).to(device)
#
#     if 'pubmedqa' in file:
#         generation_tokens = tokens
#     else:
#         generation_tokens = [
#             token_id[mask.bool()][:30] for token_id, mask in zip(tokens['input_ids'], tokens['attention_mask'])  # extracts the first 30 tokens that are not padded (attention_mask index != 0)
#         ]
#
#     outputs = []
#     start = time.time()
#     epochs = 0
#     for i in range(0, len(generation_tokens), batch_size):
#         batch = generation_tokens[i:i+batch_size]
#         # use known structure of tokenizer return to avoid decode and re-tokenize pipeline
#         input_ids = torch.stack(batch).to(device)  # tensor of size k x n where k is the batch size and n is number of tokens (30)
#         attention_mask = torch.ones_like(input_ids).to(device)  # all tensors are the same size so no padding is used so attention mask will be all 1's
#         inputs = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask
#         }
#
#         passage_lengths = tokens['attention_mask'][i:i+batch_size].sum(dim=1)
#         max_tokens = passage_lengths.quantile(.75).item()  # set max tokens to be the third quantile of the batch
#
#         batch_output_tokens = model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             repetition_penalty=1.3,
#             no_repeat_ngram_size=4,
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.9,
#         )
#
#         batch_outputs = tokenizer.batch_decode(batch_output_tokens, skip_special_tokens=True)
#         outputs.extend(batch_outputs)
#         torch.cuda.empty_cache()
#         epochs += batch_size
#         if epochs % (batch_size * 20) == 0:
#             print(f'------ Epoch {epochs} complete ------')
#     print(f'------ Took {timedelta(seconds=time.time() - start)} ------')
#
#     df = pd.DataFrame({
#         'document': outputs,
#         'is_machine': 1
#     })
#
#     combined_df = pd.concat([data, df])
#     os.makedirs(generated_datasets_dir, exist_ok=True)
#     combined_df.to_parquet(path=f'{generated_datasets_dir}/{model_name}_{file}_generated.parquet', index=False)

def generate_samples(model, tokenizer, device: str, model_name: str, file: str, dataset_dir: str, batch_size: int = 20) -> None:
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
    # handled in process_datasets.py
    # data['is_machine'] = 0
    # data = data[['document', 'is_machine']]
    
    if 'xsum' in file:
        passages = data['document'].to_list()
    else:
        passages = data['context'].to_list()

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

    for i in range(0, len(data), batch_size):

        if 'xsum' in file:
            # extract the tokens for dataset rows according to batch size
            batch = generation_tokens[i:i+batch_size]
            input_ids = torch.stack(batch).to(device)
            # creates boolean tensor for if a token at index i is the pad token then converts to model.generate expected dtype
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            q_and_a_dataset = False

        else:
            input_ids = tokens['input_ids'][i:i+batch_size].to(device)
            attention_mask = tokens['attention_mask'][i:i+batch_size].to(device)
            q_and_a_dataset = True

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

        print(f'------ Finished epoch {epoch} for file {file} ------')
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
    parser.add_argument('--dataset-dir', type=str, default='Sampled Datasets',  required=False)
    parser.add_argument('--batch-size', type=int, default=20, required=False)

    args = parser.parse_args()

    files = os.listdir(args.dataset_dir)
    print(f'\n------ Found {', '.join(files)} in provided directory "{args.dataset_dir}" ------\n')

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
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size
        )

    print(f'###### Sample generation for files in {args.dataset_dir} took {timedelta(seconds=time.time() - start)} ######')
