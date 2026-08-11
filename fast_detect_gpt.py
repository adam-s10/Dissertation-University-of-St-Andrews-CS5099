import os
import argparse
from datetime import timedelta
import time
import sys

import torch
import pandas as pd
import numpy as np

from models import load_model

def fast_detect_gpt(
        sample_model_name: str,
        data: pd.DataFrame,
        score_model_name: str | None = None
) -> tuple[list[float], list[timedelta]]:
    """
    Fast-DetectGPT analytical solution replicated from Bao et al. Citation [1] in submitted dissertation.
    :param sample_model_name: sampling model
    :param data: data
    :param score_model_name: scoring model
    :return: curvatures and runtimes
    """
    
    sampling_model, sampling_tokenizer, device = load_model(sample_model_name)


    if score_model_name:  # load different models in black-box
        scoring_model, scoring_tokenizer, _ = load_model(score_model_name)
    else:  # load the same models in white-box or as a choice
        scoring_model, scoring_tokenizer = sampling_model, sampling_tokenizer

    # set tokenizer settings
    scoring_tokenizer.clean_up_tokenization_spaces = False
    if scoring_tokenizer.pad_token is None:
        scoring_tokenizer.pad_token = scoring_tokenizer.eos_token

    sampling_tokenizer.clean_up_tokenization_spaces = False
    if sampling_tokenizer.pad_token is None:
        sampling_tokenizer.pad_token = sampling_tokenizer.eos_token
        

    curvature = []
    timings = []
    failure_tups = []
    num_failures = 0

    total_rows = len(data)
    row = 1
    for i in range(total_rows):
        row_start = time.time()
        print(f'------ Starting row {row:04d} of {total_rows:04d} ------')
        passage = data['document'].iloc[i]

        if not passage or not passage.strip():
            print(f'\n!----- Warning: an empty passage was detected at row {i}, skipping. -----!\n')
            curvature.append(np.nan)
            timings.append(np.nan)
            failure_tups.append(([i, len(passage), 'empty passage']))
            row += 1
            num_failures += 1
            continue
        
        try:
            # ------ Sample Passage ------
            # tokenized text and attention mask as 1 x n tensors where n is the sequence length and 1 represents the batch size
            sample_inputs = sampling_tokenizer(passage, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = sampling_model(**sample_inputs)
                
                # 1 x n x m tensor where 1 is the batch size, n is the sequence length, and m is the size of the vocabulary
                logits = outputs.logits.float()  # convert to 32 bit precision

            sample_shifted_logits = logits[:, :-1, :]  # shift the tokens by 1 as we can't check predictions past token n-1
            probs = torch.softmax(sample_shifted_logits, dim=-1)  # returns the normalized probability distribution of each token appearing. Size 1 x n-1 x m
    
            sampled_token_ids = torch.multinomial(  # samples 1 token per position from the probability distribution
                probs.squeeze(0),
                num_samples=1,
            ).squeeze(-1)  # returns tensor of length n-1
    
            log_probs = torch.log_softmax(sample_shifted_logits, dim=-1)  # log probability distribution of each token appearing. Size 1 x n-1 x m
    
            sampled_log_probs = (  # gathers the log probabilities of the tokens sampled
                log_probs.squeeze(0)
                .gather(dim=-1, index=sampled_token_ids.unsqueeze(-1))
                .squeeze(-1)
            )  # returns tensor of length n-1
            log_p = sampled_log_probs.sum().item()  # sum of log probabilities
    
            # this code block is not needed was just a visual during implementation
            # input_ids = sample_inputs['input_ids']
            # first_token = input_ids[:, :1]  # can't predict first token so just draw it from input
            # rest_of_tokens = sampled_token_ids.unsqueeze(0)  # 1 x n-1 tensor
            # x_tilde_ids = torch.cat(tensors=[first_token, rest_of_tokens], dim=-1)  # concatenates the first token to sampled tokens producing 1 x n tensor
            # x_tilde = sampling_tokenizer.decode(x_tilde_ids[0], skip_special_tokens=True)  # decodes token ids into words
            
    
            # ------ Score Passage ------
            score_inputs = scoring_tokenizer(passage, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = scoring_model(**score_inputs)
                logits = outputs.logits.float()  # convert to 32 bit precision
    
            score_shifted_logits = logits[:, :-1, :]
    
            # log probability distribution of each token appearing in scoring model. Size 1 x n-1 x m
            score_log_probs = torch.log_softmax(score_shifted_logits, dim=-1)

            if score_model_name:
                # returns the normalized probability distribution of each token appearing in sampling model. Size 1 x n-1 x m
                sample_probs = torch.softmax(score_shifted_logits, dim=-1)
            else:
                # if the model is both sample and score use the already calculated probabilities
                sample_probs = probs
    
    
            conditional_log_likelihood = (  # calculates the conditional probability of the scoring model observing each token in the input at each position
                score_log_probs
                .gather(dim=-1, index=score_inputs['input_ids'][:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )  # returns a 1 x n-1 size tensor
    
            mu_tilde = (sample_probs * score_log_probs).sum(dim=-1)
            sigma_tilde = torch.sqrt(
                (sample_probs * torch.square(score_log_probs)).sum(dim=-1) - torch.square(mu_tilde)
            )
            
            # ------ Calculate Discrepancy ------
            discrepancy = ((conditional_log_likelihood.sum(dim=-1) - mu_tilde.sum(dim=-1)) / sigma_tilde.sum(dim=-1)).mean()
            d_proba = discrepancy.item()
            curvature.append(d_proba)
    
            print(f'Log P: {log_p:.4f}')
            print(f'Discrepancy: {d_proba:.4f}')
    
            row_end = time.time()
            total_time_s = timedelta(seconds=row_end - row_start)
            timings.append(total_time_s)
            print(f'\n------ Finished row {row:04d}. Took {total_time_s} ------\n')
            row += 1

        except torch.OutOfMemoryError:
            print(f"!----- OOM Error for index {row}, skipping -----!")
            curvature.append(np.nan)
            timings.append(np.nan)
            failure_tups.append(([i, len(passage), "OOM Error"]))
            row += 1
            num_failures += 1
            torch.cuda.empty_cache()

    print('##########################################\n##########################################')
    print(f'------ INFO: {num_failures} row(s) failed during run ------')
    print('Failure index and length of passage that caused failure(s):')
    for i in failure_tups:
        print(i)
    return curvature, timings


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fast-DetectGPT')
    parser.add_argument('--sampling-model', type=str, required=True)
    parser.add_argument('--dataset', type=str , required=True)
    parser.add_argument('--save-file', type=str, required=True)
    parser.add_argument('--scoring-model', type=str, default=None, required=False)
    parser.add_argument('--dataset-dir', type=str, default='Generated Datasets', required=False)
    parser.add_argument('--save-dir', type=str, default='Results Files', required=False)

    args = parser.parse_args()

    if args.scoring_model and (args.scoring_model == args.sampling_model):
        user_input = ''
        while user_input != 'y' or user_input != 'n':
            user_input = input(
                f'Warning: you will load two of the same models as sample and score ({args.scoring_model}), '
                f'this is a waste of compute. Passing just --sample-model would use the same models as both '
                f'sample and score but only load the weights once. Would you like to proceed? [y/n]'
            ).lower()
            if user_input == 'n':
                sys.exit('------ Program terminated by user ------')
            if user_input == 'y':
                break

    dataset = pd.read_parquet(f'{args.dataset_dir}/{args.dataset}')

    fast_start = time.time()
    if not args.scoring_model:
        fast_results, runtime = fast_detect_gpt(args.sampling_model, dataset)
    else:
        fast_results, runtime = fast_detect_gpt(args.sampling_model, dataset, score_model_name=args.scoring_model)

    fast_end = time.time()

    print(f'###### Fast-DetectGPT took {timedelta(seconds=fast_end-fast_start)} ######')

    # verify returns are of the same length to not introduce nans or shifting of results
    assert len(fast_results) == len(dataset)
    assert len(runtime) == len(dataset)

    results = pd.DataFrame({
        'sampling_model': args.sampling_model,
        'scoring_model': args.scoring_model if args.scoring_model else args.sampling_model,
        'curvature': fast_results,
        'total_run_time_seconds': runtime
    })
    df = pd.concat(
        [
            results,
            dataset[['is_machine', 'generation_model']]
        ],
        axis=1
    )

    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f'{args.save_dir}/{args.save_file}', index=False)
