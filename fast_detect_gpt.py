# imports
import numpy as np
import os
import argparse
from models import load_model
import torch
import pandas as pd
from datetime import timedelta
import time
import sys


def fast_detect_gpt(
        sample_model_name: str,
        data:pd.DataFrame,
        score_model_name: str | None = None
) -> (list[float] | tuple[list[float], list[int]]):
    
    sampling_model, sampling_tokenizer, device = load_model(sample_model_name)


    if score_model_name:  # load different models in black-box
        scoring_model, scoring_tokenizer, _ = load_model(score_model_name)
    else:  # load the same models in white-box or as a choice
        scoring_model, scoring_tokenizer = sampling_model, sampling_tokenizer

    curvature = []
    
    total_epochs = len(data)
    epoch = 1
    for i in range(total_epochs):
        epoch_start = time.time()
        print(f'------ Starting Epoch {epoch} of {total_epochs} ------')
        passage = data['document'].iloc[i]
        
        # ------ Sample Passage ------
        # tokenized text and attention mask as 1 x n tensors where n is the sequence length and 1 represents the batch size
        sample_inputs = sampling_tokenizer(passage, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = sampling_model(**sample_inputs)
            
            # 1 x n x m tensor where 1 is the batch size, n is the sequence length, and m is the size of the vocabulary
            logits = outputs.logits.float()  # convert to 32 bit precision
        # todo: can talk about how bf16 resulted in underflow so needed to use 32 bit precision
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
        input_ids = sample_inputs['input_ids']
        first_token = input_ids[:, :1]  # can't predict first token so just draw it from input
        rest_of_tokens = sampled_token_ids.unsqueeze(0)  # 1 x n-1 tensor
        x_tilde_ids = torch.cat(tensors=[first_token, rest_of_tokens], dim=-1)  # concatenates the first token to sampled tokens producing 1 x n tensor
        x_tilde = sampling_tokenizer.decode(x_tilde_ids[0], skip_special_tokens=True)  # decodes token ids into words
        
        
        # ------ Score Passage ------
        score_inputs = scoring_tokenizer(passage, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = scoring_model(**score_inputs)
            logits = outputs.logits.float()  # convert to 32 bit precision

        score_shifted_logits = logits[:, :-1, :]

        # log probability distribution of each token appearing in scoring model. Size 1 x n-1 x m
        score_log_probs = torch.log_softmax(score_shifted_logits, dim=-1)

        # returns the normalized probability distribution of each token appearing in sampling model. Size 1 x n-1 x m
        sample_probs = torch.softmax(score_shifted_logits, dim=-1)

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

        epoch_end = time.time()
        print(f'------ Finished Epoch {epoch}. Took {timedelta(seconds=epoch_end - epoch_start)} ------')
        print()
        if epoch % 100 == 0:
            print('------ Processed 100 rows, clearing KV Cache as precaution against OOM Error ------')
            torch.cuda.empty_cache()
        epoch += 1

    return curvature


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Fast-DetectGPT')
    parser.add_argument('--sampling-model', type=str, required=True)
    parser.add_argument('--dataset', type=str , required=True)
    parser.add_argument('--generation-model', type=str, required=True)
    parser.add_argument('--save-file', type=str, required=True)
    parser.add_argument('--scoring-model', type=str, default=None, required=False)
    parser.add_argument('--dataset-dir', type=str, default='Generated Datasets', required=False)
    parser.add_argument('--save-dir', type=str, default='Results Files', required=False)

    args = parser.parse_args()

    if args.scoring_model and (args.scoring_model == args.sampling_model):
        input_ = ''
        while input_ != 'y' or input_ != 'n':
            input_ = input(
                f'Warning: you will load two of the same models as sample and score ({args.scoring_model}), '
                f'this is a waste of compute. Passing just --sample-model would use the same models as both '
                f'sample and score but only load the weights once. Would you like to proceed? [y/n]'
            ).lower()
            if input_ == 'n':
                sys.exit('------ Program terminated by user ------')
            if input_ == 'y':
                break

    dataset = pd.read_parquet(f'{args.dataset_dir}/{args.dataset}')

    fast_start = time.time()
    if not args.scoring_model:
        fast_results = fast_detect_gpt(args.sampling_model, dataset)
    else:
        fast_results = fast_detect_gpt(args.sampling_model, dataset, score_model_name=args.scoring_model)

    fast_end = time.time()

    print(f'###### Fast-DetectGPT took {timedelta(seconds=fast_end-fast_start)} ######')

    results = pd.DataFrame({
        'generating_model': args.generation_model,
        'sampling_model': args.sampling_model,
        'scoring_model': args.scoring_model if args.scoring_model else args.sampling_model,
        'curvature': fast_results
    })
    df = pd.concat([results, dataset['is_machine']], axis=1)

    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f'{args.save_dir}/{args.save_file}', index=False)
