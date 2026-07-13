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


# TODO: implement sampling algorithm using smallest chosen model as the baseline. Build step-by-step; see the forest for the trees...
# TODO: convert this to a class if we have time?

DEVICE = 'cpu'

# maybe convert to a class at later date?
def sample_passage(
        x:str,
        sampling_model,
        sampling_tokenizer
) -> tuple[str, float, torch.Tensor, torch.Tensor]:
    """
    Returns alternate passage using sampling model (q_phi) and probability chain of generating that passage.
    Step 1 from Algorithm 1 - REF HERE!

    :param x: passage
    :param sampling_model: sampling model
    :param sampling_tokenizer: sampling tokenizer
    :return: alternate passage, log probability chain of passage, x tilde token ids, sampling model probabilities
    """
    
    # ----------Skeleton Code----------

    # x_tilde = ''
    # p = 1
    # for i in range(len(x)):
    #     # sample t from p_theta (t | x_<i) <- sampling model
    #     # append t to x_tilde
    #     # p = p * p_theta (t | x_<i)
    #     pass
    #
    # return x_tilde, p

    # ----------Skeleton Code----------

    # tokenized text and attention mask as 1 x n tensors where n is the sequence length and 1 represents the batch size
    inputs = sampling_tokenizer(x, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = sampling_model(**inputs)
        # 1 x n x m tensor where 1 is the batch size, n is the sequence length, and m is the size of the vocabulary
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]  # shift the tokens by 1 as we can't check predictions past token n-1
    probs = torch.softmax(shifted_logits, dim=-1)  # returns the normalized probability distribution of each token appearing. Size 1 x n-1 x m

    sampled_token_ids = torch.multinomial(  # samples 1 token per position from the probability distribution
        probs.squeeze(0),
        num_samples=1,
    ).squeeze(-1)  # returns tensor of length n-1

    log_probs = torch.log_softmax(shifted_logits, dim=-1)  # log probability distribution of each token appearing. Size 1 x n-1 x m

    sampled_log_probs = (  # gathers the log probabilities of the tokens sampled
        log_probs.squeeze(0)
        .gather(dim=-1, index=sampled_token_ids.unsqueeze(-1))
        .squeeze(-1)
    )  # returns tensor of length n-1
    log_p = sampled_log_probs.sum().item()  # sum of log probabilities

    input_ids = inputs['input_ids']
    first_token = input_ids[:, :1]  # can't predict first token so just draw it from input
    rest_of_tokens = sampled_token_ids.unsqueeze(0)  # 1 x n-1 tensor
    x_tilde_ids = torch.cat(tensors=[first_token, rest_of_tokens], dim=-1)  # concatenates the first token to sampled tokens producing 1 x n tensor
    x_tilde = sampling_tokenizer.decode(x_tilde_ids[0], skip_special_tokens=True)  # decodes token ids into words

    return x_tilde, log_p, x_tilde_ids, probs


def score_passage(
        x:str,
        scoring_model,
        scoring_tokenizer,
        sampling_model_probabilities:torch.Tensor | None=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the conditional log probability calculated with the scoring model, mu tilde, and sigma tilde.
    All to be used between steps 2, 3, and 4 of Algorithm 1 - REF HERE! Here we use the improved methodology set
    out in Appendix B of the paper.

    :param x: passage
    :param scoring_model: scoring model
    :param scoring_tokenizer: scoring tokenizer
    :param sampling_model_probabilities: sample model probabilities
    :return: conditional log likelihood, mu tilde, sigma tilde
    """

    inputs = scoring_tokenizer(x, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = scoring_model(**inputs)
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]

    # log probability distribution of each token appearing in scoring model. Size 1 x n-1 x m
    score_log_probs = torch.log_softmax(shifted_logits, dim=-1)

    # returns the normalized probability distribution of each token appearing in sampling model. Size 1 x n-1 x m
    sample_probs = torch.softmax(shifted_logits, dim=-1) if sampling_model_probabilities is None else sampling_model_probabilities  # white vs. black-box

    conditional_log_likelihood = (  # calculates the conditional probability of the scoring model observing each token in the input at each position
        score_log_probs
        .gather(dim=-1, index=inputs['input_ids'][:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )  # returns a 1 x n-1 size tensor

    mu_tilde = (sample_probs * score_log_probs).sum(dim=-1)
    sigma_tilde = torch.sqrt(
        (sample_probs * torch.square(score_log_probs)).sum(dim=-1) - torch.square(mu_tilde)
    )

    return conditional_log_likelihood, mu_tilde, sigma_tilde


# todo: not a good idea passing around the tensors. Calculate in previous method
def calculate_conditional_probability_curvature(
        conditional_log_likelihood:torch.Tensor,
        mu_tilde:torch.Tensor,
        sigma_tilde:torch.Tensor
) -> float:
    """
    Calculate conditional log probability curvature (discrepancy). Equation 3 - REF HERE!

    :param conditional_log_likelihood: conditional log likelihood
    :param mu_tilde: mu tilde
    :param sigma_tilde: sigma tilde
    :return: conditional probability curvature
    """

    discrepancy = ((conditional_log_likelihood.sum(dim=-1) - mu_tilde.sum(dim=-1)) / sigma_tilde.sum(dim=-1)).mean()
    return discrepancy.item()


def fast_detect_gpt(
        sample_model_name:str,
        data:pd.DataFrame,
        score_model_name:str | None=None
) -> (list[float] | tuple[list[float], list[int]]):
    
    sampling_model, sampling_tokenizer, device = load_model(sample_model_name)
    DEVICE = device  # this may not be needed if we run only one function


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
        sample_inputs = sampling_tokenizer(passage, return_tensors='pt').to(DEVICE)
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
        score_inputs = scoring_tokenizer(passage, return_tensors='pt').to(DEVICE)
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
        )  # todo: producing a nan maybe square root of negative number?
        
        # ------ Calculate Discrepancy ------
        discrepancy = ((conditional_log_likelihood.sum(dim=-1) - mu_tilde.sum(dim=-1)) / sigma_tilde.sum(dim=-1)).mean()
        d_proba = discrepancy.item()
        curvature.append(d_proba)

        print(f'Log P: {log_p:.4f}')
        # print(f'Conditional Log Likelihood {conditional_log_likelihood}')
        # print(f'Mu tilde: {mu_tilde}')
        # print(f'Sigma tilde: {sigma_tilde}')  # todo: nan inside sigma tilde
        print(f'Discrepancy: {d_proba:.4f}')
        print()
        index = torch.nonzero(torch.isnan(sigma_tilde)).flatten()
        print(f'NaN at index {index}')
        print(f'Sample Probabilities {torch.index_select(sample_probs, dim=-1, index=index)}')
        print(f'Score Log Probabilities {torch.index_select(score_log_probs, dim=-1, index=index)}')
        print(f'Mu Tilde {torch.index_select(mu_tilde, dim=-1, index=index)}')
        print(f'Sigma Tilde {torch.index_select(sigma_tilde, dim=-1, index=index)}')
        # print(torch.index_select(conditional_log_likelihood, index=index))
        # print(torch.index_select(mu_tilde, index=index))
        # print(torch.index_select(sigma_tilde, index=index))
        epoch_end = time.time()
        print(f'------ Finished Epoch {epoch}. Took {timedelta(seconds=epoch_end - epoch_start)} ------')
        print()
        if epoch % 100 == 0:
            print('------ Processed 100 rows, clearing KV Cache as precaution against OOM Error ------')
            torch.cuda.empty_cache()
        epoch += 1

    return curvature


if __name__ == '__main__':
    
    # todo: add runtime error for when white box and scoring model passed
    # todo: add runtime error

    parser = argparse.ArgumentParser(description='Run Fast-DetectGPT')
    # parser.add_argument('--white-box', type=bool, required=True)
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
                f'sample and score but only load the weights once, would you like to proceed? [y/n]'
            ).lower()
            if input_ == 'n':
                sys.exit('------ Program terminated by user ------')
            if input_ == 'y':
                break

    dataset = pd.read_parquet(f'{args.dataset_dir}/{args.dataset}')
    # todo: handle thresholding in another script to avoid two runs

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
        'curvatures': fast_results
    })
    df = pd.concat([results, dataset['is_machine']], axis=1)

    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(f'{args.save_dir}/{args.save_file}', index=False)



    # model_name = 'qwen_ft'
    # model, tokenizer, device = load_model(model_name)
    # DEVICE = device
    # # passage = 'The quick brown fox jumps over the lazy dog.'
    # data = pd.read_parquet('Generated Datasets/qwen_ft_xsum_generated.parquet')
    # passages = data.document.to_list()
    # THRESHOLD = -0.1984
    # results = {
    #     'threshold': THRESHOLD,
    # }
    # predictions = []
    # curvature = []
    # start = time.time()
    # for i in range(len(passages)):
    #     x_tilde, log_p, token_ids, sample_probs = sample_passage(data.document.iloc[i], model, tokenizer)
    #     print(f'Original: {data.document.iloc[i]}')
    #     print(f'Alternative: {x_tilde}')
    #     print(f'Token ids: {token_ids}')
    #     print(f'Log p: {log_p:.8f}')
    #
    #     con_log_prob, mu_tilde, sigma_tilde = score_passage(data.document.iloc[i], model, tokenizer, sampling_model_probabilities=sample_probs)
    #     print(f'Conditional Log Probability: {con_log_prob}')
    #     print(f'Mu tilde: {mu_tilde}')
    #     print(f'Sigma tilde: {sigma_tilde}')
    #
    #     discrepancy = calculate_conditional_probability_curvature(con_log_prob, mu_tilde, sigma_tilde)
    #     print(f'Conditional Probability Curvature: {discrepancy:.8f}')
    #     curvature.append(discrepancy)
    #     predictions.append(0 if discrepancy < THRESHOLD else 1)
    # end = time.time()
    # print(f"For loop took {timedelta(seconds=end - start)}")
    #
    #
    # # TODO: improve code quality from here
    # results['predictions'] = predictions
    # results['curvature'] = curvature
    # df = pd.DataFrame(results)
    # df = pd.concat([df, data['is_machine']], axis=1)
    #
    # results_files_dir = 'Results Files'
    # os.makedirs(results_files_dir, exist_ok=True)
    # df.to_csv(f'{results_files_dir}/{model_name}_xsum_results2.csv', index=False)
    # # todo: to here
