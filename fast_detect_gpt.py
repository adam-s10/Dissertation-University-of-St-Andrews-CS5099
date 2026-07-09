# imports
import numpy as np
import os
from models import load_model
import torch
import pandas as pd

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


if __name__ == '__main__':

    model_name = 'qwen'
    model, tokenizer, device = load_model(model_name)
    DEVICE = device
    # passage = 'The quick brown fox jumps over the lazy dog.'
    data = pd.read_parquet('Generated Datasets/qwen_ft_xsum_generated.parquet')
    passages = data.document.to_list()
    import time
    from datetime import timedelta
    THRESHOLD = .8
    results = {
        'threshold': THRESHOLD,
    }
    predictions = []
    start = time.time()
    for i in range(len(passages)):
        x_tilde, log_p, token_ids, sample_probs = sample_passage(data.document.iloc[i], model, tokenizer)
        print(f'Original: {data.document.iloc[i]}')
        print(f'Alternative: {x_tilde}')
        print(f'Token ids: {token_ids}')
        print(f'Log p: {log_p:.8f}')

        con_log_prob, mu_tilde, sigma_tilde = score_passage(data.document.iloc[i], model, tokenizer, sampling_model_probabilities=sample_probs)
        print(f'Conditional Log Probability: {con_log_prob}')
        print(f'Mu tilde: {mu_tilde}')
        print(f'Sigma tilde: {sigma_tilde}')

        discrepancy = calculate_conditional_probability_curvature(con_log_prob, mu_tilde, sigma_tilde)
        print(f'Conditional Probability Curvature: {discrepancy:.8f}')

        predictions.append(1 if discrepancy > np.log(THRESHOLD) else 0)
    end = time.time()
    print(f"For loop took {timedelta(seconds=end - start)}")


    # TODO: improve code quality from here
    results['predictions'] = predictions
    df = pd.DataFrame(results)
    df = pd.concat([df, data['is_machine']], axis=1)
    
    results_files_dir = 'Results Files'
    os.makedirs(results_files_dir, exist_ok=True)
    df.to_csv(f'{results_files_dir}/{model_name}_xsum_results.csv', index=False)
    # todo: to here
