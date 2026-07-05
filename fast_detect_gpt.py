# imports
from models import load_model
import torch
import numpy as np

# TODO: implement sampling algorithm using smallest chosen model as the baseline. Build step-by-step; see the forest for the trees...

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    inputs = sampling_tokenizer(x, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = sampling_model(**inputs)
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]
    
    probs = torch.softmax(shifted_logits, dim=-1)
    sampled_token_ids = torch.multinomial(
        probs.squeeze(0),
        num_samples=1,
    ).squeeze(-1)

    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    sampled_log_probs = (
        log_probs.squeeze(0)
        .gather(dim=-1, index=sampled_token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    log_p = sampled_log_probs.sum().item()

    input_ids = inputs['input_ids']
    first_token = input_ids[:, :1]  # can't predict first token so just draw it from input
    rest_of_tokens = sampled_token_ids.unsqueeze(0)
    x_tilde_ids = torch.cat(tensors=[first_token, rest_of_tokens], dim=-1)
    x_tilde = sampling_tokenizer.decode(x_tilde_ids[0], skip_special_tokens=True)

    return x_tilde, log_p, x_tilde_ids, probs


def score_passage(
        x:str,
        scoring_model,
        scoring_tokenizer,
        sampling_model_probabilities:torch.Tensor | None=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """

    :param x: passage
    :param scoring_model: scoring model
    :param scoring_tokenizer: scoring tokenizer
    :param sampling_model_probabilities: sample model probabilities
    :return: conditional log likelihood, mu tilde, sigma tilde
    """

    inputs = scoring_tokenizer(x, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = scoring_model(**inputs)
        logits = outputs.logits

    shifted_logits = logits[:, :-1, :]

    score_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    sample_probs = torch.softmax(shifted_logits, dim=-1) if sampling_model_probabilities is None else sampling_model_probabilities  # white vs. black-box

    # print(score_log_probs, score_log_probs.shape)
    # print(inputs['input_ids'], inputs['input_ids'].shape)
    # print(inputs['input_ids'][:, 1:])
    # print(inputs["input_ids"][:, 1:].unsqueeze(-1))
    

    conditional_log_likelihood = (
        score_log_probs
        .gather(dim=-1, index=inputs['input_ids'][:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )

    mu_tilde = (sample_probs * score_log_probs).sum(dim=-1)
    sigma_tilde = torch.sqrt(
        (sample_probs * torch.square(score_log_probs)).sum(dim=-1) - torch.square(mu_tilde)
    )  # make sure this can't underflow: 0.0000e+00 was observed

    return conditional_log_likelihood, mu_tilde, sigma_tilde


#todo: forward pass
def forward_pass():
    pass


if __name__ == '__main__':

    model_name = 'qwen'
    model, tokenizer = load_model(model_name)
    passage = 'The quick brown fox jumps over the lazy dog.'

    x_tilde, log_p, token_ids, sample_probs = sample_passage(passage, model, tokenizer)
    print(f'Original: {passage}')
    print(f'Alternative: {x_tilde}')
    print(f'Token ids: {token_ids}')
    print(f'Log p: {log_p:.8f}')

    con_log_prob, mu_tilde, sigma_tilda = score_passage(passage, model, tokenizer, sampling_model_probabilities=sample_probs)
    print(f'Conditional Log Probability: {con_log_prob}')
    print(f'Mu tilde: {mu_tilde}')
    print(f'Sigma tilde: {sigma_tilda}')
