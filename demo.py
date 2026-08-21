import argparse
import time

from demo_data import (
    a_study_in_scarlet,
    a_study_in_scarlet_generated,
    just_in_time,
    just_in_time_generated,
    the_adventures_of_huckleberry_finn,
    the_adventures_of_huckleberry_finn_generated
)
from models import load_model

import torch

line_break = '-------------------------------------------\n-------------------------------------------'
passage_break = '################################'
HUMAN = 'Human Generated:'
MACHINE = 'Machine Generated:'

def fast_detect_gpt(model_name: str, passage: str):
    
    model, tokenizer, device = load_model(model_name)

    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    start = time.time()
    # ------ Sample Passage ------
    # tokenized text and attention mask as 1 x n tensors where n is the sequence length and 1 represents the batch size
    sample_inputs = tokenizer(passage, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**sample_inputs)
        
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
    input_ids = sample_inputs['input_ids']
    first_token = input_ids[:, :1]  # can't predict first token so just draw it from input
    rest_of_tokens = sampled_token_ids.unsqueeze(0)  # 1 x n-1 tensor
    x_tilde_ids = torch.cat(tensors=[first_token, rest_of_tokens], dim=-1)  # concatenates the first token to sampled tokens producing 1 x n tensor
    x_tilde = tokenizer.decode(x_tilde_ids[0], skip_special_tokens=True)  # decodes token ids into words
    

    # ------ Score Passage ------
    score_inputs = tokenizer(passage, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**score_inputs)
        logits = outputs.logits.float()  # convert to 32 bit precision

    score_shifted_logits = logits[:, :-1, :]

    # log probability distribution of each token appearing in scoring model. Size 1 x n-1 x m
    score_log_probs = torch.log_softmax(score_shifted_logits, dim=-1)


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

    # print(passage_break)
    # print(passage_break)
    # print(passage)
    # print(passage_break)
    # print(x_tilde)
    # print(passage_break)
    # print(passage_break)
    
    print(f'Sample Model Log Probability (Log P): {log_p:.4f}')
    print(f'Conditional Probability Curvature of Text: {discrepancy.item():.4f}')
    print(f'Fast-DetectGPT took {time.time() - start:.4f} seconds')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast-DetectGPT Demo')
    parser.add_argument('--model', type=str, required=True)
    
    args = parser.parse_args()
    
    print(line_break)
    print('A Study In Scarlet:')
    print(HUMAN)
    fast_detect_gpt(model_name=args.model, passage=a_study_in_scarlet)
    print(MACHINE)
    fast_detect_gpt(model_name=args.model, passage=a_study_in_scarlet_generated)
    
    print(line_break)
    print('Just In Time:')
    print(HUMAN)
    fast_detect_gpt(model_name=args.model, passage=just_in_time)
    print(MACHINE)
    fast_detect_gpt(model_name=args.model, passage=just_in_time_generated)
    
    print(line_break)
    print('The Adventures of Huckleberry Finn:')
    print(HUMAN)
    fast_detect_gpt(model_name=args.model, passage=the_adventures_of_huckleberry_finn)
    print(MACHINE)
    fast_detect_gpt(model_name=args.model, passage=the_adventures_of_huckleberry_finn_generated)
