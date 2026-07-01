# imports
from models import load_model

# TODO: implement sampling algorithm using smallest chosen model as the baseline. Build step-by-step; see the forest for the trees...

# maybe convert to a class at later date?
def sample_passage(x, model_name):
    """
    Returns alternate passage using sampling model (q_phi) and probability chain of generating that passage.
    Step 1 from Algorithm 1 - REF HERE!

    :param x: passage
    :param model: sampling model
    :return: alternate passage, probability chain of passage
    """
    model, tokenizer = load_model(model_name)

    x_tilde = ''
    p = 1
    for i in range(len(x)):
        # sample t from p_theta (t | x_<i) <- sampling model
        # append t to x_tilde
        # p = p * p_theta (t | x_<i)
        pass

    return x_tilde, p


sample_passage('Hello there', 'qwen')