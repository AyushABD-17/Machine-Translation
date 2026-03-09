import numpy as np

def beam_search_decoder(predictions, k=3):
    """
    Implements a simple beam search over the sequence probabilities.
    In autoregressive models, this would take the top k probabilities
    at each step and branch out to search the likelihood space.
    
    Since the baseline models predict sequence probabilities simultaneously,
    we stub this for the architecture requirements.
    """
    # Placeholder: revert to greedy decoding to maintain stable API
    return np.argmax(predictions, axis=-1)

def greedy_decoder(predictions):
    """
    Basic greedy decoding mapping max probabilities to tokens.
    """
    return np.argmax(predictions, axis=-1)
