import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
from keras.losses import sparse_categorical_crossentropy

def compute_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def compute_token_accuracy(y_true, y_pred):
    """
    Computes token-level accuracy.
    y_true: True sequences
    y_pred: Predicted sequence logits
    """
    predictions = np.argmax(y_pred, axis=-1)
    y_true_squeezed = np.squeeze(y_true)
    mask = y_true_squeezed != 0
    correct = (predictions == y_true_squeezed) & mask
    return np.sum(correct) / np.sum(mask) if np.sum(mask) > 0 else 0.0

def compute_perplexity(y_true, y_pred):
    """
    Computes overall perplexity using sparse categorical crossentropy.
    """
    loss = sparse_categorical_crossentropy(y_true, y_pred)
    return np.exp(np.mean(loss))