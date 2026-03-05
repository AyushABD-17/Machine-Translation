from nltk.translate.bleu_score import sentence_bleu

def compute_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())