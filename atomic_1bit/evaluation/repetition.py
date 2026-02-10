from collections import Counter


def calculate_repetition_rate(text, n=3):
    """
    Measure n-gram repetition rate in generated text.

    Args:
        text: Generated text string
        n: N-gram size (default 3)

    Returns:
        float: Repetition rate (0.0 = all unique, 1.0 = all repeated)
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    unique_ngrams = set(ngrams)

    if len(ngrams) == 0:
        return 0.0

    return 1.0 - (len(unique_ngrams) / len(ngrams))


def calculate_repetition_rates(text, max_n=5):
    """
    Calculate repetition rates for n-grams from 1 to max_n.

    Returns:
        dict: {n: repetition_rate}
    """
    return {n: calculate_repetition_rate(text, n) for n in range(1, max_n + 1)}
