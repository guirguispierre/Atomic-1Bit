from collections import Counter


def unique_ngram_ratio(text, n=1):
    """
    Calculate the ratio of unique n-grams to total n-grams.

    Args:
        text: Generated text string
        n: N-gram size

    Returns:
        float: Ratio (1.0 = maximally diverse, lower = more repetitive)
    """
    tokens = text.split()
    if len(tokens) < n:
        return 1.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if len(ngrams) == 0:
        return 1.0

    return len(set(ngrams)) / len(ngrams)


def vocabulary_utilization(texts, total_vocab_size):
    """
    Measure what fraction of the vocabulary is actually used across generated texts.

    Args:
        texts: List of generated text strings
        total_vocab_size: Size of the model's vocabulary

    Returns:
        float: Fraction of vocab used (0.0 to 1.0)
    """
    all_tokens = set()
    for text in texts:
        all_tokens.update(text.split())
    return len(all_tokens) / total_vocab_size


def diversity_metrics(text):
    """
    Compute a suite of diversity metrics for a single text.

    Returns:
        dict with unigram_ratio, bigram_ratio, trigram_ratio
    """
    return {
        "unigram_ratio": unique_ngram_ratio(text, 1),
        "bigram_ratio": unique_ngram_ratio(text, 2),
        "trigram_ratio": unique_ngram_ratio(text, 3),
    }
