import torch
import torch.nn.functional as F


def calculate_perplexity(model, dataset, vocab_size, device, num_samples=200):
    """
    Evaluate perplexity on a held-out split.

    Args:
        model: AtomicTransformer in eval mode
        dataset: Dataset with get_batch(batch_size) method
        vocab_size: Vocabulary size for cross-entropy
        device: torch device
        num_samples: Number of samples to evaluate

    Returns:
        float: Perplexity value
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(num_samples):
            x, y = dataset.get_batch(1)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction='sum')
            total_nll += loss.item()
            total_tokens += y.numel()

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item()
