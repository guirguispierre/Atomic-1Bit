import torch
import torch.nn.functional as F


def sentence_completion_accuracy(model, dataset, vocab_size, device, num_samples=50):
    """
    Simple coherence heuristic: check if the model's top-1 prediction for the
    last token matches the ground truth, given the preceding context.

    Args:
        model: AtomicTransformer in eval mode
        dataset: Dataset with get_batch() method
        vocab_size: Vocabulary size
        device: torch device
        num_samples: Number of samples to test

    Returns:
        float: Accuracy (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_samples):
            x, y = dataset.get_batch(1)
            x, y = x.to(device), y.to(device)

            logits = model(x)

            # Check last non-padded position's prediction
            # Use the second-to-last position to predict the last token
            seq_len = x.shape[1]
            if seq_len < 2:
                continue

            predicted = logits[:, -2, :].argmax(dim=-1)
            target = y[:, -1]

            correct += (predicted == target).sum().item()
            total += 1

    return correct / max(total, 1)


def top_k_accuracy(model, dataset, vocab_size, device, k=5, num_samples=50):
    """
    Check if the ground truth token is within the model's top-k predictions.

    Returns:
        float: Top-k accuracy (0.0 to 1.0)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_samples):
            x, y = dataset.get_batch(1)
            x, y = x.to(device), y.to(device)

            logits = model(x)
            seq_len = x.shape[1]
            if seq_len < 2:
                continue

            topk = logits[:, -2, :].topk(k, dim=-1).indices
            target = y[:, -1].unsqueeze(-1)
            correct += (topk == target).any(dim=-1).sum().item()
            total += 1

    return correct / max(total, 1)
