from gpt.gpt_model import GPT
import torch
import math


def calc_perplexity(model: GPT, data_loader, device):
    """
    Computes the perplexity for a given model and dataset.

    Args:
        model: The GPT model to evaluate.
        data_loader: Iterator that yields batches of inputs and targets.
        device: Device on which to run computations (CPU or GPU).

    Returns:
        float: The computed perplexity.
    """

    # Switch model to evaluation mode
    model.eval()

    # Accumulators for negative log-likelihood (NLL) sum and token count
    nll_sum, tok_count = 0.0, 0

    with torch.no_grad():
        for idx, targets in data_loader:
            # Move inputs and targets to the correct device
            idx = idx.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass through the model, returns logits and loss
            logits, loss = model(idx, targets)

            # Count number of valid tokens
            n = (targets != -1).sum().item()

            # Loss is mean NLL → multiply by token count to get total NLL
            nll_sum += loss.item() * n
            tok_count += n

    # average NLL per token
    return math.exp(nll_sum / max(tok_count, 1))