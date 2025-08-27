from gpt.gpt_model import GPT
import torch
import math

def calc_perplexity(model: GPT, data_loader, device):
    model.eval()
    nll_sum, tok_count = 0.0, 0
    with torch.no_grad():
        for idx, targets in data_loader:
            idx = idx.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, loss = model(idx, targets)
            n = (targets != -1).sum().item()
            nll_sum += loss.item() * n
            tok_count += n
    return math.exp(nll_sum / max(tok_count, 1))