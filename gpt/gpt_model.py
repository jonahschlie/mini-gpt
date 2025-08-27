import torch
from torch import nn
from gpt.block import Block
import math
import torch.nn.functional as F
from dataclasses import asdict

class GPT(nn.Module):
    """
    Minimal GPT-style language model with:
    - Token + position embeddings
    - Stack of transformer Blocks
    - Final LayerNorm and linear LM head
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        # Keep a lightweight, serializable view of the config for checkpoints
        try:
            self.config_dict = asdict(config)
        except Exception:
            self.config_dict = getattr(config, "__dict__", None)

        # Core transformer modules
        self.transformer = nn.ModuleDict(dict(
            # token embeddings
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # positional embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # Dropout Layer
            drop=nn.Dropout(config.embd_pdrop),
            # transformer blocks
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """


        Args:
            idx (LongTensor): token ids, shape (B, T)
            targets (LongTensor|None): same shape as idx, -1 positions are ignored
        Returns:
            logits (FloatTensor): (B, T, vocab_size)
            loss (FloatTensor|None): mean CE over valid positions, if targets provided
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Sequence length {t} exceeds block size {self.block_size}"

        # Positions 0..T-1 (shape (1, T))
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Embedding + dropout
        tok_emb = self.transformer.wte(idx)   # (B, T, C)
        pos_emb = self.transformer.wpe(pos)   # (1, T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final norm and LM head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        # Optional loss (ignore_index=-1 masks padding/unused positions)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Autoregressively generate tokens.

        Args:
            idx (LongTensor): Conditioning token ids of shape (B, T_start). Acts as the prompt.
            max_new_tokens (int): Number of tokens to generate and append to the prompt.
            temperature (float, optional): Logit scaling factor (>0). Higher = more random.
                Use 1.0 for neutral scaling; lower than 1.0 for more deterministic behavior.
            do_sample (bool, optional): If True, sample next token from the softmax distribution.
                If False, use greedy decoding (argmax).
            top_k (int | None, optional): If set, restrict sampling/greedy to the top-k logits at each step.

        Returns:
            LongTensor: Generated token ids of shape (B, T_start + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Next-token logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering (keep top k, set rest to -inf)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            # Probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or greedy pick
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Append and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def save_weights(self, path):
        """
        Save model state_dict and minimal metadata.
        extra: optional dict merged into meta.
        """
        payload = {
            "state_dict": self.state_dict(),
            "meta": {
                "class": "GPT",
                "config": self.config_dict
            }
        }
        torch.save(payload, path)

    def load_weights(self, path, map_location=None, strict: bool = True):
        """
        Load weights from a file produced by save_weights.
        Returns the stored 'meta' dict (empty if missing).
        """
        ckpt = torch.load(path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=strict)
        return ckpt.get("meta", {})