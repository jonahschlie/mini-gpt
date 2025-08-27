import torch
from torch import nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi-head masked self-attention with an output projection.
    Uses Flash Attention kernels if available (PyTorch >= 2.0),
    otherwise falls back to a slower implementation with an explicit mask.
    """

    def __init__(self, config):
        super().__init__()

        # sanity check: embedding dimension must be divisible by number of heads
        assert config.n_embd % config.n_head == 0

        # linear layer to compute queries, keys, and values in one pass
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection layer
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # dropout for attention weights and residual connection
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # check if Flash Attention kernels are available
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # if not available, register a causal mask buffer
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            mask = mask.view(1, 1, config.block_size, config.block_size)
            self.register_buffer("mask", mask)

    def forward(self, x):
        # input has shape (batch_size, seq_len, emb_dim)
        B, T, C = x.size()

        # compute queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # attention computation
        if self.flash:
            # efficient Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # scaled dot-product between queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # apply causal mask so tokens cannot attend to future positions
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

            # normalize attention weights
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # weighted sum of values
            y = att @ v

        # reshape back to (batch_size, seq_len, emb_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection with dropout
        y = self.resid_dropout(self.c_proj(y))

        return y