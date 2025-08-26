import torch
from torch import nn
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection
    at the end.
    It's important in decoder block to have diagonal mask
    It is also possible to use torch.nn.MultiheadAttention.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.flash = hasattr(
            torch.nn.functional,
            'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size)
                           ).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # batch_size, seq_len, emb_dim
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (b, seq_len, emb_dim) --> (b, seq_len, emb_dim * 3) --> (b, seq_len, emb_dim)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (b, h, seq_len, d_k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # (b, h, seq_len, d_k) matmul (b, h, d_k, seq_len) --> (b, h, seq_len, seq_len)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # diagonal mask
            # fill 0 mask with super small number so it wont affect the softmax weight
            # (batch_size, h, seq_len, seq_len)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # (b, h, seq_len, seq_len) matmul (b, h, seq_len, d_k) --> (b, h, seq_len, d_k)
            y = att @ v

            # (b, h, seq_len, d_k) --> (b, seq_len, h, d_k) --> (b, seq_len, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y