from gpt.self_attention import SelfAttention
from torch import nn
from gpt.gelu import NewGELU

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # LayerNorm before self-attention
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # multi-head self-attention
        self.attn = SelfAttention(config)
        # LayerNorm before MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # feed-forward network (position-wise MLP)
        self.mlp = nn.ModuleDict(dict(
            # expand embedding dimension
            c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
            # non-linear activation
            act=NewGELU(),
            # project back to embedding size
            c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
            # dropout for regularization
            dropout=nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        # forward pass of the MLP
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        # residual connection around attention
        x = x + self.attn(self.ln_1(x))
        # residual connection around feed-forward network
        x = x + self.mlpf(self.ln_2(x))
        return x