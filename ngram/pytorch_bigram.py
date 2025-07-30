import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.ngram.preprocessing import get_ngram_batch

class PytorchBigram(nn.Module):
    def __init__(self, vocab_size, ngram_size=2, lr=1e-3, device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.lr = lr

        # Each token directly mapped to logits (like a bigram table)
        self.embedding_layer = nn.Embedding(vocab_size, vocab_size)

        # Pick device automatically (MPS if available on Mac)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, targets=None):
        """
        x: (B, context_size)
        targets: (B,) or (B, context_size) depending on ngram
        """
        logits = self.embedding_layer(x)  # (B, context_size, vocab_size)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)      # flatten batch & time
        if targets is not None:
            targets = targets.view(B*T).to(self.device)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None

    def fit(self, data, epochs=10, steps_per_epoch=10000, batch_size=32, lr_decay=1.0):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            epoch_loss = 0.0

            for step in range(steps_per_epoch):
                # ---- Random batch ----
                x_batch, y_batch = get_ngram_batch(data, n=self.ngram_size, batch_size=batch_size)

                # Convert to tensors
                x_batch = torch.tensor(x_batch, dtype=torch.long, device=self.device)
                y_batch = torch.tensor(y_batch, dtype=torch.long, device=self.device)

                # ---- Forward pass ----
                _, loss = self.forward(x_batch, targets=y_batch)

                # ---- Backward & update ----
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Average loss for epoch
            epoch_loss /= steps_per_epoch

            # Learning rate decay (optional)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - lr: {optimizer.param_groups[0]['lr']:.6f}")