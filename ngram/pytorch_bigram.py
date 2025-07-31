import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNGramTorch(nn.Module):
    def __init__(self, vocab_size, ngram_size=3, embedding_dim=64, hidden_dim=128, lr=1e-2, device=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hidden Layer
        input_dim = (ngram_size - 1) * embedding_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.05)

    def forward(self, x, targets=None):
        """
        x: (B, context_size)
        targets: (B,)
        """
        # Embedding lookup and flatten
        emb = self.embedding(x)             # (B, context_size, embedding_dim)
        emb = emb.view(x.size(0), -1)       # (B, context_size*embedding_dim)

        # Hidden layer with tanh
        h = torch.tanh(self.fc1(emb))

        # Output logits
        logits = self.fc2(h)                # (B, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def fit(self, train_data, valid_data, epochs=10, batch_size=32, patience=5, lr_decay=0.95):
        train_data = np.array(train_data, dtype=np.int64)
        valid_data = np.array(valid_data, dtype=np.int64)

        train_num_samples = len(train_data) - self.ngram_size + 1
        val_num_samples = len(valid_data) - self.ngram_size + 1

        # Prepare n-grams
        train_contexts = np.stack([train_data[i:i+self.ngram_size-1] for i in range(train_num_samples)])
        train_targets = np.array([train_data[i+self.ngram_size-1] for i in range(train_num_samples)])

        val_contexts = np.stack([valid_data[i:i+self.ngram_size-1] for i in range(val_num_samples)])
        val_targets = np.array([valid_data[i+self.ngram_size-1] for i in range(val_num_samples)])

        # Convert validation to torch (fixed)
        val_contexts = torch.tensor(val_contexts, dtype=torch.long, device=self.device)
        val_targets = torch.tensor(val_targets, dtype=torch.long, device=self.device)

        best_val_loss = float("inf")
        wait = 0
        best_state = None

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(train_num_samples)
            train_contexts = train_contexts[indices]
            train_targets = train_targets[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(train_num_samples / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min((b + 1) * batch_size, train_num_samples)

                x_batch = torch.tensor(train_contexts[start:end], dtype=torch.long, device=self.device)
                y_batch = torch.tensor(train_targets[start:end], dtype=torch.long, device=self.device)

                logits, loss = self.forward(x_batch, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * (end - start)

            epoch_loss /= train_num_samples

            # ---- Validation on full set ----
            with torch.no_grad():
                _, val_loss = self.forward(val_contexts, val_targets)
            val_loss = val_loss.item()

            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

            # ---- Early Stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping!")
                    self.load_state_dict(best_state)
                    break

            # ---- Learning rate decay ----
            for g in self.optimizer.param_groups:
                g["lr"] *= lr_decay

    def perplexity(self, data):
        data = np.array(data, dtype=np.int64)
        num_samples = len(data) - self.ngram_size + 1
        contexts = np.stack([data[i:i+self.ngram_size-1] for i in range(num_samples)])
        targets = np.array([data[i+self.ngram_size-1] for i in range(num_samples)])

        contexts = torch.tensor(contexts, dtype=torch.long, device=self.device)
        targets = torch.tensor(targets, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, _ = self.forward(contexts, targets)
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[torch.arange(num_samples), targets].mean().item()

        return np.exp(nll)