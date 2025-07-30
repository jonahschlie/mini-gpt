import numpy as np

class NeuralBigram:

    def __init__(self, embedding_dimension, vocab_size, ngram_size, lr=1e-2):
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.lr = lr

        # Embedding: vocab_size → embedding_dimension
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dimension) * 0.01

        # Linear layer: (context_size * embedding_dim) → vocab_size
        input_dim = (self.ngram_size - 1) * self.embedding_dimension
        self.linear_W = np.random.randn(input_dim, vocab_size) * 0.01
        self.linear_b = np.zeros((1, vocab_size))

    def forward(self, x, y=None, target=True):
        # Embedding lookup
        self.embeddings = self.embedding_matrix[x]  # (B, context_size, D)
        self.embeddings_flat = self.embeddings.reshape(x.shape[0], -1)  # (B, context_size*D)

        # Linear projection
        self.logits = self.embeddings_flat @ self.linear_W + self.linear_b  # (B, vocab_size)

        # Softmax
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if target:
            B = x.shape[0]
            loss = -np.log(self.probs[np.arange(B), y]).mean()
            return loss, self.logits
        else:
            return self.probs

    def backwards(self, x, y):
        B = x.shape[0]

        # -------- Softmax + Cross-Entropy Gradient --------
        dlogits = self.probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B  # (B, vocab_size)

        # -------- Gradients for linear layer --------
        dW = self.embeddings_flat.T @ dlogits  # (context_size*D, vocab_size)
        db = np.sum(dlogits, axis=0, keepdims=True)  # (1, vocab_size)

        # Gradient through embeddings_flat
        demb_flat = dlogits @ self.linear_W.T  # (B, context_size*D)
        demb = demb_flat.reshape(self.embeddings.shape)  # (B, context_size, D)

        # Update Embeddings
        np.add.at(self.embedding_matrix, x, -self.lr * demb)

        # Update weights
        self.linear_W -= self.lr * dW
        self.linear_b -= self.lr * db

    def fit(self, data, epochs=10, batch_size=32, lr_decay=1.0):
        """
        Epoch-based training with shuffling
        """
        data = np.array(data, dtype=np.int64)
        num_samples = len(data) - self.ngram_size + 1

        # Precompute all n-grams once
        contexts = np.stack([data[i:i+self.ngram_size-1] for i in range(num_samples)])
        targets = np.array([data[i+self.ngram_size-1] for i in range(num_samples)])

        for epoch in range(epochs):
            # Shuffle data each epoch
            perm = np.random.permutation(num_samples)
            contexts_shuffled = contexts[perm]
            targets_shuffled = targets[perm]

            epoch_loss = 0.0
            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min((b+1) * batch_size, num_samples)
                x_batch = contexts_shuffled[start:end]
                y_batch = targets_shuffled[start:end]

                # Forward and backward
                loss, _ = self.forward(x_batch, y_batch, target=True)
                self.backwards(x_batch, y_batch)
                epoch_loss += loss * (end - start)

            epoch_loss /= num_samples
            self.lr *= lr_decay  # Optional learning rate decay
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - lr: {self.lr:.6f}")

    def perplexity(self, data, batch_size=1) -> float:
        """
        Calculate the perplexity for that model after training on validation data

        :param self: the model itfels
        :param data: The encoded validation data
        :param batch_size: should be 1 perplexity calculation
        :return: the perplexity value
        """

        # Load Data and calculate num_samples based on data length and ngram size
        data = np.array(data, dtype=np.int64)
        num_samples = len(data) - self.ngram_size + 1

        contexts = np.stack([data[i:i + self.ngram_size - 1] for i in range(num_samples)])
        targets = np.array([data[i + self.ngram_size - 1] for i in range(num_samples)])

        nll = 0.0
        for i in range(num_samples):
            probs = self.forward(contexts[i:i+1], target=False)
            nll += -np.log(probs[0, targets[i]])

        avg_nll = nll / num_samples
        return float(np.exp(avg_nll))



