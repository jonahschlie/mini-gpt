import numpy as np
import random

class NeuralNGram:

    def __init__(self, embedding_dimension, vocab_size, ngram_size, lr=1e-2, hidden_layer_size = 128):
        self.embedding_dimension = embedding_dimension
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.lr = lr
        self.hidden_layer_size = hidden_layer_size

        # Embedding: vocab_size → embedding_dimension
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dimension) * 0.01

        #Hidden Layer: (context_size * embedding_dim) → hidden layer size
        input_dim = (self.ngram_size - 1) * self.embedding_dimension
        self.linear_W1 = np.random.randn(input_dim, self.hidden_layer_size) * 0.01
        self.linear_b1 = np.zeros((1, self.hidden_layer_size))

        # Linear layer: hidden layer size → vocab_size
        self.linear_W2 = np.random.randn(self.hidden_layer_size, self.vocab_size) * 0.01
        self.linear_b2 = np.zeros((1, self.vocab_size))

    def forward(self, x, y=None, target=True):
        # Embedding lookup
        self.embeddings = self.embedding_matrix[x]  # (B, context_size, D)
        self.embeddings_flat = self.embeddings.reshape(x.shape[0], -1)  # (B, context_size*D)

        # Hidden layer with tanh activation
        self.hidden_layer = self.embeddings_flat @ self.linear_W1 + self.linear_b1
        self.hidden_activation = np.tanh(self.hidden_layer)

        # Linear projection to vocab size
        self.logits = self.hidden_activation @ self.linear_W2 + self.linear_b2  # (B, vocab_size)

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

        # -------- Gradients for output layer (W2, b2) --------
        dW2 = self.hidden_activation.T @ dlogits  # (H, V)
        db2 = np.sum(dlogits, axis=0, keepdims=True)  # (1, V)

        # -------- Backprop into hidden activation --------
        dha = dlogits @ self.linear_W2.T  # (B, H)

        # -------- Backprop through tanh --------
        dh = dha * (1 - self.hidden_activation ** 2)  # (B, H)

        # -------- Gradients for hidden layer (W1, b1) --------
        dW1 = self.embeddings_flat.T @ dh  # (C*D, H)
        db1 = np.sum(dh, axis=0, keepdims=True)  # (1, H)

        # -------- Backprop into embeddings --------
        demb_flat = dh @ self.linear_W1.T  # (B, C*D)
        demb = demb_flat.reshape(self.embeddings.shape)  # (B, C, D)

        # -------- Parameter updates --------
        # Embeddings
        np.add.at(self.embedding_matrix, x, -self.lr * demb)
        # Hidden layer
        self.linear_W1 -= self.lr * dW1
        self.linear_b1 -= self.lr * db1
        # Output layer
        self.linear_W2 -= self.lr * dW2
        self.linear_b2 -= self.lr * db2

    def fit(self, train_data, valid_data, patience=5, epochs=10, batch_size=32, lr_decay=1.0) -> tuple:
        """
        Epoch-based training with shuffling
        """

        #Initialize best model configuration for restoring after patience time
        best_embeddings = self.embedding_matrix.copy()
        best_hidden_weights = self.linear_W1.copy()
        best_hidden_bias = self.linear_b1.copy()
        best_output_weights = self.linear_W2.copy()
        best_output_bias = self.linear_b2.copy()

        best_val_loss = float('inf')

        wait = 0
        stop_training = False

        # Training Data Preparation
        train_data = np.array(train_data, dtype=np.int64)
        train_num_samples = len(train_data) - self.ngram_size + 1

        val_data = np.array(valid_data, dtype=np.int64)
        val_num_samples = len(val_data) - self.ngram_size + 1

        # Precompute all n-grams once
        contexts = np.stack([train_data[i:i+self.ngram_size-1] for i in range(train_num_samples)])
        targets = np.array([train_data[i+self.ngram_size-1] for i in range(train_num_samples)])

        val_contexts = np.stack([val_data[i:i + self.ngram_size - 1] for i in range(val_num_samples)])
        val_targets = np.array([val_data[i + self.ngram_size - 1] for i in range(val_num_samples)])

        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            # Shuffle data each epoch
            perm = np.random.permutation(train_num_samples)
            contexts_shuffled = contexts[perm]
            targets_shuffled = targets[perm]

            epoch_loss = 0.0
            num_batches = int(np.ceil(train_num_samples / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min((b+1) * batch_size, train_num_samples)
                x_batch = contexts_shuffled[start:end]
                y_batch = targets_shuffled[start:end]

                # Forward and backward
                loss, _ = self.forward(x_batch, y_batch, target=True)
                self.backwards(x_batch, y_batch)
                epoch_loss += loss * (end - start)


            epoch_loss /= train_num_samples
            train_losses.append(epoch_loss)

            num_val_batches = int(np.ceil(val_num_samples / batch_size))
            val_loss = 0.0
            for i in range(num_val_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, val_num_samples)
                loss, _ = self.forward(val_contexts[start:end], val_targets[start:end], target=True)
                val_loss += loss * (end - start)
            val_loss /= val_num_samples
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_embeddings = self.embedding_matrix.copy()
                best_hidden_weights = self.linear_W1.copy()
                best_hidden_bias = self.linear_b1.copy()
                best_output_weights = self.linear_W2.copy()
                best_output_bias = self.linear_b2.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping!")
                    self.embedding_matrix = best_embeddings
                    self.linear_W1 = best_hidden_weights
                    self.linear_b1 = best_hidden_bias
                    self.linear_W2 = best_output_weights
                    self.linear_b2 = best_output_bias
                    stop_training = True
                    break


            if not stop_training:
                self.lr *= lr_decay  # Optional learning rate decay
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - lr: {self.lr:.6f}")

            if stop_training:
                break

        return train_losses, val_losses

    def perplexity(self, data, batch_size=1) -> float:
        """
        Calculate the perplexity for that model after training on validation data

        :param self: the model itfels
        :param data: The encoded validation data
        :param batch_size: should be 1 perplexity calculation
        :return: the perplexity value
        """

        # Load Data and calculate num_samples based on data length and neural_ngram size
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

    def generate_sequence(self, seed, idx_to_token, length=20000, sample=False):
        """
        Generate a sequence of tokens using the trained neural model.

        Args:
            seed (list[int]): A list of token indices of length n-1 as the initial context.
            length (int): Max sequence length to generate (default 20000).
            sample (bool): If True, sample next token based on probability distribution;
                           otherwise, choose the most probable token (argmax).
            idx_to_token (callable or dict, optional): A mapping from token index to string
                           for punctuation checking. If None, raw indices are used.

        Returns:
            list[int]: Generated token sequence including the initial seed.
        """
        if len(seed) != self.ngram_size - 1:
            raise ValueError(f"Seed must have length {self.ngram_size - 1}")

        context = seed.copy()
        output = seed.copy()

        for _ in range(length):
            # Get probability distribution for next token
            probs = self.forward(np.array([context]), target=False)[0]

            # Choose next token
            if sample:
                next_token = random.choices(range(self.vocab_size), weights=probs, k=1)[0]
            else:
                next_token = int(np.argmax(probs))

            output.append(next_token)

            # Slide context window
            context = output[-(self.ngram_size - 1):]

            # --- Break Conditions ---
            # Check punctuation token if mapping provided
            if idx_to_token:
                next_word = idx_to_token[next_token] if callable(idx_to_token) else idx_to_token.get(next_token, "")
                if any(punct in next_word for punct in [".", "?", "!"]):
                    break
            # Stop if max length reached
            if len(output) > length:
                print(len(output))
                break

        return output



