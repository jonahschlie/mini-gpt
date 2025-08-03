import json
from collections import Counter, defaultdict
import math
import random
import os


class NGramModel:
    """
    An N-gram language model supporting interpolation across multiple n-gram orders.

    Attributes:
        n (int): The maximum order of n-grams (e.g., n=3 for trigram model).
        lambdas (list): Interpolation weights for each n-gram order.
        ngrams (dict): A dictionary of counters for storing n-gram counts by order.
        context_counts (dict): Counts of context occurrences for n-grams of order >= 2.
        unigram (Counter): Counts of individual words (order 1).
        vocab_size (int): Number of unique words in the training data.
    """

    def __init__(self, n=3, model_path: str = None, lambdas=None):
        """
        Initialize the n-gram model.

        Args:
            n (int): Maximum n-gram size to use (default=3).
            lambdas (list, optional): Interpolation weights for each n-gram order.
                                      If None, equal weights are used.
        """
        self.unigram = None
        self.n = n
        self.ngrams = {k: defaultdict(Counter) for k in range(1, n + 1)}
        self.context_counts = {k: Counter() for k in range(2, n + 1)}
        self.vocab_size = 0
        self.lambdas = lambdas if lambdas else [1 / n] * n
        self.model_path = model_path

    def fit(self, training_data):
        """
        Fit the n-gram model to a sequence of training tokens.

        Args:
            training_data (list): List of tokens (words or subwords).
        """
        self.vocab_size = len(set(training_data))
        for i in range(len(training_data)):
            for size in range(1, self.n + 1):
                if i - size + 1 >= 0:
                    context = tuple(training_data[i - size + 1: i])
                    word = training_data[i]
                    self.ngrams[size][context][word] += 1
                    if size > 1:
                        self.context_counts[size][context] += 1
        self.unigram = Counter(training_data)

    def probability(self, context, word, alpha=0.4):
        """
        Compute the interpolated probability of a word given its context.

        Uses stupid backoff for unseen higher-order n-grams:
            - If an n-gram exists: use its relative frequency.
            - If not: back off to lower order multiplied by alpha.

        Args:
            context (tuple): Previous words.
            word (str): Word to predict.
            alpha (float): Backoff weight (default=0.4).

        Returns:
            float: Interpolated probability.
        """
        # Identify active n-gram orders based on non-zero lambdas
        active_orders = [i + 1 for i, w in enumerate(self.lambdas) if w > 0]
        if not active_orders:
            return 0.0

        prob = 0.0
        for order in sorted(active_orders, reverse=True):
            # get the weight for this order
            lambda_weight = self.lambdas[order - 1]
            if lambda_weight == 0: # if no lambda skip this one
                continue

            if order == 1:
                prob_order = (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size) # laplace smoothing on unigram
            else:
                context_slice = tuple(context[-(order - 1):]) # get the context we are focusing at (trigram, bigram etc.)
                count_context = self.context_counts[order].get(context_slice, 0)
                count_word = self.ngrams[order][context_slice].get(word, 0)
                if count_word > 0:
                    prob_order = count_word / count_context
                else:
                    # Back off to unigram probability with laplace smoothing
                    prob_order = alpha * (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
            prob += lambda_weight * prob_order

        # Normalize by sum of lambdas to avoid scaling issues
        return prob / sum(self.lambdas)

    def calculate_perplexity(self, test_data):
        """
        Calculate perplexity of the model on test data.

        Dynamically adjusts context length based on active lambda weights.
        """
        # Determine maximum order used based on non-zero lambdas
        active_orders = [i + 1 for i, w in enumerate(self.lambdas) if w > 0]
        max_order = max(active_orders) if active_orders else 1

        total_log_prob = 0.0
        count = 0

        for i in range(max_order - 1, len(test_data)):
            context = tuple(test_data[i - max_order + 1:i])
            word = test_data[i]
            p = self.probability(context, word)
            total_log_prob += math.log(p)
            count += 1

        return math.exp(-total_log_prob / count) if count > 0 else float('inf')

    def predict_next_word(self, context):
        """
        Predict the most likely next word for a given context.

        Args:
            context (list or tuple): Previous words (up to n-1).

        Returns:
            str: Predicted next word.
        """
        context = tuple(context[-(self.n - 1):])
        candidates = self.ngrams[self.n].get(context, {})
        if not candidates:
            return random.choices(list(self.unigram.keys()), weights=self.unigram.values())[0]
        return max(candidates, key=candidates.get)

    def generate_sequence(self, length=20, seed=None):
        """
        Generate a sequence of words using the model.

        Args:
            length (int): Number of words to generate.
            seed (tuple, optional): Initial context of length n-1.
                                    If None, one is randomly selected.

        Returns:
            list: Generated word sequence including the seed.
        """
        if seed is None:
            seed = random.choice(list(self.ngrams[self.n].keys()))
        elif len(seed) != self.n - 1:
            raise ValueError(f"Seed must have length {self.n - 1}")
        sequence = list(seed)
        for _ in range(length):
            next_word = self.predict_next_word(sequence[-(self.n - 1):])
            sequence.append(next_word)
        return sequence

    def save(self, filepath: str = None):
        """
        Save the trained n-gram model to disk in a JSON-safe format.

        Args:
            filepath (str, optional): Path to save the model file.
                                      If None, uses self.model_path.
        """
        path = filepath or self.model_path
        if not path:
            raise ValueError("No file path specified for saving the model.")

        # Convert everything to JSON-safe format
        data = {
            'ngrams': {
                size: [
                    {"context": list(context), "words": dict(words)}
                    for context, words in contexts.items()
                ]
                for size, contexts in self.ngrams.items()
            },
            'context_counts': {
                size: [
                    {"context": list(context), "count": count}
                    for context, count in counts.items()
                ]
                for size, counts in self.context_counts.items()
            },
            'n': self.n,
            'vocab_size': self.vocab_size,
            'unigram': dict(self.unigram)
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Model saved to {path}")

    def load(self, filepath: str = None):
        """
        Load a previously saved n-gram model from disk.

        Args:
            filepath (str, optional): Path to load the model from.
                                      If None, uses self.model_path.
        """
        path = filepath or self.model_path
        if not path:
            raise ValueError("No file path specified for loading the model.")
        if not os.path.exists(path):
            raise FileNotFoundError(f'No ngram model found at {path}.')

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Restore attributes
        self.n = data['n']
        self.vocab_size = data['vocab_size']
        self.unigram = Counter(data['unigram'])

        # Restore n-grams
        self.ngrams = {int(size): defaultdict(Counter) for size in range(1, self.n + 1)}
        for size, entries in data['ngrams'].items():
            for entry in entries:
                context = tuple(entry["context"])
                self.ngrams[int(size)][context] = Counter(entry["words"])

        # Restore context counts
        self.context_counts = {int(size): Counter() for size in range(2, self.n + 1)}
        for size, entries in data['context_counts'].items():
            for entry in entries:
                context = tuple(entry["context"])
                self.context_counts[int(size)][context] = entry["count"]

        print(f"Model successfully loaded from {path}")


