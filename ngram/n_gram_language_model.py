from collections import Counter, defaultdict
import math
import random


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

    def __init__(self, n=3, lambdas=None):
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

    def probability(self, context, word):
        """
        Compute the interpolated probability of a word given its context.

        Args:
            context (tuple): Previous words.
            word (str): The target word to score.

        Returns:
            float: Interpolated probability.
        """
        probs = []
        for size in range(1, self.n + 1):
            context_slice = tuple(context[-(size - 1):]) if size > 1 else tuple()
            if size == 1:  # unigram
                prob = (self.unigram[word] + 1) / (sum(self.unigram.values()) + self.vocab_size)
            else:
                prob = (self.ngrams[size][context_slice][word] + 1) / (
                            self.context_counts[size][context_slice] + self.vocab_size)
            probs.append(self.lambdas[size - 1] * prob)
        return sum(probs)

    def calculate_perplexity(self, test_data):
        """
        Calculate the perplexity of the model on test data.

        Args:
            test_data (list): Sequence of tokens for evaluation.

        Returns:
            float: Perplexity value (lower is better).
        """
        total_log_prob = 0
        n = len(test_data) - self.n + 1
        for i in range(self.n - 1, len(test_data)):
            context = tuple(test_data[i - self.n + 1:i])
            word = test_data[i]
            total_log_prob += math.log(self.probability(context, word))
        return math.exp(-total_log_prob / n)

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

    def load(self):
        pass

    def save(self):
        pass