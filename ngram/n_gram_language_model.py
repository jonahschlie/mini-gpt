from collections import Counter, defaultdict
import math
import random


class NGramModel:
    def __init__(self, context_size):
        self.context_size = context_size
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()
        self.unigram = Counter()
        self.vocab_size = 0

    def fit(self, training_data: list):
        i = self.context_size
        self.vocab_size = len(set(training_data))
        while i < len(training_data):
            self.unigram[training_data[i]] += 1
            context_key = tuple(training_data[i- self.context_size:i])
            self.ngrams[context_key][training_data[i]] += 1
            self.context_counts[context_key] += 1
            i = i + 1

        print(self.ngrams)

    def probability(self, context, word):
        return (self.ngrams[context][word] + 1) / (self.context_counts[context] + self.vocab_size)

    def calculate_perplexity(self, test_data):
        i = self.context_size
        probabilities = []
        while i < len(test_data):
            probabilities.append(math.log(self.probability(tuple(test_data[i-self.context_size:i]),test_data[i])))
            i += 1
        N = len(test_data) - self.context_size
        return math.exp(- (1 / N) * sum(probabilities))

    def predict_next_word(self, context):
        if len(context) != self.context_size:
            raise ValueError('The provided context doesn`t fit the context size of this model')

        if context in self.ngrams:
            probability_distribution = self.ngrams[context]
            words = list(probability_distribution.keys())
            counts = list(probability_distribution.values())
            word_index = counts.index(max(counts))
            return words[word_index]
        else:
            words = list(self.unigram.keys())
            weights = list(self.unigram.values())
            return random.choices(words, weights, k=1)[0]

    def generate_sequence(self, length=20, seed=None):
        # 1. If no seed is given, choose one randomly from existing contexts
        if seed is None:
            seed = random.choice(list(self.ngrams.keys()))
        elif len(seed) != self.context_size:
            raise ValueError(f"Seed must have length {self.context_size}")

        # Convert seed to list so we can append
        sequence = list(seed)
        # 2. Generate words until reaching desired length
        for _ in range(length):
            next_word = self.predict_next_word(tuple(sequence[-self.context_size:]))
            sequence.append(next_word)

        return sequence