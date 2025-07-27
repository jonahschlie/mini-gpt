from collections import Counter, defaultdict
import math


class NGramModel:
    def __init__(self, context_size):
        self.context_size = context_size
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab_size = 0

    def fit(self, training_data: list):
        i = self.context_size
        self.vocab_size = len(set(training_data))
        while i < len(training_data):
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

    def predict_next_word(self):
        pass

    def generate_sequence(self):
        pass