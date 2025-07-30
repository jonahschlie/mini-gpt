from utils.load_data import load_data
from bpe.byte_pair_encoder import BytePairEncoder
from ngram.n_gram_language_model import NGramModel
from ngram.neural_ngram import NeuralBigram
import re
from utils.ngram.preprocessing import prepare_data

def main():
    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Prepare data for neural n-gram
    vocab_size, train_data, valid_data, test_data = prepare_data(training_data, valid_data, test_data)

    model = NeuralBigram(embedding_dimension=512, vocab_size=vocab_size, ngram_size=3, lr=0.5)
    model.fit(train_data, epochs=40, batch_size=32, lr_decay=0.95)
    print(model.perplexity())

if __name__ == '__main__':
    main()