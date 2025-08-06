from utils.load_data import load_data
from neural_ngram.neural_ngram import NeuralNGram
from classical_ngram.utils.preprocessing import prepare_data
from neural_ngram.pytorch_bigram import NeuralNGramTorch


def main():
    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Prepare data for neural n-gram
    vocab_size, train_data, valid_data, test_data = prepare_data(training_data, valid_data, test_data, vocab_size=1000, datatype='shakespeare', neural=True)

    '''
    model = NeuralNGram(embedding_dimension=64,
                         vocab_size=vocab_size,
                         ngram_size=3,
                         lr=0.5,
                         hidden_layer_size=128)
    model.fit(train_data, valid_data, patience=5, epochs=50, batch_size=32, lr_decay=0.95)
    print(model.perplexity(test_data))

    '''
    model = NeuralNGramTorch(vocab_size=vocab_size, ngram_size=3, embedding_dim=64, hidden_dim=128, lr=0.5)
    model.fit(train_data, valid_data, epochs=50, batch_size=32, patience=5, lr_decay=0.95)
    print("Perplexity:", model.perplexity(valid_data))  


if __name__ == '__main__':
    main()