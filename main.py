from utils.load_data import load_data
from bpe.byte_pair_encoder import BytePairEncoder
from ngram.n_gram_language_model import NGramModel
import re

def main():
    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Initialize BPE with default model file path
    bpe_encoder = BytePairEncoder(vocab_size=1000, verbose=False, model_path="models/bpe/bpe_model.json")

    # Train or load existing BPE
    try:
        bpe_encoder.load()  # will use model_path
        print("Loaded existing BPE model.")
    except FileNotFoundError:
        print("No saved BPE found. Training...")
        bpe_encoder.fit(full_data)
        bpe_encoder.save()

    # Encode training data
    encoded_training_data = bpe_encoder.encode(training_data)
    encoded_test_data = bpe_encoder.encode(test_data)
    encoded_valid_data = bpe_encoder.encode(valid_data)


    model = NGramModel(n=2, lambdas=[1, 0], model_path="models/ngram/bigram.json")
    model = NGramModel(n=2, lambdas=[0, 1], model_path="models/ngram/bigram.json")
    model = NGramModel(n=3, lambdas=[0,0.4,0.6], model_path="models/ngram/trigram.json")
    model = NGramModel(n=4, lambdas=[0.1,0.1,0.3,0.5], model_path="models/ngram/fourgram.json")
    #model = NGramModel(n=1, model_path="models/ngram/unigram.json")

    try:
        model.load()
        print('Loaded existing ngram model')
    except FileNotFoundError:
        model.fit(encoded_training_data)
        model.save()


    print(model.calculate_perplexity(encoded_test_data))
    # generated = model.generate_sequence(length=30, seed=("be_",))
    # print(re.sub('_', ' ', "".join(generated)))


if __name__ == '__main__':
    main()