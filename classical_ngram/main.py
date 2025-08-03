from utils.load_data import load_data
from classical_ngram.utils.preprocessing import prepare_data
from bpe.byte_pair_encoder import BytePairEncoder
from classical_ngram.n_gram_language_model import NGramModel
from datasets import load_dataset

def k_optimization(training_data, valid_data, test_data):
    best_dicts = {}
    ngram_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for size in ngram_size:
        k =0
        best_k = k
        step = 200
        best_perplexity = float('inf')
        no_improve_count = 0
        max_no_improve = 3

        while True:
            # Train BPE with current k
            _, training_tokens, _, test_tokens = prepare_data(training_data, valid_data, test_data, vocab_size=k, datatype='wikitext', neural=False)

            # Train N-gram on BPE tokens
            ngram = NGramModel(n=size)
            ngram.fit(training_tokens)
            perplexity = ngram.calculate_perplexity(test_tokens)

            # Early stopping
            if perplexity < best_perplexity:
                # print(f"Perplexity {perplexity} for ngram size {size}")
                best_perplexity = perplexity
                best_k = k
                no_improve_count = 0
            else:
                print(f"Perplexity {perplexity} for ngram size {size} with k={k}")
                no_improve_count += 1

            if no_improve_count >= max_no_improve:
                print(f"N={size}: stopping at best k={best_k}, perplexity={best_perplexity:.4f}")
                best_dicts[size] = best_k
                break

            k += step

    return best_dicts


def main():
    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()


    # Load the modern english wikitext data
    wiki_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    modern_training_data = ' '.join(wiki_dataset['train']['text'])[:len(training_data)*5]
    modern_valid_data = ' '.join(wiki_dataset['validation']['text'])[:len(valid_data)*5]
    modern_test_data = ' '.join(wiki_dataset['test']['text'])[:len(test_data)*5]

    best_k = k_optimization(modern_training_data, modern_valid_data, modern_test_data)
    print(best_k)


if __name__ == '__main__':
    main()