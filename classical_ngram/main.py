from utils.load_data import load_data
from classical_ngram.utils.preprocessing import prepare_data
from classical_ngram.n_gram_language_model import NGramModel
from datasets import load_dataset
import matplotlib.pyplot as plt
from copy import deepcopy
from classical_ngram.n_gram_language_model import NGramModel

import numpy as np
from copy import deepcopy
from classical_ngram.n_gram_language_model import NGramModel

def sequence_generation(context: list, train_tokens: list) -> list:
    """
    Generate a sequence from the trained N-gram model given an initial context.

    Args:
        context (list): A list of tokens (words/subwords/characters) to start generation from.
        train_tokens (list): A list of tokens (words/subwords/characters) to fit the neural_ngram.

    Returns:
        str: Generated sequence as a string.
    """

    # Use a trigram model for generation (you can adjust n)
    n = 3
    model = NGramModel(n=n, lambdas=[0.0, 0.488, 0.507])
    model.fit(train_tokens)

    # Generate a sequence of up to 20 tokens using argmax prediction
    generated_tokens = model.generate_sequence(seed=tuple(context), sample=True)

    return generated_tokens


def optimize_interpolation_weights(training_tokens, valid_tokens, n, step=0.1, max_iter=50, patience=5):
    """
    Greedy search with patience:
    - Only update lambdas if perplexity improves.
    - Stop if no improvement for `patience` iterations.
    """
    if n <= 1:
        raise ValueError("Interpolation weights are only relevant for n > 1")

    # Initialize equal weights
    lambdas = np.ones(n) / n
    model = NGramModel(n=n, lambdas=lambdas.tolist())
    model.fit(training_tokens)
    best_perplexity = model.calculate_perplexity(valid_tokens)
    best_lambdas = lambdas.copy()

    no_improve_count = 0
    iteration = 0

    while iteration < max_iter and no_improve_count < patience:
        iteration += 1
        improved = False

        for i in range(n):
            for delta in [+step, -step]:
                new_lambdas = lambdas.copy()
                new_lambdas[i] += delta
                if np.any(new_lambdas < 0):
                    continue
                new_lambdas = new_lambdas / new_lambdas.sum()

                model = NGramModel(n=n, lambdas=new_lambdas.tolist())
                model.fit(training_tokens)
                perplexity = model.calculate_perplexity(valid_tokens)

                # Only accept move if perplexity improves
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_lambdas = new_lambdas.copy()
                    lambdas = new_lambdas.copy()
                    improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        #print(f"Iteration {iteration}: lambdas={np.round(lambdas, 3)}, "
              #f"best_perplexity={best_perplexity:.4f}, "
              #f"no_improve_count={no_improve_count}")

    return best_lambdas.tolist(), best_perplexity

def perplexity_comparison(training_data, valid_data, test_data):
    perplexity_dict = {}
    ngram_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for size in ngram_size:
        step = 200

        for k in range(0,2200, step):
            # Train BPE with current k
            _, training_tokens, _, test_tokens = prepare_data(training_data, valid_data, test_data, vocab_size=k, neural=False)

            # Train N-gram on BPE tokens
            ngram = NGramModel(n=size)
            ngram.fit(training_tokens)
            perplexity = ngram.calculate_perplexity(test_tokens)

            if size not in perplexity_dict:
                perplexity_dict[size] = []
            perplexity_dict[size].append(perplexity)

    return perplexity_dict


def main():
    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Fixed BPE vocab_size
    vocab_size = 1000
    bpe_encoder, _, train_tokens, valid_tokens, test_tokens = prepare_data(
        training_data, valid_data, test_data, vocab_size=vocab_size, neural=False
    )

    print("Choose an experiment to run:")
    print("1: Sequence Generation")
    print("2: Optimize Interpolation Weights")
    print("3: Perplexity Comparison")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        initial_context = ["shall_", "i_"]
        generated_sequence = bpe_encoder.decode(
            sequence_generation(initial_context, train_tokens)
        )
        print("Initial context:", " ".join(initial_context))
        print("Generated sequence:", generated_sequence)

    elif choice == "2":
        sizes = np.arange(2, 11)
        for size in sizes:
            lambdas, best_perplexity = optimize_interpolation_weights(
                train_tokens, valid_tokens, n=size, step=0.1, patience=5
            )
            print(f"Optimal {size}-gram lambdas: {lambdas}, best perplexity: {best_perplexity:.4f}")

    elif choice == "3":
        perplexity = perplexity_comparison(training_data, valid_data, test_data)
        print("Perplexity comparison results:", perplexity)

        for size, values in perplexity.items():
            ks = list(range(0, 2200, 200))
            plt.plot(ks[:len(values)], values, label=f'N={size}')

        plt.xlabel('BPE merges (k)')
        plt.ylabel('Perplexity')
        plt.title('Perplexity vs BPE merges for different N-gram sizes')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("Invalid choice. Please run again and enter 1, 2, or 3.")



if __name__ == '__main__':
    main()