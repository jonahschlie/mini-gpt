from utils.load_data import load_data
from bpe.byte_pair_encoder import BytePairEncoder
from datasets import load_dataset
import os


def k_optimization_generalisation(training_data: str, test_data: str, step_size: int, max_no_improve=3, model_path= "models/bpe/shakespeare" ):
    k = 0
    step = step_size
    best_k = k
    best_score = float("inf")
    no_improve_count = 0
    max_no_improve = max_no_improve

    while True:
        model_file = f"{model_path}/model_{k}k.json"

        bpe_encoder = BytePairEncoder(vocab_size=k, verbose=False, model_path=f"{model_path}/model_{k}k.json")

        if os.path.exists(model_file):
            print(f"loaded model {model_file}")
            bpe_encoder.load()
        else:
            print(f"creating model {model_file}")
            bpe_encoder.fit(training_data)
            bpe_encoder.save()

        _, train_tpw = bpe_encoder.calculate_metrics(training_data, verbose=False)
        _, test_tpw = bpe_encoder.calculate_metrics(test_data, verbose=False)

        # Generalization ratio: closer to 1 is better
        generalization_ratio = test_tpw / train_tpw
        # Combine TPW and generalization penalty into one score
        score = test_tpw * generalization_ratio

        print(f"k={k} | train_tpw={train_tpw:.4f}, test_tpw={test_tpw:.4f}, "
              f"generalization={generalization_ratio:.4f}, score={score:.4f}")

        # Check if we improved
        if score < best_score:
            best_score = score
            best_k = k
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping after no improvement for several steps
        if no_improve_count >= max_no_improve:
            print(f"Stopping search. Best k={best_k} with score={best_score:.4f}")
            break

        k += step

    return best_k




def main():
    # Load the shakespeare data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    #Load the modern english wikitext data
    wiki_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    modern_test_data = ' '.join(wiki_dataset['test']['text'])

    # Don't forget to give in the data you want to test for
    best_k = k_optimization_generalisation(training_data, modern_test_data, step_size=100, max_no_improve=3)
    best_k = k_optimization_generalisation(training_data, test_data, step_size=2000, max_no_improve=3)
    print(f"Optimal k found: {best_k}")


if __name__ == '__main__':
    main()