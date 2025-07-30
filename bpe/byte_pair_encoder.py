# Import helper functions for preprocessing and BPE operations
from utils.bpe.normalization import text_normalization
from utils.bpe.char_tokenization import character_tokenization
from utils.bpe.get_most_frequent_token_pair import get_most_frequent_pair
from utils.bpe.merge_token_pair_in_corpus import update_corpus

import json
import os

#TODO: Store dictionaries also in json file..
class BytePairEncoder:
    def __init__(self, vocab_size: int, verbose: bool, model_path: str = None, neural=False):
        """
        Initialize the Byte Pair Encoder.

        :param vocab_size: Number of desired merge operations (controls model size)
        :param verbose: If True, detailed output is printed during training
        :param model_path: Optional default path for saving/loading model
        """
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.model_path = model_path
        self.bpe_codes = []  # Stores learned BPE merge codes (tokens)
        self.stoi = {}
        self.itos = {}
        self.neural = neural

    def fit(self, corpus):
        """
        Train the BPE model on the given text corpus.
        """
        normalized_corpus = text_normalization(corpus)
        # print(normalized_corpus)
        tokenized_corpus = character_tokenization(normalized_corpus)
        self.bpe_codes = sorted(list(set("".join(tokenized_corpus))), key=len, reverse=True)

        for i in range(self.vocab_size):
            if not self.verbose:
                print(str(i + 1) + 'th iteration')

            most_frequent_pair = get_most_frequent_pair(tokenized_corpus)
            new_code = most_frequent_pair[0] + most_frequent_pair[1]

            if new_code not in self.bpe_codes:
                self.bpe_codes.append(new_code)
                self.bpe_codes.sort(key=len, reverse=True)

            tokenized_corpus = update_corpus(
                tokenized_corpus,
                most_frequent_pair[0],
                most_frequent_pair[1]
            )

        self.stoi = {c: i for i, c in enumerate(self.bpe_codes)}
        self.itos = {i: c for i, c in enumerate(self.bpe_codes)}


    def encode(self, corpus: str):
        """
        Encode new text based on the trained BPE codes.
        """
        corpus = text_normalization(corpus)
        corpus = character_tokenization(corpus)

        result = []
        i = 0
        while i < len(corpus):
            match = None
            for subword in self.bpe_codes:
                if len(subword) <= len(corpus) - i and ''.join(corpus[i:i + len(subword)]) == subword:
                    match = subword
                    break
            if match:
                result.append(match)
                i += len(match)
            else:
                result.append(corpus[i])
                i += 1

        if self.neural:
            return [self.stoi[char] for char in result]
        else:
            return result


    def save(self, filepath: str = None):
        """
        Save the trained BPE codes and configuration to JSON.
        Uses constructor-defined model_path if filepath is not given.
        """
        path = filepath or self.model_path
        if not path:
            raise ValueError("No file path specified for saving the model.")

        data = {
            "vocab_size": self.vocab_size,
            "bpe_codes": self.bpe_codes,
            "stoi": self.stoi,
            "itos": self.itos
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.verbose:
            print(f"BPE saved to {path}")

    def load(self, filepath: str = None):
        """
        Load BPE codes and configuration from JSON.
        Uses constructor-defined model_path if filepath is not given.
        """
        path = filepath or self.model_path
        if not path:
            raise ValueError("No file path specified for loading the model.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No BPE model found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.bpe_codes = data["bpe_codes"]
        self.stoi = data["stoi"]

        self.itos = {int(k): v for k, v in data["itos"].items()}

        if self.verbose:
            print(f"BPE loaded from {path}")