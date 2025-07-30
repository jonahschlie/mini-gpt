import numpy as np
from bpe.byte_pair_encoder import BytePairEncoder
import numpy as np


def get_ngram_batch(data, n=3, batch_size=32):
    """
    Get training batch for an n-gram model.

    :param data: tokenized sequence (list or np.array)
    :param n: size of n-gram (e.g., 3 means 2 tokens context -> 1 token target)
    :param batch_size: how many samples per batch
    :return: (x, y) where
             x.shape = (batch_size, n-1)
             y.shape = (batch_size,)
    """
    data = np.array(data, dtype=np.int64)
    ix = np.random.randint(0, len(data) - n + 1, size=batch_size)

    # Collect n-1 tokens for context and one token for target
    x = np.stack([data[i:i + n - 1] for i in ix])
    y = np.array([data[i + n - 1] for i in ix])
    return x, y


def prepare_data(training_data, valid_data, test_data):
    bpe_encoder = BytePairEncoder(1000, verbose=False, model_path="models/bpe/model.json", neural=True)

    # Train or load existing BPE
    try:
        bpe_encoder.load()  # will use model_path
        print("Loaded existing BPE model.")
    except FileNotFoundError:
        print("No saved BPE found. Training...")
        bpe_encoder.fit(training_data)
        bpe_encoder.save()

    return len(bpe_encoder.bpe_codes), bpe_encoder.encode(training_data), bpe_encoder.encode(valid_data), bpe_encoder.encode(test_data)