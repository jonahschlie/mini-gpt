from bpe.byte_pair_encoder import BytePairEncoder

def prepare_data(training_data, valid_data, test_data, vocab_size=1000, datatype='shakespeare', neural= False):
    bpe_encoder = BytePairEncoder(vocab_size, verbose=False, model_path=f"models/bpe/{datatype}/model_{vocab_size}k.json", neural=neural)

    # Train or load existing BPE
    try:
        bpe_encoder.load()  # will use model_path
        print("Loaded existing BPE model.")
    except FileNotFoundError:
        print("No saved BPE found. Training...")
        bpe_encoder.fit(training_data)
        bpe_encoder.save()

    # Encode datasets
    train_tokens = bpe_encoder.encode(training_data)
    valid_tokens = bpe_encoder.encode(valid_data)
    test_tokens = bpe_encoder.encode(test_data)

    return len(bpe_encoder.bpe_codes), train_tokens, valid_tokens, test_tokens