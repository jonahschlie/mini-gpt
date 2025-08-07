from utils.load_data import load_data
from classical_ngram.utils.preprocessing import prepare_data
from neural_ngram.neural_ngram import NeuralNGram
from neural_ngram.pytorch_neural_ngram import NeuralNGramTorch
import matplotlib.pyplot as plt

def main():
    # Ask user which model to run
    choice = input("Choose model to run ('1' = Numpy NeuralNGram, '2' = PyTorch NeuralNGramTorch): ").strip()

    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Prepare data for neural n-gram
    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data, valid_data, test_data, vocab_size=1000, datatype='shakespeare', neural=True
    )

    if choice == "1":
        print("Running Numpy NeuralNGram (custom implementation)...")
        model = NeuralNGram(
            embedding_dimension=64,
            vocab_size=vocab_size,
            ngram_size=3,
            lr=0.5,
            hidden_layer_size=128
        )
        train_loss, val_loss = model.fit(
            train_data, valid_data, patience=5, epochs=50, batch_size=32, lr_decay=0.95
        )
        perplexity = model.perplexity(test_data)
        sequence = model.generate_sequence(seed=encoder.encode("shall i "), length= 20, idx_to_token=encoder.itos, sample=False)
        print(encoder.decode(sequence))
        sequence = model.generate_sequence(seed=encoder.encode("shall i "), length= 20, idx_to_token=encoder.itos, sample=True)
        print(encoder.decode(sequence))

    elif choice == "2":
        print("Running PyTorch NeuralNGramTorch...")
        model = NeuralNGramTorch(
            vocab_size=vocab_size, ngram_size=3, embedding_dim=64, hidden_dim=128, lr=0.5
        )
        train_loss, val_loss = model.fit(
            train_data, valid_data, epochs=50, batch_size=32, patience=5, lr_decay=0.95
        )
        perplexity = model.perplexity(test_data)
        sequence = model.generate_sequence(seed=encoder.encode("shall i "), length=20, idx_to_token=encoder.itos, sample=False)
        print(encoder.decode(sequence))
        sequence = model.generate_sequence(seed=encoder.encode("shall i "), length=20, idx_to_token=encoder.itos, sample=True)
        print(encoder.decode(sequence))
    else:
        print("Invalid choice. Exiting.")
        return

    print(f"Perplexity: {perplexity:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)

    # Add title and labels
    plt.title('Neural N-Gram Training & Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Add legend and grid
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()