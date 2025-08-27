from utils.load_data import load_data
from utils.preprocessing import prepare_data
from gpt.gpt_model import GPT
from gpt.utils.model_config import OwnConfig
import torch
from gpt.utils.train_model import train_model
import matplotlib.pyplot as plt
import os
from gpt.utils.perplexity_calculation import calc_perplexity

def ensure_dir(path: str):
    """Create folder if it doesn’t exist."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main():
    # TODO: ask the user to whether train the model or generate sequences..

    """
        Train a GPT model on the dataset, save weights,
        plot training/validation losses, and compute perplexity
        on validation and test splits.
    """

    # Load & prepare data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data,
        valid_data,
        test_data,
        vocab_size=1000,
        datatype='shakespeare',
        neural=True
    )

    # convert to tensors and flatten
    train_data = torch.as_tensor(train_data, dtype=torch.long).view(-1)
    valid_data = torch.as_tensor(valid_data, dtype=torch.long).view(-1)
    test_data  = torch.as_tensor(test_data,  dtype=torch.long).view(-1)

    # Model config & setup
    config = OwnConfig(vocab_size=vocab_size)

    # device: default to CPU unless specified in config
    device = torch.device(getattr(config, 'device', 'cpu'))
    print(f"Info - Using device: {device}")

    # build GPT model
    model = GPT(config).to(device)

    # use PyTorch 2.0 for "faster" learning hardware-performance-wise
    if getattr(config, 'use_torch_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # Training
    train_losses, val_losses, result_paths, (val_loader, test_loader) = train_model(
        model, train_data, valid_data, test_data, config, device
    )

    # Perplexity evaluation
    val_ppl  = calc_perplexity(model, val_loader,  device)
    test_ppl = calc_perplexity(model, test_loader, device)
    print(f"Validation Perplexity: {val_ppl:.3f}")
    print(f"Test Perplexity:        {test_ppl:.3f}")

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses,   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()