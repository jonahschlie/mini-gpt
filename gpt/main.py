from utils.load_data import load_data
from utils.preprocessing import prepare_data
from gpt.gpt_model import GPT
from gpt.utils.model_config import OwnConfig
import torch
from gpt.utils.train_model import train_model
import matplotlib.pyplot as plt


def main():

    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Prepare data for gpt
    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data, valid_data, test_data, vocab_size=1000, datatype='shakespeare', neural=True
    )

    train_data = torch.as_tensor(train_data, dtype=torch.long).view(-1)
    valid_data = torch.as_tensor(valid_data, dtype=torch.long).view(-1)
    test_data = torch.as_tensor(test_data, dtype=torch.long).view(-1)

    config = OwnConfig(vocab_size=vocab_size)

    device = torch.device(getattr(config, 'device', 'auto'))
    print(f"[Info] Using device: {device}")

    model = GPT(config).to(device)
    if getattr(config, 'compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    train_losses, val_losses = train_model(model, train_data, valid_data, test_data, config, device)

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()