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
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def main():

    # Load the data
    test_string, full_data, training_data, test_data, valid_data = load_data()

    # Prepare data for gpt
    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data, valid_data, test_data, vocab_size=1000, datatype='shakespeare', neural=True
    )

    train_data = torch.as_tensor(train_data, dtype=torch.long).view(-1)
    valid_data = torch.as_tensor(valid_data, dtype=torch.long).view(-1)
    test_data  = torch.as_tensor(test_data,  dtype=torch.long).view(-1)

    config = OwnConfig(vocab_size=vocab_size)

    # Speicher-Ordner (aus Config oder Default)
    save_dir = getattr(config, "save_dir", "weights")
    ensure_dir(save_dir)
    weights_path = os.path.join(save_dir, "final_weights.pt")

    # Device wählen
    device = torch.device(getattr(config, 'device', 'cpu'))
    print(f"[Info] Using device: {device}")

    # Modell bauen (+ optional torch.compile)
    model = GPT(config).to(device)
    if getattr(config, 'use_torch_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)  # nutzt config.use_torch_compile

    # === TRAINING ===
    # train_model gibt (train_losses, val_losses, result_paths, (val_loader, test_loader)) zurück
    train_losses, val_losses, result_paths, (val_loader, test_loader) = train_model(
        model, train_data, valid_data, test_data, config, device
    )

    # (optional) finale Gewichte speichern
    try:
        model.save_weights(weights_path, extra={"note": "final"})
        print(f"[Save] final weights -> {weights_path}")
    except Exception as e:
        print(f"[Warn] konnte finale Gewichte nicht speichern: {e}")

    # === PERPLEXITY-BERECHNUNG ===
    val_ppl  = calc_perplexity(model, val_loader,  device)
    test_ppl = calc_perplexity(model, test_loader, device)
    print(f"[PPL] Validation Perplexity: {val_ppl:.3f}")
    print(f"[PPL] Test Perplexity:        {test_ppl:.3f}")

    # === PLOTS ===
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