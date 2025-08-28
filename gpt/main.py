import sys

from gpt.utils.build_model_and_data import build_model_and_data
from gpt.utils.model_config import OwnConfig
import torch
from gpt.utils.train_model import train_model
import matplotlib.pyplot as plt
import os
from gpt.utils.perplexity_calculation import calc_perplexity


def ask_choice() -> str:
    """
    Ask the user whether to train/evaluate or to generate text.
    Returns: "train" or "gen".
    """
    print("What would you like to do?")
    print("[1] Train the model and then calculate perplexity")
    print("[2] Generate text")
    while True:
        choice = input("Please choose 1 or 2: ").strip()
        if choice == "1":
            return "train"
        if choice == "2":
            return "gen"
        print("Invalid input. Please enter 1 or 2.")


def train_and_evaluate():
    """
    Train the model, evaluate perplexity, and plot the training/validation losses.
    """
    encoder, model, device, train_tensor, valid_tensor, test_tensor = build_model_and_data()

    # Training
    train_losses, val_losses, result_paths, (val_loader, test_loader) = train_model(
        model, train_tensor, valid_tensor, test_tensor,
        getattr(model, "config_dict", None) or getattr(model, "config", None) or OwnConfig(),
        device
    )

    # Perplexity
    val_ppl  = calc_perplexity(model, val_loader,  device)
    test_ppl = calc_perplexity(model, test_loader, device)
    print(f"Validation Perplexity: {val_ppl:.3f}")
    print(f"Test Perplexity:        {test_ppl:.3f}")

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses,   label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def generate_text():
    """
    Load saved weights and generate text from a user prompt.
    """
    # Build model & encoder (without training)
    encoder, model, device, _train_tensor, _valid_tensor, _test_tensor = build_model_and_data()

    # Automatically load weights
    weights_path = os.path.join("weights", "best_weights.pt")
    if os.path.isfile(weights_path):
        try:
            model.load_weights(weights_path, map_location=device, strict=True)
            print(f"Weights loaded successfully from '{weights_path}'.")
        except Exception as e:
            print(f"Error while loading weights ({e}). Using untrained weights for generation.")
    else:
        print(f"Warning: '{weights_path}' not found. Using untrained weights for generation.")

    # Ask for prompt
    print("\nText generation:")
    user_prompt = input("Please enter your starting text/prompt: ").strip()
    if not user_prompt:
        print("Empty prompt entered. Aborting.")
        return

    # Hardcoded generation parameters
    max_new_tokens = 120      # <- adjust generation length here
    temperature    = 1.0
    top_k          = 50
    do_sample      = True

    # Tokenize
    prompt_ids = encoder.encode(user_prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    # Generate
    with torch.no_grad():
        out_ids = model.generate(
            idx=idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k
        )

    # Decode & print
    out_ids_list = out_ids[0].tolist()
    generated_text = encoder.decode(out_ids_list)

    print("\n" + "=" * 80)
    print("Generated Text:")
    print("-" * 80)
    print(generated_text)
    print("=" * 80 + "\n")


def main():
    """
    Interactive CLI:
    - Choose between training (+ perplexity) or text generation
    """
    try:
        mode = ask_choice()
        if mode == "train":
            train_and_evaluate()
        else:
            generate_text()
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()