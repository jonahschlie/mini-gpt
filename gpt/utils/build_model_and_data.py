import torch

from gpt.gpt_model import GPT
from gpt.utils.model_config import OwnConfig
from utils.preprocessing import prepare_data
from utils.load_data import load_data

def build_model_and_data(vocab_size_override: int = 1000, datatype: str = "shakespeare", neural: bool = True):
    """
    Lädt Daten, bereitet sie vor, baut das Modell gemäß OwnConfig.
    """
    # Rohdaten laden
    _test_string, _full_data, training_data, test_data, valid_data = load_data()

    # Tokenizer/Encoder + numerische Daten
    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data,
        valid_data,
        test_data,
        vocab_size=vocab_size_override,
        datatype=datatype,
        neural=neural
    )

    # Tensors flach machen
    train_tensor = torch.as_tensor(train_data, dtype=torch.long).view(-1)
    valid_tensor = torch.as_tensor(valid_data, dtype=torch.long).view(-1)
    test_tensor  = torch.as_tensor(test_data,  dtype=torch.long).view(-1)

    # Config & Gerät
    config = OwnConfig(vocab_size=vocab_size)
    device = torch.device(getattr(config, 'device', 'cpu'))
    print(f"Info - Verwende Gerät: {device}")

    # Modell
    model = GPT(config).to(device)

    # Optional: PyTorch 2.0 compile
    if getattr(config, 'use_torch_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)

    return encoder, model, device, train_tensor, valid_tensor, test_tensor
