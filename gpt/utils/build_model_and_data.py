import torch

from gpt.gpt_model import GPT
from gpt.utils.model_config import OwnConfig
from utils.preprocessing import prepare_data
from utils.load_data import load_data

def build_model_and_data(vocab_size_override: int = 1000, datatype: str = "shakespeare", neural: bool = True, block_size=64):
    """
    Loads data, prepares it, builds the model according to OwnConfig.
    """
    # Load raw data
    _test_string, _full_data, training_data, test_data, valid_data = load_data()

    # Tokenizer/encoder + numerical data
    encoder, vocab_size, train_data, valid_data, test_data = prepare_data(
        training_data,
        valid_data,
        test_data,
        vocab_size=vocab_size_override,
        datatype=datatype,
        neural=neural
    )

    # Flatten tensors
    train_tensor = torch.as_tensor(train_data, dtype=torch.long).view(-1)
    valid_tensor = torch.as_tensor(valid_data, dtype=torch.long).view(-1)
    test_tensor  = torch.as_tensor(test_data,  dtype=torch.long).view(-1)

    # Config & device
    config = OwnConfig(vocab_size=vocab_size, block_size=block_size)
    device = torch.device(getattr(config, 'device', 'cpu'))
    print(f"Info - Using device: {device}")

    # Model
    model = GPT(config).to(device)

    return encoder, model, device, train_tensor, valid_tensor, test_tensor