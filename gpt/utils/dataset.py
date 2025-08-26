from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size: int = 128):
        self.block_size = int(block_size)
        # store as a 1D torch tensor on CPU
        self.data = torch.as_tensor(data, dtype=torch.long).view(-1)

    def __len__(self):
        return max(0, self.data.size(0) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y