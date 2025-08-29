from torch.utils.data import Dataset
import torch

class ShakespeareDataset(Dataset):
    """
    A simple subword-level dataset for language modeling.
    Produces input (x) and target (y) sequences of fixed length `block_size`.
    """

    def __init__(self, data, block_size: int = 128):
        """
        Args:
            data (array-like): Sequence of token IDs (ints).
            block_size (int): Length of each training sequence (context window).
        """
        self.block_size = int(block_size)
        # Store tokens as a flat 1D tensor on CPU for slicing.
        self.data = torch.as_tensor(data, dtype=torch.long).view(-1)

    def __len__(self):
        """
        Number of samples = total tokens - block_size.
        Each sample uses block_size tokens for input, and needs +1 token for target shift.
        """
        return max(0, self.data.size(0) - self.block_size)

    def __getitem__(self, idx):
        """
        Retrieve one training example.

        Args:
            idx (int): Starting position in the token sequence.

        Returns:
            x (LongTensor): Input sequence of length block_size.
            y (LongTensor): Target sequence (same length),
                            shifted one token ahead of x.
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y