"""
data/dataset.py

Two jobs happen here:

1. `make_windows()` — turns one long trajectory into many small
   (input, target) pairs. This is called "windowing" or
   "sliding window": we look at `in_len` past points and try to
   predict the next `out_len` points, then slide forward by 1 and
   repeat (10 seconds history -> 5 seconds prediction).

2. `TrajectoryDataset` — a small wrapper class so PyTorch knows how
   to read our data. PyTorch's training tools (DataLoader) expect
   objects with two specific methods: __len__ and __getitem__. This
   is a PyTorch convention, not something specific to our project.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Wraps our numpy arrays (X, Y) so PyTorch's DataLoader can use them.

    A PyTorch Dataset just needs to answer two questions:
      - How many samples do you have?  (__len__)
      - Give me sample number i.        (__getitem__)

    PyTorch's DataLoader then uses these to build shuffled mini-batches
    for us automatically during training.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]
