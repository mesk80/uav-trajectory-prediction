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


def make_windows(
    trajectory: np.ndarray,
    in_len: int,
    out_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice one trajectory into (input_window, target_window) pairs.

    Example with in_len=10, out_len=5:
        window 0: input = points[0:10],  target = points[10:15]
        window 1: input = points[1:11],  target = points[11:16]
        ... and so on, sliding by 1 step each time.

    Returns
    -------
    X : np.ndarray, shape (num_windows, in_len, 3)
    Y : np.ndarray, shape (num_windows, out_len, 3)
    """
    X, Y = [], []
    n = len(trajectory)
    last_start = n - in_len - out_len + 1

    for start in range(last_start):
        X.append(trajectory[start : start + in_len])
        Y.append(trajectory[start + in_len : start + in_len + out_len])

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_test_split_by_time(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows into train/test sets.

    IMPORTANT: this split by TIME ORDER (first 80% = train, last 20% =
    test), not randomly. For trajectory/time-series data, a random
    split would leak information from the future into training and
    make test results look better than they really are.
    """
    split_index = int(train_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    return X_train, X_test, Y_train, Y_test


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
