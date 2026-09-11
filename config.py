"""
config.py

All the "settings" of the project live here, in one place.

We use a Python @dataclass because it's simple: it's basically a
class that just holds values, with a clean printable form.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ----- data settings -----
    n_points: int = 300        # how many time steps in our fake trajectory
    noise_std: float = 0.05    # amount of random noise added (simulates sensor noise)
    in_len: int = 10           # how many past steps the model sees (input window)
    out_len: int = 5           # how many future steps the model must predict
    train_ratio: float = 0.8   # fraction of data used for training vs testing

    # ----- model settings -----
    input_size: int = 3        # x, y, z -> 3 numbers per time step
    hidden_size: int = 32      # size of the LSTM's internal memory
    output_size: int = 3       # predict x, y, z too

    # ----- training settings -----
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3

    # ----- misc -----
    seed: int = 0
    output_dir: str = "outputs"
