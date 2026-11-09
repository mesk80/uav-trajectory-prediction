"""
config.py

All the "settings" of the project live here, in one place.

We use a Python @dataclass because it's simple: it's basically a
class that just holds values, with a clean printable form.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ----- data settings -----
    data_folder: str = "/home/mamali/Uni/data/flight_review/app/data/downloaded/fixedwing/csvs"  # flight CSV files
    train_ratio: float = 0.7     # fraction of flights used for training
    val_ratio: float = 0.15      # fraction of flights used for validation (the rest is test)
    in_sec: float = 10.0         # how many seconds the model sees (input window)
    out_sec: float = 5.0         # how many seconds the model must predict
    dt: float = 0.1              # time step of the data, in seconds
    stride: int = 1              # how many samples the window moves each step
    relative_input: bool = True  # if True, input positions are relative to the last input point
    max_gap: float = 0.5         # a time gap larger than this (seconds) cuts the flight
    start_tol: float = 1.0       # meters; rows this close to the start (before takeoff) are dropped
    feature_cols: Tuple[str, ...] = ("N", "E", "D")  # input columns, position (N, E, D) first

    # ----- model settings -----
    hidden_size: int = 32      # size of the LSTM/GRU's internal memory
    output_size: int = 3       # predict N, E, D

    # ----- training settings -----
    epochs: int = 60
    patience: int = 10         # stop if the validation loss does not improve for this many epochs
    batch_size: int = 16
    learning_rate: float = 1e-3

    # ----- misc -----
    seed: int = 0
    output_dir: str = "outputs"

    @property
    def input_size(self) -> int:
        """Number of input features per time step (follows feature_cols)."""
        return len(self.feature_cols)
