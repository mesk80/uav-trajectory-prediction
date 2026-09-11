"""
models/recurrent_predictor.py

Defines the neural network itself.

a class that can act as either an LSTM model or a GRU
model, controlled by a single argument (`cell_type`). This is
important since chapter 3 of proposal compares LSTM vs GRU, 
we want the *only* difference between two experiments to be 
the cell type — not two separately-written, slightly-different 
pieces of code. Using one shared class makes the comparison fair 
and the code much shorter.

PyTorch already provides `nn.LSTM` and `nn.GRU` as ready-made,
tested building blocks — we are not writing LSTM math by hand. We
are just wiring the pieces together: recurrent layer -> a small
linear (fully connected) layer that turns the LSTM/GRU's internal
state into an actual (x, y, z) prediction.
"""

import torch
import torch.nn as nn


class RecurrentTrajectoryPredictor(nn.Module):
    """
    A recurrent network (LSTM or GRU) that reads a window of past
    positions and predicts a window of future positions.

    Parameters
    ----------
    input_size : number of features per time step (3 for x, y, z)
    hidden_size : size of the recurrent layer's internal memory
    output_size : number of features to predict per future step (3)
    out_len : how many future time steps to output
    cell_type : "lstm" or "gru"
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 32,
        output_size: int = 3,
        out_len: int = 5,
        cell_type: str = "lstm",
    ):
        super().__init__()

        cell_type = cell_type.lower()
        if cell_type == "lstm":
            self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif cell_type == "gru":
            self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type!r}")

        self.cell_type = cell_type
        self.out_len = out_len
        self.output_size = output_size

        # This small layer converts the recurrent layer's final hidden
        # state into out_len * output_size numbers (our future path).
        self.head = nn.Linear(hidden_size, out_len * output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, in_len, input_size)
        returns shape: (batch_size, out_len, output_size)
        """
        # `recurrent_out` holds the output at every time step;
        # we only need the LAST time step's summary of the whole
        # input window, which is why we index [:, -1, :].
        recurrent_out, _ = self.recurrent(x)
        last_step = recurrent_out[:, -1, :]

        flat_prediction = self.head(last_step)
        prediction = flat_prediction.view(-1, self.out_len, self.output_size)
        return prediction
