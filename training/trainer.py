"""
training/trainer.py

A `Trainer` class wraps everything about "teaching" the model:
- looping over the data multiple times (epochs)
- computing the loss (how wrong the model is)
- letting PyTorch adjust the model's internal numbers (backpropagation)
- keeping track of loss history, so we can see if the model is
  actually learning

Why wrap this in a class instead of one long script? Because later,
when you train BOTH an LSTM model and a GRU model, you will reuse
this exact same class for both — you just pass in a different model.
This is the benefit of "modular" code: you write the training logic
once, and it works for any model that follows the same interface.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import compute_all_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()  # standard choice for predicting continuous numbers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loss_history: List[float] = []

    def train(self, train_loader: DataLoader, epochs: int, verbose: bool = True) -> None:
        """
        Run the training loop.

        For each epoch (one full pass over the training data):
          1. Show the model a batch of examples
          2. Compare its prediction to the real answer (the loss)
          3. Adjust the model's weights slightly to reduce the loss
        Repeat for every batch, then repeat the whole thing `epochs` times.
        """
        self.model.train()  # tells PyTorch layers like dropout to behave in "training mode"

        for epoch in range(1, epochs + 1):
            epoch_losses = []

            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                self.optimizer.zero_grad()          # clear old gradients
                predictions = self.model(X_batch)    # forward pass
                loss = self.criterion(predictions, Y_batch)
                loss.backward()                      # compute new gradients
                self.optimizer.step()                # update model weights

                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            self.train_loss_history.append(avg_loss)

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"epoch {epoch:3d}/{epochs} | train MSE loss: {avg_loss:.5f}")

    @torch.no_grad()  # we don't need gradients when just evaluating
    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
        """
        Run the trained model on test data and compute metrics.
        Returns both the metrics dict and the raw predictions, so you
        can plot them afterward.
        """
        self.model.eval()  # switch to "evaluation mode"

        X_tensor = torch.from_numpy(X_test).to(self.device)
        predictions = self.model(X_tensor).cpu().numpy()

        metrics = compute_all_metrics(predictions, Y_test)
        return {"metrics": metrics, "predictions": predictions}
