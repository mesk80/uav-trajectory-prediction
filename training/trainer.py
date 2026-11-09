"""
training/trainer.py

A `Trainer` class wraps everything about "teaching" the model:
- looping over the data multiple times (epochs)
- computing the loss (how wrong the model is)
- letting PyTorch adjust the model's internal numbers (backpropagation)
- checking the loss on validation data after every epoch
- stopping early, and keeping the best weights, when validation stops improving
- predicting on new windows
- keeping track of loss history, so we can see if the model is
  actually learning

Why wrap this in a class instead of one long script? Because later,
when you train BOTH an LSTM model and a GRU model, you will reuse
this exact same class for both — you just pass in a different model.
This is the benefit of "modular" code: you write the training logic
once, and it works for any model that follows the same interface.

Note: the trainer works on SCALED data (see data/normalization.py).
Losses and predictions here are in scaled units. Converting back to
meters and computing the metrics is done in main.py.
"""

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        self.val_loss_history: List[float] = []
        self.best_epoch: Optional[int] = None  # epoch with the lowest validation loss

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        patience: Optional[int] = 10,
        verbose: bool = True,
    ) -> None:
        """
        Run the training loop.

        For each epoch (one full pass over the training data):
          1. Show the model a batch of examples
          2. Compare its prediction to the real answer (the loss)
          3. Adjust the model's weights slightly to reduce the loss
        Repeat for every batch, then check the loss on the validation data.

        If `val_loader` is given, the weights of the epoch with the lowest
        validation loss are restored at the end. If `patience` is also set,
        training stops when the validation loss has not improved for that
        many epochs (use patience=None to always run all epochs).
        """
        best_val_loss = float("inf")
        best_state = None
        epochs_without_gain = 0

        for epoch in range(1, epochs + 1):
            # ---- training pass ----
            self.model.train()  # set every epoch, because validation switches to "eval" mode
            total_loss, count = 0.0, 0

            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                self.optimizer.zero_grad()          # clear old gradients
                predictions = self.model(X_batch)    # forward pass
                loss = self.criterion(predictions, Y_batch)
                loss.backward()                      # compute new gradients
                self.optimizer.step()                # update model weights

                total_loss += loss.item() * len(X_batch)  # weight by batch size
                count += len(X_batch)

            train_loss = total_loss / count
            self.train_loss_history.append(train_loss)
            message = f"epoch {epoch:3d}/{epochs} | train MSE loss: {train_loss:.5f}"

            # ---- validation pass ----
            if val_loader is not None:
                val_loss = self._loss_on(val_loader)
                self.val_loss_history.append(val_loss)
                message += f" | val MSE loss: {val_loss:.5f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())  # remember these weights
                    epochs_without_gain = 0
                else:
                    epochs_without_gain += 1

            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(message)

            # ---- early stopping ----
            if val_loader is not None and patience is not None and epochs_without_gain >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best epoch: {self.best_epoch})")
                break

        # go back to the best weights seen on the validation data
        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()  # we don't need gradients when just evaluating
    def _loss_on(self, loader: DataLoader) -> float:
        """Average MSE loss over a data loader, without changing the weights."""
        self.model.eval()  # switch to "evaluation mode"
        total_loss, count = 0.0, 0

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            loss = self.criterion(self.model(X_batch), Y_batch)
            total_loss += loss.item() * len(X_batch)
            count += len(X_batch)

        return total_loss / count

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Run the trained model on windows X.

        X : float32 array of shape (n, in_len, F), already scaled
        Returns an array of shape (n, out_len, 3), still in scaled units.
        Use y_scaler.inverse_transform() to get meters.
        """
        self.model.eval()
        outputs = []

        # predict in batches, so a large test set does not use too much memory
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(self.device)
            outputs.append(self.model(batch).cpu().numpy())

        return np.concatenate(outputs)
