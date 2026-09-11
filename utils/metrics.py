"""
utils/metrics.py

The error metrics:
MAE, RMSE, ADE, FDE.

These work on numpy arrays, so we `detach()` and `.numpy()` any
PyTorch tensor before passing it in (done in training/trainer.py).

Shapes expected: (num_samples, out_len, 3)
"""

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Absolute Error: average of |predicted - real|, over all points/axes."""
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Root Mean Squared Error: penalizes big errors more than MAE does."""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def average_displacement_error(pred: np.ndarray, true: np.ndarray) -> float:
    """
    ADE: average straight-line (Euclidean) distance between predicted
    and real position, averaged over every future time step.
    """
    diff = pred - true  # shape (num_samples, out_len, 3)
    distances = np.linalg.norm(diff, axis=-1)  # shape (num_samples, out_len)
    return float(np.mean(distances))


def final_displacement_error(pred: np.ndarray, true: np.ndarray) -> float:
    """
    FDE: straight-line distance only at the LAST predicted time step
    (i.e., how far off is the model at the end of the prediction horizon).
    """
    diff = pred[:, -1, :] - true[:, -1, :]  # shape (num_samples, 3)
    distances = np.linalg.norm(diff, axis=-1)
    return float(np.mean(distances))


def compute_all_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Convenience function: run every metric at once and return a dict."""
    return {
        "MAE": mae(pred, true),
        "RMSE": rmse(pred, true),
        "ADE": average_displacement_error(pred, true),
        "FDE": final_displacement_error(pred, true),
    }
