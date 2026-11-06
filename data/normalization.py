"""
normalization.py

Min-Max scaling to [0, 1], fitted on the training data only.

Use one scaler for the inputs X and one for the targets Y:
    x_scaler = MinMaxScaler().fit(X_train)
    y_scaler = MinMaxScaler(shared=True).fit(Y_train)

Fit on train only, then apply the same scaler to val and test.
Otherwise information from val/test leaks into training.
"""

import numpy as np


class MinMaxScaler:
    """
    Scale the last axis of an array to [0, 1].

    Works on any shape (..., F). Each of the F features gets its own
    min and max, found over all windows and all time steps.

    Parameters
    ----------
    shared : if True, one min and one max are used for all F features.
        This keeps the same scale for N, E, D (used for displacements Y,
        so that errors in the scaled space stay proportional to meters).
    """

    def __init__(self, shared: bool = False):
        self.shared = shared
        self.min_ = None    # minimum of the training data
        self.range_ = None  # max - min of the training data

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Compute min and range from the training data."""
        flat = data.reshape(-1, data.shape[-1])  # (all points, F)
        if self.shared:
            low, high = flat.min(), flat.max()
        else:
            low, high = flat.min(axis=0), flat.max(axis=0)

        value_range = np.asarray(high - low, dtype=np.float64)
        value_range = np.where(value_range == 0, 1.0, value_range)  # constant feature

        self.min_ = np.asarray(low, dtype=np.float64)
        self.range_ = value_range
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Real units -> scaled. Val/test values can fall outside [0, 1]."""
        self._check_fitted()
        return ((data - self.min_) / self.range_).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Scaled -> real units (meters, m/s, ...)."""
        self._check_fitted()
        return (data * self.range_ + self.min_).astype(np.float32)

    def save(self, path: str) -> None:
        """Save the scaler so it can be reused after training."""
        self._check_fitted()
        np.savez(path, min_=self.min_, range_=self.range_, shared=self.shared)

    @classmethod
    def load(cls, path: str) -> "MinMaxScaler":
        """Load a scaler saved with `save` (path must end with .npz)."""
        saved = np.load(path)
        scaler = cls(shared=bool(saved["shared"]))
        scaler.min_ = saved["min_"]
        scaler.range_ = saved["range_"]
        return scaler

    def _check_fitted(self) -> None:
        if self.min_ is None:
            raise RuntimeError("Scaler is not fitted. Call fit() first.")
