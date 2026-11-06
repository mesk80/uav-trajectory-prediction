"""
loader.py

Read flight CSV files and return clean trajectory segments.

For each flight file, in this order:
    1. read the needed columns, drop bad rows and repeated timestamps
    2. move the origin: the first point of the flight becomes (0, 0, 0)
    3. cut the flight wherever the time gap is larger than `max_gap`
    4. resample every piece to a fixed time step `dt`

The result is a list of arrays of shape (N, F), one per continuous
segment. This is the input of `make_windows` in windowing.py.
"""

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

POS_COLS = ["N", "E", "D"]            # position columns, always first
ANGLE_COLS = {"roll", "pitch", "yaw"}  # not supported (angle wrap-around)


def resample(t: np.ndarray, values: np.ndarray, dt: float) -> np.ndarray:
    """Linear interpolation of `values` (N, F) onto a uniform time grid."""
    n_steps = int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1
    grid = t[0] + dt * np.arange(n_steps)
    columns = [np.interp(grid, t, values[:, j]) for j in range(values.shape[1])]
    return np.stack(columns, axis=1)


def load_flight(
    path: Path,
    feature_cols: Sequence[str] = POS_COLS,
    dt: float = 0.1,
    max_gap: float = 2.0,
) -> List[np.ndarray]:
    """
    Load one flight CSV and return its segments.

    Parameters
    ----------
    path : CSV file made by extract_ulog.py
    feature_cols : columns to keep. The first three must be N, E, D.
    dt : time step of the output, in seconds
    max_gap : a time gap larger than this (seconds) cuts the flight

    Returns
    -------
    List of arrays, each of shape (N_i, len(feature_cols)).
    """
    feature_cols = list(feature_cols)
    if feature_cols[:3] != POS_COLS:
        raise ValueError(f"The first 3 feature columns must be {POS_COLS}")
    if ANGLE_COLS & set(feature_cols):
        raise NotImplementedError("Angle columns need unwrapping, not supported yet")

    # 1. Read, remove bad rows and repeated timestamps, sort by time
    df = pd.read_csv(path, usecols=["t", *feature_cols])
    df = df.dropna().drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    if len(df) < 2:
        return []

    # 2. Same origin for the whole flight: the first point becomes (0, 0, 0)
    df[POS_COLS] = df[POS_COLS] - df[POS_COLS].iloc[0]
    
    # 3. Cut where the time gap is too large
    t = df["t"].to_numpy()
    values = df[feature_cols].to_numpy(dtype=float)
    cut_points = np.where(np.diff(t) > max_gap)[0] + 1

    # 4. Resample each piece to a fixed time step
    segments = []
    for t_piece, v_piece in zip(np.split(t, cut_points), np.split(values, cut_points)):
        if len(t_piece) >= 2:  # need two points to interpolate
            segments.append(resample(t_piece, v_piece, dt))
    return segments


def load_segments(
    files: Sequence[Path],
    feature_cols: Sequence[str] = POS_COLS,
    dt: float = 0.1,
    max_gap: float = 2.0,
) -> List[np.ndarray]:
    """Load many flight files and return all their segments in one list."""
    segments = []
    for path in files:
        segments += load_flight(path, feature_cols, dt, max_gap)

    total_time = sum(len(s) for s in segments) * dt
    print(f"loaded {len(files)} files -> {len(segments)} segments, "
          f"{total_time / 60:.1f} min of data")
    return segments
