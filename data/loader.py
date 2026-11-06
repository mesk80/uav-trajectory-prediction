"""
loader.py

Read flight CSV files and return clean trajectory segments.

For each flight file, in this order:
    1. read the needed columns, drop bad rows and repeated timestamps
    2. move the origin: the first point of the flight becomes (0, 0, 0)
    3. drop the stationary start (rows still at the origin)
    4. cut the flight wherever the time gap is larger than `max_gap`

The result is a list of arrays of shape (N, F), one per continuous
segment. This is the input of `make_windows` in windowing.py.

The time step is assumed to be constant (0.1 s in our logs), so the
data is not resampled.
"""

from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

POS_COLS = ["N", "E", "D"]  # position columns, always first


def load_flight(
    path: Path,
    feature_cols: Sequence[str] = POS_COLS,
    max_gap: float = 2.0,
    start_tol: float = 0.0,
) -> List[np.ndarray]:
    """
    Load one flight CSV and return its segments.

    Parameters
    ----------
    path : CSV file made by extract_ulog.py
    feature_cols : columns to keep. The first three must be N, E, D.
    max_gap : a time gap larger than this (seconds) cuts the flight
    start_tol : rows closer than this (meters) to the origin, before the
        first real movement, are dropped. 0 means exactly at the origin.

    Returns
    -------
    List of arrays, each of shape (N_i, len(feature_cols)).
    """
    feature_cols = list(feature_cols)
    if feature_cols[:3] != POS_COLS:
        raise ValueError(f"The first 3 feature columns must be {POS_COLS}")

    # 1. Read, remove bad rows and repeated timestamps, sort by time
    df = pd.read_csv(path, usecols=["t", *feature_cols])
    df = df.dropna().drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    if len(df) < 2:
        return []

    # 2. Same origin for the whole flight: the first point becomes (0, 0, 0)
    df[POS_COLS] = df[POS_COLS] - df[POS_COLS].iloc[0]

    # 3. Drop the stationary start (everything before the first movement)
    moved = (df[POS_COLS].abs().max(axis=1) > start_tol).to_numpy()
    if not moved.any():
        return []  # the vehicle never moved
    df = df.iloc[moved.argmax():]

    # 4. Cut where the time gap is too large
    t = df["t"].to_numpy()
    values = df[feature_cols].to_numpy(dtype=float)
    cut_points = np.where(np.diff(t) > max_gap)[0] + 1
    return list(np.split(values, cut_points))


def load_segments(
    files: Sequence[Path],
    feature_cols: Sequence[str] = POS_COLS,
    max_gap: float = 2.0,
    start_tol: float = 0.0,
) -> List[np.ndarray]:
    """Load many flight files and return all their segments in one list."""
    segments = []
    for path in files:
        segments += load_flight(path, feature_cols, max_gap, start_tol)

    n_points = sum(len(s) for s in segments)
    print(f"loaded {len(files)} files -> {len(segments)} segments, {n_points} points")
    return segments