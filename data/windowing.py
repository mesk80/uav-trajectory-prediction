"""
windowing.py

Turn trajectory segments into (input, target) windows for training.

A segment is one continuous piece of a flight: an array of shape (N, F).
The first 3 columns must be the position (N, E, D, in meters). Extra
columns (for example ground speed) are used as input features only.

For each window:
    X : the last `in_len` points (the history the model sees)
    Y : the next `out_len` positions, relative to the last input point
    anchor : the last input point (position only)

To get the predicted path back in the flight frame: Y + anchor.
"""

from typing import List, Tuple

import numpy as np

N_POS = 3  # number of position columns (N, E, D)


def seconds_to_steps(seconds: float, dt: float) -> int:
    """Convert a duration in seconds to a number of samples."""
    return int(round(seconds / dt))


def make_windows(
    segments: List[np.ndarray],
    in_len: int,
    out_len: int,
    stride: int = 1,
    relative_input: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over every segment and collect all windows.

    Parameters
    ----------
    segments : list of arrays, each of shape (N, F), position in columns 0-2
    in_len : number of input points
    out_len : number of future points to predict
    stride : how many points the window moves each step (1 = every point)
    relative_input : if True, the input position is also made relative to
        the last input point. If False, it stays in the flight frame.

    Returns
    -------
    X : (num_windows, in_len, F)   input history
    Y : (num_windows, out_len, 3)  future positions minus the anchor
    anchor : (num_windows, 3)      last input position (flight frame)
    """
    window_len = in_len + out_len
    X, Y, anchor = [], [], []
    skipped = 0

    for seg in segments:
        # A window must fit inside one segment, so short segments are skipped
        if len(seg) < window_len:
            skipped += 1
            continue

        for start in range(0, len(seg) - window_len + 1, stride):
            x = seg[start : start + in_len].copy()
            future = seg[start + in_len : start + window_len, :N_POS]
            last = x[-1, :N_POS].copy()

            if relative_input:
                x[:, :N_POS] -= last

            X.append(x)
            Y.append(future - last)  # displacement from the last input point
            anchor.append(last)

    if not X:
        raise ValueError(
            f"No windows created: all {len(segments)} segments are shorter "
            f"than {window_len} points."
        )

    print(f"windows: {len(X)} | segments used: {len(segments) - skipped}"
          f" | segments skipped (too short): {skipped}")

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(anchor, dtype=np.float32),
    )
