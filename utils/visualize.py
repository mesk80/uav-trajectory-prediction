"""
utils/visualize.py

One plotting function: draw the real path vs the predicted path,
so you can SEE how good (or bad) the model is, not just read numbers.

"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_full_trajectory(
    trajectory: np.ndarray,
    split_index: int,
    in_len: int,
    predicted_test: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Full trajectory: train vs test",
) -> None:
    """
    Plot the WHOLE generated trajectory (not just one small window),
    split into two parts:

    - the train part (blue): every raw point the model was trained on
    - the test part (green): the real path the model never saw during
      training, plus (red) what the model predicted for that same part

    How the split point is found
    -----------------------------
    `split_index` is how many WINDOWS went to training (this is just
    `len(X_train)` from train_test_split_by_time — you already have
    it, no extra computation needed). Since each window starts
    `in_len` steps before its target, the first RAW point that
    belongs to the test period is at:

        boundary = split_index + in_len

    Everything in `trajectory` before `boundary` is train data.
    Everything from `boundary` onward is real test data.

    How the predicted line is built
    ---------------------------------
    `predicted_test` has one prediction window per test sample, shape
    (num_test_windows, out_len, 3), and these windows overlap (each
    slides forward by only 1 raw step). To turn this into a single
    smooth curve instead of a pile of overlapping short lines, we
    keep only the FIRST predicted step of every window
    (`predicted_test[:, 0, :]`). Because windows slide by 1 step,
    this gives exactly one predicted point per raw time step,
    starting at `boundary` — so it lines up point-for-point with the
    real test curve.
    """
    boundary = split_index + in_len

    train_part = trajectory[:boundary]
    test_part_real = trajectory[boundary:]
    test_part_predicted = predicted_test[:, 0, :]  # one point per raw test step

    plt.figure(figsize=(7, 7))

    plt.plot(train_part[:, 0], train_part[:, 1], "-", color="steelblue", label="train (seen by model)")
    plt.plot(test_part_real[:, 0], test_part_real[:, 1], "-", color="green", label="test - real path")
    plt.plot(test_part_predicted[:, 0], test_part_predicted[:, 1], "--", color="red", label="test - predicted path")

    plt.scatter(*trajectory[boundary], color="black", zorder=5, label="train/test boundary")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_vs_real(
    input_window: np.ndarray,
    predicted: np.ndarray,
    real: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Trajectory Prediction",
) -> None:
    """
    Plot one example: the past input path (context), the model's
    predicted future path, and the real future path — using the x
    and y coordinates (a top-down view, like looking at a map).

    Parameters
    ----------
    input_window : shape (in_len, 3) — the past points the model saw
    predicted : shape (out_len, 3) — what the model predicted
    real : shape (out_len, 3) — what actually happened
    save_path : if given, ALSO saves the plot as an image file
                (in addition to showing it on screen)
    """
    plt.figure(figsize=(6, 6))

    plt.plot(input_window[:, 0], input_window[:, 1], "o-", color="gray", label="past (input)")
    plt.plot(real[:, 0], real[:, 1], "o-", color="green", label="real future")
    plt.plot(predicted[:, 0], predicted[:, 1], "o--", color="red", label="predicted future")

    # mark the boundary between "past" and "future" clearly
    plt.scatter(*input_window[-1, :2], color="black", zorder=5, label="prediction start")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # IMPORTANT: save BEFORE show(). On some matplotlib backends,
    # show() clears the figure once the window closes, so a savefig()
    # called after show() would save a blank image.
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()
    plt.close()
