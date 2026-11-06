"""
split.py

Split flight CSV files into train / validation / test groups.

The split is done on whole files (one file = one flight), so no flight
is shared between groups. The result depends only on the file names,
the ratios and the seed, so it is reproducible.
"""

import random
from pathlib import Path
from typing import List, Tuple

# Files in the folder that are not flights
IGNORED_FILES = {"metadata.csv"}


def split_files(
    folder: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Return (train_files, val_files, test_files) as lists of CSV paths.

    The test ratio is what remains: 1 - train_ratio - val_ratio.
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Need train_ratio > 0, val_ratio >= 0 and train + val < 1")

    # Sort first so the result does not depend on the OS file order
    files = sorted(p for p in Path(folder).glob("*.csv") if p.name not in IGNORED_FILES)
    if len(files) < 3:
        raise ValueError(f"Need at least 3 CSV files, found {len(files)} in '{folder}'")

    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = round(train_ratio * n)
    n_val = round(val_ratio * n)

    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]

    if not train or not test:
        raise ValueError("Train or test group is empty. Change the ratios or add files.")

    return train, val, test


if __name__ == "__main__":
    # Quick check: python split.py <csv_folder>
    import sys

    train, val, test = split_files(sys.argv[1])
    for name, group in [("train", train), ("val", val), ("test", test)]:
        print(f"{name} ({len(group)}): {[p.name for p in group]}")

        