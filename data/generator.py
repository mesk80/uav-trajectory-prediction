"""
data/generator.py

This file creates FAKE flight data
"""

#TODO  Real Data loader


import numpy as np


def generate_trajectory(
    n_points: int = 300,
    noise_std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Create one fake UAV trajectory shaped like a slow circular path
    with a bit of vertical wave motion, plus random noise (to act
    like imperfect sensor/GPS readings).

    Parameters
    ----------
    n_points : how many time steps to generate
    noise_std : standard deviation of random noise added to each point
    seed : random seed, so results are repeatable

    Returns
    -------
    np.ndarray of shape (n_points, 3) -> columns are x, y, z
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_points)

    x = 10 * np.cos(t) + rng.normal(0, noise_std, n_points)
    y = 10 * np.sin(t) + rng.normal(0, noise_std, n_points)
    z = 5 + 0.5 * np.sin(t / 2) + rng.normal(0, noise_std, n_points)

    trajectory = np.stack([x, y, z], axis=1)
    return trajectory.astype(np.float32)
