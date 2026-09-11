"""
main.py
it just calls functions
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.generator import generate_trajectory
from data.dataset import make_windows, train_test_split_by_time, TrajectoryDataset
from models.recurrent_predictor import RecurrentTrajectoryPredictor
from training.trainer import Trainer
from utils.visualize import plot_prediction_vs_real, plot_full_trajectory


def run_pipeline(cfg: Config, cell_type: str = "lstm") -> dict:
    """Runs the full pipeline once, for a given cell_type ('lstm' or 'gru')."""

    # ---- 1. reproducibility ----
    torch.manual_seed(cfg.seed)

    # ---- 2. data ----
    trajectory = generate_trajectory(
        n_points=cfg.n_points, noise_std=cfg.noise_std, seed=cfg.seed
    )
    X, Y = make_windows(trajectory, in_len=cfg.in_len, out_len=cfg.out_len)
    X_train, X_test, Y_train, Y_test = train_test_split_by_time(X, Y, cfg.train_ratio)

    train_dataset = TrajectoryDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    print(f"\n=== [{cell_type.upper()}] data ready: "
          f"{len(X_train)} train windows, {len(X_test)} test windows ===")

    # ---- 3. model ----
    model = RecurrentTrajectoryPredictor(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        out_len=cfg.out_len,
        cell_type=cell_type,
    )

    # ---- 4. training ----
    trainer = Trainer(model, learning_rate=cfg.learning_rate)
    trainer.train(train_loader, epochs=cfg.epochs)

    # ---- 5. evaluation ----
    result = trainer.evaluate(X_test, Y_test)
    print(f"[{cell_type.upper()}] test metrics: {result['metrics']}")

    # ---- 6. visualization: plot one example from the test set ----
    example_index = 0
    plot_prediction_vs_real(
        input_window=X_test[example_index],
        predicted=result["predictions"][example_index],
        real=Y_test[example_index],
        save_path=f"{cfg.output_dir}/{cell_type}_example_prediction.png",
        title=f"{cell_type.upper()} prediction example",
    )

    # ---- 7. visualization: plot the FULL trajectory (train vs test) ----
    plot_full_trajectory(
        trajectory=trajectory,
        split_index=len(X_train),
        in_len=cfg.in_len,
        predicted_test=result["predictions"],
        save_path=f"{cfg.output_dir}/{cell_type}_full_trajectory.png",
        title=f"{cell_type.upper()} - full trajectory (train vs test)",
    )

    return result["metrics"]


if __name__ == "__main__":
    cfg = Config()

    lstm_metrics = run_pipeline(cfg, cell_type="lstm")
    gru_metrics = run_pipeline(cfg, cell_type="gru")

    print("\n=== Summary ===")
    print("LSTM:", lstm_metrics)
    print("GRU :", gru_metrics)
