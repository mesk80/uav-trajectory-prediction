"""
main.py
it just calls functions
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from data.dataset import TrajectoryDataset
from data.loader import load_segments
from data.normalization import MinMaxScaler
from data.split import split_files
from data.windowing import make_windows, seconds_to_steps
from models.recurrent_predictor import RecurrentTrajectoryPredictor
from training.trainer import Trainer
from utils.metrics import compute_all_metrics
from utils.visualize import plot_prediction_vs_real


def build_windows(files, cfg: Config, in_len: int, out_len: int):
    """CSV files -> clean segments -> windows (X, Y, anchor)."""
    segments = load_segments(
        files, feature_cols=cfg.feature_cols, max_gap=cfg.max_gap, start_tol=cfg.start_tol
    )
    return make_windows(
        segments, in_len=in_len, out_len=out_len,
        stride=cfg.stride, relative_input=cfg.relative_input,
    )


def prepare_data(cfg: Config) -> dict:
    """
    Build train/val/test windows and fit the scalers.
    Runs once, so LSTM and GRU are trained on exactly the same data.
    """
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # convert in_len and out_len from seconds to number of samples
    in_len = seconds_to_steps(cfg.in_sec, cfg.dt)
    out_len = seconds_to_steps(cfg.out_sec, cfg.dt)

    # split data into train / val / test groups (by flight file)
    train_files, val_files, test_files = split_files(
        cfg.data_folder, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed
    )
    print(f"flights: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # save which flight is in which group (for reproducibility and the thesis)
    with open(Path(cfg.output_dir) / "split.txt", "w") as f:
        for name, group in [("train", train_files), ("val", val_files), ("test", test_files)]:
            f.write(f"{name}: {', '.join(p.name for p in group)}\n")

    # windows for each group
    print("\n--- train ---")
    X_train, Y_train, _ = build_windows(train_files, cfg, in_len, out_len)
    print("--- val ---")
    X_val, Y_val, _ = build_windows(val_files, cfg, in_len, out_len)
    print("--- test ---")
    X_test, Y_test, anchor_test = build_windows(test_files, cfg, in_len, out_len)

    # scalers are fitted on train only
    x_scaler = MinMaxScaler().fit(X_train)             # one min/max per feature
    y_scaler = MinMaxScaler(shared=True).fit(Y_train)  # one scale for N, E, D (keeps meters proportional)
    x_scaler.save(f"{cfg.output_dir}/x_scaler.npz")
    y_scaler.save(f"{cfg.output_dir}/y_scaler.npz")

    return {
        "out_len": out_len,
        "x_scaler": x_scaler, "y_scaler": y_scaler,
        "X_train": X_train, "Y_train": Y_train,
        "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test, "anchor_test": anchor_test,
    }


def evaluate_in_meters(trainer, x_scaler, y_scaler, X, Y):
    """Predict on real-unit windows. Returns (predictions, metrics) in meters."""
    # trainer.evaluate works on scaled data, so its own metrics are ignored here
    scaled_pred = trainer.evaluate(x_scaler.transform(X), y_scaler.transform(Y))["predictions"]
    pred = y_scaler.inverse_transform(scaled_pred)
    return pred, compute_all_metrics(pred, Y)


def run_pipeline(cfg: Config, data: dict, cell_type: str = "lstm") -> dict:
    """Trains and evaluates one model, for a given cell_type ('lstm' or 'gru')."""
    x_scaler, y_scaler = data["x_scaler"], data["y_scaler"]

    # ---- 1. reproducibility ----
    torch.manual_seed(cfg.seed)

    # ---- 2. data loader (train windows, scaled) ----
    train_dataset = TrajectoryDataset(
        x_scaler.transform(data["X_train"]), y_scaler.transform(data["Y_train"])
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    print(f"\n=== [{cell_type.upper()}] {len(train_dataset)} train windows ===")

    # ---- 3. model ----
    model = RecurrentTrajectoryPredictor(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        out_len=data["out_len"],
        cell_type=cell_type,
    )

    # ---- 4. training ----
    trainer = Trainer(model, learning_rate=cfg.learning_rate)
    trainer.train(train_loader, epochs=cfg.epochs)

    # ---- 5. evaluation (metrics in meters) ----
    _, val_metrics = evaluate_in_meters(trainer, x_scaler, y_scaler, data["X_val"], data["Y_val"])
    pred_test, test_metrics = evaluate_in_meters(
        trainer, x_scaler, y_scaler, data["X_test"], data["Y_test"]
    )
    print(f"[{cell_type.upper()}] val metrics:  {val_metrics}")
    print(f"[{cell_type.upper()}] test metrics: {test_metrics}")

    # ---- 6. visualization: a few examples from the test set ----
    X_test, Y_test, anchor = data["X_test"], data["Y_test"], data["anchor_test"]
    for idx in np.linspace(0, len(X_test) - 1, 3, dtype=int):
        history = X_test[idx, :, :3]
        if cfg.relative_input:
            history = history + anchor[idx]  # back to the flight frame
        plot_prediction_vs_real(
            input_window=history,
            predicted=pred_test[idx] + anchor[idx],
            real=Y_test[idx] + anchor[idx],
            save_path=f"{cfg.output_dir}/{cell_type}_example_{idx}.png",
            title=f"{cell_type.upper()} prediction example (test window {idx})",
        )

    return {"val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    cfg = Config()
    data = prepare_data(cfg)

    lstm_metrics = run_pipeline(cfg, data, cell_type="lstm")
    gru_metrics = run_pipeline(cfg, data, cell_type="gru")

    print("\n=== Summary (test, meters) ===")
    print("LSTM:", lstm_metrics["test"])
    print("GRU :", gru_metrics["test"])