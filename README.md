# UAV Trajectory Prediction — Walking Skeleton

A minimal, end-to-end pipeline: fake data -> LSTM/GRU model -> training -> metrics -> plot.
Goal: prove the whole pipeline works before plugging in real ADS-B data.

## How to run

```bash
pip install -r requirements.txt --break-system-packages   # if needed on Ubuntu
python main.py
```

This trains both an LSTM and a GRU model on the same fake trajectory, prints
MAE/RMSE/ADE/FDE for each, and saves two plots into `outputs/`.

## Project structure

```
config.py                       # all hyperparameters in one place
data/
    generator.py                 # fake trajectory generator (stand-in for real data)
    dataset.py                   # windowing + PyTorch Dataset class
models/
    recurrent_predictor.py       # LSTM/GRU model (one class, cell_type switch)
training/
    trainer.py                   # training loop + evaluation, reusable for any model
utils/
    metrics.py                   # MAE, RMSE, ADE, FDE
    visualize.py                 # plot predicted vs real path
main.py                          # runs the full pipeline end-to-end
```

## Swapping in real data later

Only `data/generator.py` needs a replacement (e.g. `data/real_loader.py` that
reads a cleaned ADS-B CSV and returns an array shaped `(n_points, 3)`).
Everything downstream — windowing, model, training, metrics — stays the same.
