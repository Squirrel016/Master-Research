"""Feature scaling utilities (fit on train only, transform train/val)."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

POPULATION_SCALER_KEY = "population"


def fit_minmax_scaler(values: np.ndarray) -> MinMaxScaler:
    """Fit MinMaxScaler on a 1-D array of training values."""
    scaler = MinMaxScaler()
    scaler.fit(np.asarray(values, dtype=float).reshape(-1, 1))
    return scaler


def fit_population_scaler_on_indices(
    seq: np.ndarray,
    targets: np.ndarray,
    train_indices: list[int] | np.ndarray,
) -> MinMaxScaler:
    """
    Fit population scaler using only training-window values.

    Uses all population points in each train window (24 history steps + 1 target).
    """
    train_indices = np.asarray(train_indices, dtype=int)
    values: list[np.ndarray] = []
    for idx in train_indices:
        values.append(seq[idx][:, 0])
        values.append([targets[idx]])
    return fit_minmax_scaler(np.concatenate(values))


def transform_population(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.transform(flat).reshape(values.shape).astype(np.float32)


def inverse_population(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(flat).flatten()


def save_scalers(scalers: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, path)


def load_scalers(path: Path) -> dict[str, Any]:
    return joblib.load(path)
