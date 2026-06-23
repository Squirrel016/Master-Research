"""Simple forecasting baselines for comparison with CrowdLSTM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import NUM_STATIONS

MAPE_EPSILON = 1e-8
STATION_NAMES = ["ikebukuro", "nihonbashi", "shibuya", "shinjuku"]

BASELINE_NAMES = {
    "persistence": "Persistence (t-1)",
    "seasonal_naive": "Seasonal Naive (t-24)",
}


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    n_samples: int


def station_label(station_id: int) -> str:
    if 0 <= station_id < len(STATION_NAMES):
        return STATION_NAMES[station_id]
    return f"station_{station_id}"


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    epsilon: float = MAPE_EPSILON,
) -> Metrics:
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    n = len(targets)

    if n == 0:
        return Metrics(mae=float("nan"), rmse=float("nan"), mape=float("nan"), r2=float("nan"), n_samples=0)

    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors) / (np.abs(targets) + epsilon)) * 100.0)

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return Metrics(mae=mae, rmse=rmse, mape=mape, r2=r2, n_samples=n)


def persistence_predict(seq_population: np.ndarray) -> float:
    """Predict next hour as the previous hour (last step in the window)."""
    return float(seq_population[-1])


def seasonal_naive_predict(seq_population: np.ndarray) -> float:
    """Predict next hour as the same hour yesterday (first step in the 24h window)."""
    return float(seq_population[0])


def collect_baseline_predictions(
    dataset,
    indices: list[int] | np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    persistence_preds: list[float] = []
    seasonal_preds: list[float] = []
    targets: list[float] = []
    station_ids: list[int] = []
    hours: list[int] = []

    for idx in indices:
        seq_pop = dataset._seq[idx][:, 0]
        persistence_preds.append(persistence_predict(seq_pop))
        seasonal_preds.append(seasonal_naive_predict(seq_pop))
        targets.append(float(dataset._targets[idx]))
        station_ids.append(int(dataset._station_ids[idx]))
        hours.append(dataset.get_target_hour(idx))

    preds = {
        "persistence": np.array(persistence_preds, dtype=float),
        "seasonal_naive": np.array(seasonal_preds, dtype=float),
    }
    return (
        preds,
        np.array(targets, dtype=float),
        np.array(station_ids, dtype=int),
        np.array(hours, dtype=int),
    )


def evaluate_baselines(
    dataset,
    indices: list[int] | np.ndarray,
) -> tuple[dict[str, Metrics], dict[str, dict[int, Metrics]]]:
    preds, targets, station_ids, _ = collect_baseline_predictions(dataset, indices)

    overall: dict[str, Metrics] = {}
    per_station: dict[str, dict[int, Metrics]] = {k: {} for k in preds}

    for name, pred in preds.items():
        overall[name] = compute_metrics(pred, targets)
        for sid in range(NUM_STATIONS):
            mask = station_ids == sid
            if mask.any():
                per_station[name][sid] = compute_metrics(pred[mask], targets[mask])

    return overall, per_station


def log_baseline_results(
    overall: dict[str, Metrics],
    per_station: dict[str, dict[int, Metrics]],
) -> None:
    print("\n" + "=" * 50)
    print("SIMPLE BASELINES (validation set, raw population scale)")
    print("=" * 50)
    for key, title in BASELINE_NAMES.items():
        m = overall[key]
        print(f"\n{title}")
        print(f"  Samples : {m.n_samples}")
        print(f"  MAE     : {m.mae:.2f}")
        print(f"  RMSE    : {m.rmse:.2f}")
        print(f"  MAPE    : {m.mape:.2f}%")
        print(f"  R^2     : {m.r2:.4f}")

    print("\n" + "-" * 50)
    print("Per-station baselines")
    print("-" * 50)
    for sid in range(NUM_STATIONS):
        if not any(sid in per_station[k] for k in BASELINE_NAMES):
            continue
        print(f"\n  Station: {station_label(sid)}")
        for key, title in BASELINE_NAMES.items():
            if sid not in per_station[key]:
                continue
            m = per_station[key][sid]
            print(f"    {title:28s} MAE={m.mae:.2f} RMSE={m.rmse:.2f} MAPE={m.mape:.2f}% R^2={m.r2:.4f}")
