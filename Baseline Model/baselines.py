"""Forecasting baselines for comparison with CrowdLSTM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import NUM_STATIONS
from time_encoding import TIME_FEATURE_DIM

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

MAPE_EPSILON = 1e-8
STATION_NAMES = ["ikebukuro", "nihonbashi", "shibuya", "shinjuku"]
WINDOW_LAGS = 24

BASELINE_KEYS = ("persistence", "seasonal_naive", "ridge_lag", "lightgbm")

BASELINE_NAMES = {
    "persistence": "Persistence (t-1)",
    "seasonal_naive": "Seasonal Naive (t-24)",
    "ridge_lag": "Ridge (lag features)",
    "lightgbm": "LightGBM (lag features)",
}

RULE_BASELINE_KEYS = ("persistence", "seasonal_naive")
ML_BASELINE_KEYS = ("ridge_lag", "lightgbm")


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


def get_model_comparison_specs(scope_suffix: str = "") -> list[tuple[str, str]]:
    """Return (display_name, result_csv_scope) pairs for report aggregation."""
    lstm_scope = "overall" if not scope_suffix else f"{scope_suffix}_overall"
    specs: list[tuple[str, str]] = [("CrowdLSTM", lstm_scope)]
    for key in BASELINE_KEYS:
        baseline_scope = f"baseline_{key}"
        if scope_suffix:
            baseline_scope = f"{scope_suffix}_{baseline_scope}"
        specs.append((BASELINE_NAMES[key], baseline_scope))
    return specs


def get_baseline_per_station_specs(scope_suffix: str = "") -> list[tuple[str, str]]:
    """Return (display_name, result_csv_scope) pairs for per-station baseline tables."""
    station_scope = "station" if not scope_suffix else f"{scope_suffix}_station"
    specs: list[tuple[str, str]] = [("CrowdLSTM", station_scope)]
    for key in BASELINE_KEYS:
        baseline_scope = f"baseline_{key}"
        if scope_suffix:
            baseline_scope = f"{scope_suffix}_{baseline_scope}"
        specs.append((BASELINE_NAMES[key], baseline_scope))
    return specs


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


def build_tabular_features(
    dataset,
    indices: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build tabular features for Ridge / LightGBM.

    Features: 24 population lags (t-1 .. t-24), target-hour sin/cos encodings,
    and station one-hot indicators. All values use raw (unscaled) population.
    """
    indices = np.asarray(indices, dtype=int)
    n = len(indices)
    n_features = WINDOW_LAGS + TIME_FEATURE_DIM + NUM_STATIONS
    features = np.zeros((n, n_features), dtype=np.float32)

    for row, idx in enumerate(indices):
        population = dataset._seq[idx][:, 0]
        features[row, :WINDOW_LAGS] = population[::-1]
        features[row, WINDOW_LAGS : WINDOW_LAGS + TIME_FEATURE_DIM] = dataset._current_time_enc[idx]
        station_id = int(dataset._station_ids[idx])
        features[row, WINDOW_LAGS + TIME_FEATURE_DIM + station_id] = 1.0

    targets = dataset._targets[indices].astype(np.float32)
    station_ids = dataset._station_ids[indices].astype(np.int64)
    hours = dataset._target_hours[indices].astype(np.int64)
    return features, targets, station_ids, hours


def collect_rule_baseline_predictions(
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


def fit_predict_ridge(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def fit_predict_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
) -> np.ndarray:
    if lgb is None:
        raise ImportError("lightgbm is required for the LightGBM baseline. Install with: pip install lightgbm")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def collect_baseline_predictions(
    dataset,
    eval_indices: list[int] | np.ndarray,
    train_indices: list[int] | np.ndarray,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rule_preds, targets, station_ids, hours = collect_rule_baseline_predictions(dataset, eval_indices)

    X_train, y_train, _, _ = build_tabular_features(dataset, train_indices)
    X_eval, _, _, _ = build_tabular_features(dataset, eval_indices)

    preds = dict(rule_preds)
    preds["ridge_lag"] = fit_predict_ridge(X_train, y_train, X_eval)
    preds["lightgbm"] = fit_predict_lightgbm(X_train, y_train, X_eval, seed)
    return preds, targets, station_ids, hours


def evaluate_baselines(
    dataset,
    eval_indices: list[int] | np.ndarray,
    train_indices: list[int] | np.ndarray,
    seed: int = 42,
) -> tuple[dict[str, Metrics], dict[str, dict[int, Metrics]]]:
    preds, targets, station_ids, _ = collect_baseline_predictions(
        dataset, eval_indices, train_indices, seed=seed
    )

    overall: dict[str, Metrics] = {}
    per_station: dict[str, dict[int, Metrics]] = {k: {} for k in BASELINE_KEYS}

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
    split_name: str,
) -> None:
    for key, title in BASELINE_NAMES.items():
        m = overall[key]
        print(f"\n{title}")
        print(f"  Samples : {m.n_samples}")
        print(f"  MAE     : {m.mae:.2f}")
        print(f"  RMSE    : {m.rmse:.2f}")
        print(f"  MAPE    : {m.mape:.2f}%")
        print(f"  R^2     : {m.r2:.4f}")

    print("\n" + "-" * 50)
    print(f"Per-station baselines ({split_name} set)")
    print("-" * 50)
    for sid in range(NUM_STATIONS):
        if not any(sid in per_station[k] for k in BASELINE_KEYS):
            continue
        print(f"\n  Station: {station_label(sid)}")
        for key, title in BASELINE_NAMES.items():
            if sid not in per_station[key]:
                continue
            m = per_station[key][sid]
            print(f"    {title:28s} MAE={m.mae:.2f} RMSE={m.rmse:.2f} MAPE={m.mape:.2f}% R^2={m.r2:.4f}")
