import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Subset

from baselines import evaluate_baselines, log_baseline_results
from dataset import CrowdDataset, temporal_train_val_indices
from model import NUM_STATIONS, CrowdLSTM
from scaling import (
    POPULATION_SCALER_KEY,
    fit_population_scaler_on_indices,
    inverse_population,
    save_scalers,
)

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "processed_data.csv"
BEST_MODEL_PATH = PROJECT_DIR / "best_model.pth"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "Result"
SCALERS_FILENAME = "scalers.joblib"

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MAPE_EPSILON = 1e-8

STATION_NAMES = ["ikebukuro", "nihonbashi", "shibuya", "shinjuku"]
HOURS = list(range(24))


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    n_samples: int


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataset_scaler(
    dataset: CrowdDataset,
    train_indices: list[int],
    output_dir: Path,
) -> MinMaxScaler:
    """Fit population scaler on train only, attach to dataset, and persist."""
    scaler = fit_population_scaler_on_indices(
        dataset._seq, dataset._targets, train_indices
    )
    dataset.set_population_scaler(scaler)
    save_scalers({POPULATION_SCALER_KEY: scaler}, output_dir / SCALERS_FILENAME)
    print(
        f"Population scaler fit on {len(train_indices)} train windows "
        f"(min={scaler.data_min_[0]:.2f}, max={scaler.data_max_[0]:.2f})"
    )
    print(f"Scalers saved to {output_dir / SCALERS_FILENAME}")
    return scaler


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


def station_label(station_id: int) -> str:
    if 0 <= station_id < len(STATION_NAMES):
        return STATION_NAMES[station_id]
    return f"station_{station_id}"


def print_metrics_report(title: str, metrics: Metrics) -> None:
    print(f"\n{title}")
    print(f"  Samples : {metrics.n_samples}")
    print(f"  MAE     : {metrics.mae:.2f}")
    print(f"  RMSE    : {metrics.rmse:.2f}")
    print(f"  MAPE    : {metrics.mape:.2f}%")
    print(f"  R^2     : {metrics.r2:.4f}")


def run_epoch(
    model: CrowdLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0

    for x_seq, station_id, current_time, y, _target_hour in loader:
        x_seq = x_seq.to(device)
        station_id = station_id.to(device)
        current_time = current_time.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(x_seq, station_id, current_time)
        loss = criterion(pred, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def collect_predictions(
    model: CrowdLSTM,
    loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets, station_ids, hours = [], [], [], []

    for x_seq, station_id, current_time, y, target_hour in loader:
        x_seq = x_seq.to(device)
        station_id = station_id.to(device)
        current_time = current_time.to(device)

        pred = model(x_seq, station_id, current_time).cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
        station_ids.append(station_id.cpu().numpy())
        hours.append(target_hour.numpy().astype(int))

    preds = inverse_population(scaler, np.concatenate(preds))
    targets = inverse_population(scaler, np.concatenate(targets))
    station_ids = np.concatenate(station_ids).astype(int)
    hours = np.concatenate(hours).astype(int)
    return preds, targets, station_ids, hours


def compute_hourly_stats(
    preds: np.ndarray,
    targets: np.ndarray,
    hours: np.ndarray,
    station_id: int | None = None,
    station_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute MAE / RMSE / MAPE for each hour (0-23) on the given sample subset."""
    rows: list[dict] = []
    for hour in HOURS:
        mask = hours == hour
        if station_id is not None and station_ids is not None:
            mask = mask & (station_ids == station_id)

        metrics = compute_metrics(preds[mask], targets[mask])
        rows.append(
            {
                "hour": hour,
                "n_samples": metrics.n_samples,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
            }
        )
    return pd.DataFrame(rows)


def format_hourly_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display["hour"] = display["hour"].apply(lambda h: f"{int(h):02d}")
    display["mae"] = display["mae"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display["rmse"] = display["rmse"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display["mape"] = display["mape"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    return display.to_string(index=False)


def log_hourly_stats(
    global_hourly: pd.DataFrame,
    per_station_hourly: dict[int, pd.DataFrame],
) -> None:
    print("\n" + "=" * 72)
    print("GLOBAL HOURLY STATS (all stations, full validation set)")
    print("=" * 72)
    print(format_hourly_table(global_hourly))

    print("\n" + "=" * 72)
    print("PER-STATION HOURLY STATS (full validation set)")
    print("=" * 72)
    for sid in sorted(per_station_hourly):
        print(f"\n--- {station_label(sid)} ---")
        print(format_hourly_table(per_station_hourly[sid]))


def plot_hourly_performance_by_station(
    per_station_hourly: dict[int, pd.DataFrame],
    save_path: Path,
) -> None:
    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for sid in sorted(per_station_hourly):
        df = per_station_hourly[sid]
        valid = df["n_samples"] > 0
        plt.plot(
            df.loc[valid, "hour"],
            df.loc[valid, "mape"],
            "o-",
            linewidth=2,
            markersize=5,
            label=station_label(sid),
            color=colors[sid % len(colors)],
        )

    plt.xticks(HOURS)
    plt.xlabel("Hour of day", fontsize=12)
    plt.ylabel("MAPE (%)", fontsize=12)
    plt.title("Hourly MAPE by Station (full validation set)", fontsize=13)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_scatter(preds: np.ndarray, targets: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(targets, preds, alpha=0.45, s=28, edgecolors="none", color="steelblue")
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    margin = (hi - lo) * 0.05 if hi > lo else 1.0
    line = np.linspace(lo - margin, hi + margin, 100)
    plt.plot(line, line, "r--", linewidth=2, label="y = x")
    plt.xlabel("Actual population", fontsize=12)
    plt.ylabel("Predicted population", fontsize=12)
    plt.title("Predicted vs Actual Population", fontsize=13)
    plt.legend(fontsize=11)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_validation(
    model: CrowdLSTM,
    val_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
) -> tuple[Metrics, dict[int, Metrics], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preds, targets, station_ids, hours = collect_predictions(model, val_loader, scaler, device)
    overall = compute_metrics(preds, targets)

    per_station: dict[int, Metrics] = {}
    for sid in range(NUM_STATIONS):
        mask = station_ids == sid
        if mask.any():
            per_station[sid] = compute_metrics(preds[mask], targets[mask])

    return overall, per_station, preds, targets, station_ids, hours


def log_evaluation_results(overall: Metrics, per_station: dict[int, Metrics]) -> None:
    print("\n" + "=" * 50)
    print("VALIDATION METRICS (real population scale, all samples)")
    print("=" * 50)
    print_metrics_report("Overall (all stations)", overall)

    print("\n" + "-" * 50)
    print("Per-station summary")
    print("-" * 50)
    for sid in sorted(per_station):
        print_metrics_report(f"Station: {station_label(sid)} (id={sid})", per_station[sid])


def save_run_results_csv(
    run_id: int,
    seed: int,
    overall: Metrics,
    per_station: dict[int, Metrics],
    global_hourly: pd.DataFrame,
    per_station_hourly: dict[int, pd.DataFrame],
    best_val_loss: float,
    output_dir: Path,
    baseline_overall: dict[str, Metrics] | None = None,
    baseline_per_station: dict[str, dict[int, Metrics]] | None = None,
) -> Path:
    """Save all validation metrics for one run into a single CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def append_metric_row(
        scope: str,
        m: Metrics,
        station_id: str | int = "",
        station_name: str = "",
        hour: str | int = "",
        loss: str | float = "",
    ) -> None:
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "scope": scope,
                "station_id": station_id,
                "station_name": station_name,
                "hour": hour,
                "mae": m.mae,
                "rmse": m.rmse,
                "mape": m.mape,
                "r2": m.r2,
                "n_samples": m.n_samples,
                "best_val_loss": loss,
            }
        )

    append_metric_row("overall", overall, loss=best_val_loss)

    for sid in sorted(per_station):
        m = per_station[sid]
        append_metric_row("station", m, station_id=sid, station_name=station_label(sid))

    if baseline_overall and baseline_per_station:
        for baseline_key in ("persistence", "seasonal_naive"):
            scope = f"baseline_{baseline_key}"
            append_metric_row(scope, baseline_overall[baseline_key])
            for sid in sorted(baseline_per_station[baseline_key]):
                m = baseline_per_station[baseline_key][sid]
                append_metric_row(scope, m, station_id=sid, station_name=station_label(sid))

    for _, row in global_hourly.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "scope": "global_hourly",
                "station_id": "",
                "station_name": "",
                "hour": int(row["hour"]),
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
                "r2": "",
                "n_samples": int(row["n_samples"]),
                "best_val_loss": "",
            }
        )

    for sid in sorted(per_station_hourly):
        for _, row in per_station_hourly[sid].iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "scope": "station_hourly",
                    "station_id": sid,
                    "station_name": station_label(sid),
                    "hour": int(row["hour"]),
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape": row["mape"],
                    "r2": "",
                    "n_samples": int(row["n_samples"]),
                    "best_val_loss": "",
                }
            )

    out_path = output_dir / f"result_run_{run_id}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def run_full_evaluation(
    model: CrowdLSTM,
    val_loader: DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
    output_dir: Path,
) -> tuple[Metrics, dict[int, Metrics], pd.DataFrame, dict[int, pd.DataFrame], np.ndarray, np.ndarray]:
    overall, per_station, preds, targets, station_ids, hours = evaluate_validation(
        model, val_loader, scaler, device
    )
    log_evaluation_results(overall, per_station)

    global_hourly = compute_hourly_stats(preds, targets, hours)
    per_station_hourly: dict[int, pd.DataFrame] = {}
    for sid in sorted(per_station):
        per_station_hourly[sid] = compute_hourly_stats(
            preds, targets, hours, station_id=sid, station_ids=station_ids
        )

    log_hourly_stats(global_hourly, per_station_hourly)

    scatter_path = output_dir / "eval_scatter.png"
    hourly_path = output_dir / "hourly_performance_by_station.png"
    plot_hourly_performance_by_station(per_station_hourly, hourly_path)
    plot_scatter(preds, targets, scatter_path)

    print("\nEvaluation figures saved:")
    print(f"  {hourly_path}")
    print(f"  {scatter_path}")
    return overall, per_station, global_hourly, per_station_hourly, preds, targets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrowdLSTM baseline model.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--run-id", type=int, default=None, help="Run index for result CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and figure outputs.",
    )
    return parser.parse_args()


def log_temporal_split(dataset: CrowdDataset, train_indices: list[int], val_indices: list[int]) -> None:
    """Print per-station train/val time ranges for the temporal split."""
    train_idx = np.array(train_indices, dtype=int)
    val_idx = np.array(val_indices, dtype=int)

    print("\nTemporal train/val split (per station, earlier -> train, later -> val)")
    print("-" * 72)
    for sid in sorted(np.unique(dataset._station_ids)):
        train_mask = dataset._station_ids[train_idx] == sid
        val_mask = dataset._station_ids[val_idx] == sid
        train_t = pd.to_datetime(dataset._target_datetimes[train_idx[train_mask]])
        val_t = pd.to_datetime(dataset._target_datetimes[val_idx[val_mask]])
        if len(train_t) == 0 or len(val_t) == 0:
            continue
        name = station_label(int(sid))
        print(
            f"  {name:12s} | train: {train_t.min()} -> {train_t.max()} ({len(train_t):4d})"
            f" | val: {val_t.min()} -> {val_t.max()} ({len(val_t):4d})"
        )
    print("-" * 72)


def main() -> None:
    args = parse_args()
    seed = args.seed
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")
    if args.run_id is not None:
        print(f"Run id: {args.run_id}")
    print(f"Output directory: {output_dir.resolve()}")

    dataset = CrowdDataset(DATA_PATH)
    print(f"Valid samples after is_fake filtering: {len(dataset)}")

    train_indices, val_indices = temporal_train_val_indices(dataset, TRAIN_RATIO)
    population_scaler = prepare_dataset_scaler(dataset, train_indices, output_dir)
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    log_temporal_split(dataset, train_indices, val_indices)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CrowdLSTM(num_stations=NUM_STATIONS).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    print(f"\nBest validation loss (Huber, normalized): {best_val_loss:.6f}")
    print(f"Model saved to {BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    overall, per_station, global_hourly, per_station_hourly, _, _ = run_full_evaluation(
        model, val_loader, population_scaler, device, output_dir
    )

    baseline_overall, baseline_per_station = evaluate_baselines(dataset, val_indices)
    log_baseline_results(baseline_overall, baseline_per_station)

    if args.run_id is not None:
        csv_path = save_run_results_csv(
            args.run_id,
            seed,
            overall,
            per_station,
            global_hourly,
            per_station_hourly,
            best_val_loss,
            output_dir,
            baseline_overall=baseline_overall,
            baseline_per_station=baseline_per_station,
        )
        print(f"\nRun metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
