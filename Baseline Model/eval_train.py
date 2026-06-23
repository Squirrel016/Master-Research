"""
Run repeated training experiments, aggregate metrics, and produce summary report/plots.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import TEST_RATIO, TRAIN_RATIO, VAL_RATIO

PROJECT_DIR = Path(__file__).parent
TRAIN_SCRIPT = PROJECT_DIR / "train.py"

# Run count = len(SEEDS); add or remove seeds to change experiment times.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
HOURS = list(range(24))
STATION_NAMES = ["ikebukuro", "nihonbashi", "shibuya", "shinjuku"]
METRIC_COLS_OVERALL = ["mae", "rmse", "mape", "r2"]
METRIC_COLS_HOURLY = ["mae", "rmse", "mape"]

def create_timestamped_result_dir() -> Path:
    """Create Result_MMDD_HHMM folder (24-hour clock, e.g. Result_0527_0603)."""
    folder_name = f"Result_{datetime.now():%m%d_%H%M}"
    result_dir = PROJECT_DIR / folder_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def run_single_experiment(run_id: int, seed: int, result_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--seed",
        str(seed),
        "--run-id",
        str(run_id),
        "--output-dir",
        str(result_dir),
    ]
    print(f"\n{'=' * 72}")
    print(f"Run {run_id}/{len(SEEDS)} | seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 72)
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)
    csv_path = result_dir / f"result_run_{run_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected output not found: {csv_path}")
    return csv_path


def load_all_runs(result_dir: Path) -> pd.DataFrame:
    frames = []
    for run_id in range(1, len(SEEDS) + 1):
        path = result_dir / f"result_run_{run_id}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    for col in ["station_id", "hour", "mae", "rmse", "mape", "r2", "n_samples"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def mean_std(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) == 0:
        return float("nan"), float("nan")
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return float(s.mean()), std


def format_pm(mean: float, std: float, decimals: int = 2, pct: bool = False) -> str:
    if np.isnan(mean):
        return "N/A"
    suffix = "%" if pct else ""
    if np.isnan(std):
        return f"{mean:.{decimals}f}{suffix}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}{suffix}"


def aggregate_overall(df: pd.DataFrame, scope: str = "overall") -> pd.DataFrame:
    sub = df[df["scope"] == scope]
    rows = []
    for metric in METRIC_COLS_OVERALL:
        m, s = mean_std(sub[metric])
        rows.append({"metric": metric, "mean": m, "std": s})
    return pd.DataFrame(rows)


def aggregate_per_station(df: pd.DataFrame, scope: str = "station") -> pd.DataFrame:
    sub = df[df["scope"] == scope]
    rows = []
    for name in STATION_NAMES:
        station_df = sub[sub["station_name"] == name]
        for metric in METRIC_COLS_OVERALL:
            m, s = mean_std(station_df[metric])
            rows.append({"station": name, "metric": metric, "mean": m, "std": s})
    return pd.DataFrame(rows)


def aggregate_hourly(df: pd.DataFrame, scope: str, station_name: str | None = None) -> pd.DataFrame:
    sub = df[df["scope"] == scope].copy()
    if station_name is not None:
        sub = sub[sub["station_name"] == station_name]
    rows = []
    for hour in HOURS:
        hour_df = sub[sub["hour"] == hour]
        row: dict = {"hour": hour}
        for metric in METRIC_COLS_HOURLY:
            m, s = mean_std(hour_df[metric])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hourly_rmse_mae_ratio(df: pd.DataFrame, scope: str, station_name: str | None = None) -> pd.DataFrame:
    sub = df[df["scope"] == scope].copy()
    if station_name is not None:
        sub = sub[sub["station_name"] == station_name]
    rows = []
    for hour in HOURS:
        hour_df = sub[(sub["hour"] == hour) & (sub["n_samples"] > 0) & (sub["mae"] > 0)]
        if hour_df.empty:
            rows.append({"hour": hour, "ratio_mean": np.nan, "ratio_std": np.nan})
        else:
            ratios = hour_df["rmse"] / hour_df["mae"]
            rows.append(
                {
                    "hour": hour,
                    "ratio_mean": float(ratios.mean()),
                    "ratio_std": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _is_overall_row(df: pd.DataFrame) -> pd.Series:
    return df["station_name"].isna() | (df["station_name"].astype(str).str.strip() == "")


def aggregate_model_comparison(df: pd.DataFrame, scope_suffix: str = "") -> pd.DataFrame:
    """Aggregate overall metrics for LSTM and simple baselines across runs."""
    overall_scope = "overall" if not scope_suffix else f"{scope_suffix}_overall"
    baseline_persistence = (
        "baseline_persistence" if not scope_suffix else f"{scope_suffix}_baseline_persistence"
    )
    baseline_seasonal = (
        "baseline_seasonal_naive"
        if not scope_suffix
        else f"{scope_suffix}_baseline_seasonal_naive"
    )
    model_specs = [
        ("CrowdLSTM", overall_scope),
        ("Persistence (t-1)", baseline_persistence),
        ("Seasonal Naive (t-24)", baseline_seasonal),
    ]
    rows: list[dict] = []
    for model_name, scope in model_specs:
        sub = df[(df["scope"] == scope) & _is_overall_row(df)]
        if sub.empty:
            continue
        row: dict = {"model": model_name}
        for metric in METRIC_COLS_OVERALL:
            m, s = mean_std(sub[metric])
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_baseline_per_station(df: pd.DataFrame, scope_suffix: str = "") -> pd.DataFrame:
    """Per-station MAE/RMSE/MAPE for LSTM vs baselines."""
    station_scope = "station" if not scope_suffix else f"{scope_suffix}_station"
    baseline_persistence = (
        "baseline_persistence" if not scope_suffix else f"{scope_suffix}_baseline_persistence"
    )
    baseline_seasonal = (
        "baseline_seasonal_naive"
        if not scope_suffix
        else f"{scope_suffix}_baseline_seasonal_naive"
    )
    rows: list[dict] = []
    specs = [
        ("CrowdLSTM", station_scope),
        ("Persistence (t-1)", baseline_persistence),
        ("Seasonal Naive (t-24)", baseline_seasonal),
    ]
    for station in STATION_NAMES:
        for model_name, scope in specs:
            sub = df[(df["scope"] == scope) & (df["station_name"] == station)]
            if sub.empty:
                continue
            entry: dict = {"station": station, "model": model_name}
            for metric in METRIC_COLS_OVERALL:
                m, s = mean_std(sub[metric])
                entry[f"{metric}_mean"] = m
                entry[f"{metric}_std"] = s
            rows.append(entry)
    return pd.DataFrame(rows)


def build_report(
    overall_agg: pd.DataFrame,
    station_agg: pd.DataFrame,
    global_hourly_agg: pd.DataFrame,
    station_hourly_aggs: dict[str, pd.DataFrame],
    model_comparison_agg: pd.DataFrame,
    baseline_station_agg: pd.DataFrame,
    val_overall_agg: pd.DataFrame,
    val_model_comparison_agg: pd.DataFrame,
    result_dir: Path,
) -> str:
    lines: list[str] = []
    w = 78

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * w)

    lines.append("EXPERIMENT REPORT (CrowdLSTM + Simple Baselines)")
    lines.append(f"Runs: {len(SEEDS)} | Seeds: {', '.join(map(str, SEEDS))}")
    lines.append(
        f"Temporal split: train {TRAIN_RATIO:.0%} | val {VAL_RATIO:.0%} | test {TEST_RATIO:.0%}"
    )
    lines.append(f"Output directory: {result_dir.resolve()}")
    lines.append("=" * w)

    section("1. MODEL COMPARISON (Mean +/- Std, held-out test set)")
    header = (
        f"  {'Model':<22} | {'MAE':>18} | {'RMSE':>18} | "
        f"{'MAPE':>18} | {'R^2':>14}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for _, row in model_comparison_agg.iterrows():
        lines.append(
            f"  {row['model']:<22} | "
            f"{format_pm(row['mae_mean'], row['mae_std']):>18} | "
            f"{format_pm(row['rmse_mean'], row['rmse_std']):>18} | "
            f"{format_pm(row['mape_mean'], row['mape_std'], pct=True):>18} | "
            f"{format_pm(row['r2_mean'], row['r2_std'], decimals=4):>14}"
        )

    section("2. CROWDLSTM TEST SET METRICS (Mean +/- Std)")
    for _, row in overall_agg.iterrows():
        pct = row["metric"] == "mape"
        dec = 4 if row["metric"] == "r2" else 2
        lines.append(
            f"  {row['metric'].upper():6s}: "
            f"{format_pm(row['mean'], row['std'], decimals=dec, pct=pct)}"
        )

    section("3. CROWDLSTM PER-STATION TEST METRICS (Mean +/- Std)")
    for station in STATION_NAMES:
        lines.append(f"\n  [{station}]")
        sub = station_agg[station_agg["station"] == station]
        for _, row in sub.iterrows():
            pct = row["metric"] == "mape"
            dec = 4 if row["metric"] == "r2" else 2
            lines.append(
                f"    {row['metric'].upper():6s}: "
                f"{format_pm(row['mean'], row['std'], decimals=dec, pct=pct)}"
            )

    section("4. BASELINE vs LSTM BY STATION — TEST SET (MAE / RMSE / MAPE, Mean +/- Std)")
    for station in STATION_NAMES:
        sub = baseline_station_agg[baseline_station_agg["station"] == station]
        if sub.empty:
            continue
        lines.append(f"\n  [{station}]")
        header = f"    {'Model':<22} | {'MAE':>18} | {'RMSE':>18} | {'MAPE':>18}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for _, row in sub.iterrows():
            lines.append(
                f"    {row['model']:<22} | "
                f"{format_pm(row['mae_mean'], row['mae_std']):>18} | "
                f"{format_pm(row['rmse_mean'], row['rmse_std']):>18} | "
                f"{format_pm(row['mape_mean'], row['mape_std'], pct=True):>18}"
            )

    section("5. GLOBAL HOURLY METRICS - CrowdLSTM TEST SET (Mean +/- Std)")
    header = f"{'Hour':>4} | {'MAE':>18} | {'RMSE':>18} | {'MAPE':>18}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in global_hourly_agg.iterrows():
        h = int(row["hour"])
        lines.append(
            f"{h:04d} | "
            f"{format_pm(row['mae_mean'], row['mae_std']):>18} | "
            f"{format_pm(row['rmse_mean'], row['rmse_std']):>18} | "
            f"{format_pm(row['mape_mean'], row['mape_std'], pct=True):>18}"
        )

    section("6. PER-STATION HOURLY METRICS - CrowdLSTM TEST SET (Mean +/- Std)")
    for station in STATION_NAMES:
        lines.append(f"\n  [{station}]")
        hourly = station_hourly_aggs[station]
        header = f"    {'Hr':>4} | {'MAE':>18} | {'RMSE':>18} | {'MAPE':>18}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))
        for _, row in hourly.iterrows():
            h = int(row["hour"])
            lines.append(
                f"    {h:04d} | "
                f"{format_pm(row['mae_mean'], row['mae_std']):>18} | "
                f"{format_pm(row['rmse_mean'], row['rmse_std']):>18} | "
                f"{format_pm(row['mape_mean'], row['mape_std'], pct=True):>18}"
            )

    section("7. VALIDATION SET SUMMARY (model selection, Mean +/- Std)")
    lines.append("  CrowdLSTM overall:")
    for _, row in val_overall_agg.iterrows():
        pct = row["metric"] == "mape"
        dec = 4 if row["metric"] == "r2" else 2
        lines.append(
            f"    {row['metric'].upper():6s}: "
            f"{format_pm(row['mean'], row['std'], decimals=dec, pct=pct)}"
        )
    lines.append("")
    header = (
        f"  {'Model':<22} | {'MAE':>18} | {'RMSE':>18} | "
        f"{'MAPE':>18} | {'R^2':>14}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for _, row in val_model_comparison_agg.iterrows():
        lines.append(
            f"  {row['model']:<22} | "
            f"{format_pm(row['mae_mean'], row['mae_std']):>18} | "
            f"{format_pm(row['rmse_mean'], row['rmse_std']):>18} | "
            f"{format_pm(row['mape_mean'], row['mape_std'], pct=True):>18} | "
            f"{format_pm(row['r2_mean'], row['r2_std'], decimals=4):>14}"
        )

    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


def plot_with_error_band(
    ax: plt.Axes,
    hours: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
) -> None:
    valid = ~np.isnan(mean)
    h = hours[valid]
    m = mean[valid]
    s = np.nan_to_num(std[valid], nan=0.0)
    ax.plot(h, m, "o-", linewidth=2.2, markersize=5, label=label, color=color)
    ax.fill_between(h, m - s, m + s, color=color, alpha=0.18, linewidth=0)


def plot_hourly_mape(
    global_hourly_agg: pd.DataFrame,
    station_hourly_aggs: dict[str, pd.DataFrame],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {
        "overall": "#222222",
        "ikebukuro": "#1f77b4",
        "nihonbashi": "#ff7f0e",
        "shibuya": "#2ca02c",
        "shinjuku": "#d62728",
    }

    plot_with_error_band(
        ax,
        global_hourly_agg["hour"].to_numpy(),
        global_hourly_agg["mape_mean"].to_numpy(),
        global_hourly_agg["mape_std"].to_numpy(),
        "Overall (mean)",
        colors["overall"],
    )
    for station in STATION_NAMES:
        df = station_hourly_aggs[station]
        plot_with_error_band(
            ax,
            df["hour"].to_numpy(),
            df["mape_mean"].to_numpy(),
            df["mape_std"].to_numpy(),
            station,
            colors[station],
        )

    ax.set_xticks(HOURS)
    ax.set_xlabel("Hour of day", fontsize=13)
    ax.set_ylabel("MAPE (%)", fontsize=13)
    ax.set_title("Hourly MAPE: Overall vs Stations (test set, Mean +/- Std over runs)", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_stability_index(
    global_ratio_agg: pd.DataFrame,
    station_ratio_aggs: dict[str, pd.DataFrame],
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {
        "overall": "#222222",
        "ikebukuro": "#1f77b4",
        "nihonbashi": "#ff7f0e",
        "shibuya": "#2ca02c",
        "shinjuku": "#d62728",
    }

    plot_with_error_band(
        ax,
        global_ratio_agg["hour"].to_numpy(),
        global_ratio_agg["ratio_mean"].to_numpy(),
        global_ratio_agg["ratio_std"].to_numpy(),
        "Overall (mean)",
        colors["overall"],
    )
    for station in STATION_NAMES:
        df = station_ratio_aggs[station]
        plot_with_error_band(
            ax,
            df["hour"].to_numpy(),
            df["ratio_mean"].to_numpy(),
            df["ratio_std"].to_numpy(),
            station,
            colors[station],
        )

    ax.set_xticks(HOURS)
    ax.set_xlabel("Hour of day", fontsize=13)
    ax.set_ylabel("RMSE / MAE", fontsize=13)
    ax.set_title("Stability Index (RMSE/MAE) by Hour (Mean +/- Std over runs)", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_aggregate_csv(
    overall_agg: pd.DataFrame,
    station_agg: pd.DataFrame,
    global_hourly_agg: pd.DataFrame,
    station_hourly_aggs: dict[str, pd.DataFrame],
    model_comparison_agg: pd.DataFrame,
    baseline_station_agg: pd.DataFrame,
    path: Path,
    val_overall_agg: pd.DataFrame | None = None,
    val_model_comparison_agg: pd.DataFrame | None = None,
) -> None:
    parts = [
        overall_agg.assign(section="test_overall"),
        station_agg.assign(section="test_per_station"),
        global_hourly_agg.assign(section="test_global_hourly"),
        model_comparison_agg.assign(section="test_model_comparison"),
        baseline_station_agg.assign(section="test_baseline_by_station"),
    ]
    if val_overall_agg is not None:
        parts.append(val_overall_agg.assign(section="val_overall"))
    if val_model_comparison_agg is not None:
        parts.append(val_model_comparison_agg.assign(section="val_model_comparison"))
    for station, hourly in station_hourly_aggs.items():
        parts.append(hourly.assign(section=f"hourly_{station}"))
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    result_dir = create_timestamped_result_dir()
    hourly_mape_plot = result_dir / "hourly_mape_comparison.png"
    stability_plot = result_dir / "stability_index_rmse_mae.png"
    final_report = result_dir / "final_report.txt"
    aggregate_csv = result_dir / "aggregate_summary.csv"

    print(f"Results will be saved to: {result_dir.resolve()}")

    for run_id, seed in enumerate(SEEDS, start=1):
        run_single_experiment(run_id, seed, result_dir)

    runs_df = load_all_runs(result_dir)

    overall_agg = aggregate_overall(runs_df, scope="overall")
    station_agg = aggregate_per_station(runs_df, scope="station")
    global_hourly_agg = aggregate_hourly(runs_df, "global_hourly")
    station_hourly_aggs = {
        name: aggregate_hourly(runs_df, "station_hourly", station_name=name) for name in STATION_NAMES
    }
    global_ratio_agg = compute_hourly_rmse_mae_ratio(runs_df, "global_hourly")
    station_ratio_aggs = {
        name: compute_hourly_rmse_mae_ratio(runs_df, "station_hourly", station_name=name)
        for name in STATION_NAMES
    }

    model_comparison_agg = aggregate_model_comparison(runs_df)
    baseline_station_agg = aggregate_baseline_per_station(runs_df)
    val_overall_agg = aggregate_overall(runs_df, scope="val_overall")
    val_model_comparison_agg = aggregate_model_comparison(runs_df, scope_suffix="val")

    report = build_report(
        overall_agg,
        station_agg,
        global_hourly_agg,
        station_hourly_aggs,
        model_comparison_agg,
        baseline_station_agg,
        val_overall_agg,
        val_model_comparison_agg,
        result_dir,
    )
    print("\n" + report)
    final_report.write_text(report, encoding="utf-8")
    save_aggregate_csv(
        overall_agg,
        station_agg,
        global_hourly_agg,
        station_hourly_aggs,
        model_comparison_agg,
        baseline_station_agg,
        aggregate_csv,
        val_overall_agg=val_overall_agg,
        val_model_comparison_agg=val_model_comparison_agg,
    )

    plot_hourly_mape(global_hourly_agg, station_hourly_aggs, hourly_mape_plot)
    plot_stability_index(global_ratio_agg, station_ratio_aggs, stability_plot)

    print("\nGenerated outputs:")
    for run_id in range(1, len(SEEDS) + 1):
        print(f"  {result_dir / f'result_run_{run_id}.csv'}")
    print(f"  {final_report}")
    print(f"  {aggregate_csv}")
    print(f"  {hourly_mape_plot}")
    print(f"  {stability_plot}")


if __name__ == "__main__":
    main()
