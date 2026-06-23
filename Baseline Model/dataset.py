from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from scaling import transform_population

WINDOW_SIZE = 24
SEQ_FEATURE_COLS = ["population", "hour_of_day", "day_of_week"]
TARGET_COL = "population"
DEFAULT_DATA_PATH = Path(__file__).parent / "processed_data.csv"


class CrowdDataset(Dataset):
    """
    Sliding-window dataset: past 24 hours -> next-hour population.

    Drops any window that contains is_fake rows (long-gap interpolation).
    """

    def __init__(self, csv_path: str | Path = DEFAULT_DATA_PATH) -> None:
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])

        seq_list: list[np.ndarray] = []
        station_ids: list[int] = []
        current_hours: list[np.ndarray] = []
        targets_list: list[float] = []
        datetimes_list: list[np.ndarray] = []
        was_missing_list: list[np.ndarray] = []

        for _, station_df in df.groupby("station_name", sort=False):
            station_df = station_df.sort_values("datetime").reset_index(drop=True)
            seq_features = station_df[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32)
            is_fake = station_df["is_fake"].to_numpy()
            was_missing = station_df["was_missing"].to_numpy()
            station_id = int(station_df["station_name"].iloc[0])
            targets = station_df[TARGET_COL].to_numpy(dtype=np.float32)
            datetimes = station_df["datetime"].to_numpy()

            n_samples = len(station_df) - WINDOW_SIZE
            if n_samples <= 0:
                continue

            for start in range(n_samples):
                end = start + WINDOW_SIZE
                window_slice = slice(start, end + 1)

                if is_fake[window_slice].any():
                    continue

                seq_list.append(seq_features[start:end])
                station_ids.append(station_id)
                current_hours.append(seq_features[end, 1:3])
                targets_list.append(targets[end])
                datetimes_list.append(datetimes[window_slice])
                was_missing_list.append(was_missing[window_slice])

        if not seq_list:
            raise ValueError("No valid samples after filtering is_fake windows.")

        self._seq = np.stack(seq_list)
        self._station_ids = np.array(station_ids, dtype=np.int64)
        self._current_hours = np.stack(current_hours)
        self._targets = np.array(targets_list, dtype=np.float32)
        self._datetimes = datetimes_list
        self._was_missing = was_missing_list
        self._target_datetimes = np.array([d[-1] for d in datetimes_list])
        self._population_scaler: MinMaxScaler | None = None

    def set_population_scaler(self, scaler: MinMaxScaler) -> None:
        self._population_scaler = scaler

    def _scale_population(self, seq: np.ndarray, target: float) -> tuple[np.ndarray, float]:
        if self._population_scaler is None:
            raise RuntimeError(
                "Population scaler not set. Fit on train indices and call set_population_scaler()."
            )
        scaled_seq = seq.copy()
        scaled_seq[:, 0] = transform_population(seq[:, 0], self._population_scaler)
        scaled_target = float(transform_population(np.array([target]), self._population_scaler)[0])
        return scaled_seq, scaled_target

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self._seq[idx]
        target = float(self._targets[idx])
        seq, target = self._scale_population(seq, target)
        return (
            torch.from_numpy(seq),
            torch.tensor(self._station_ids[idx], dtype=torch.long),
            torch.from_numpy(self._current_hours[idx]),
            torch.tensor(target, dtype=torch.float32),
        )

    def get_datetimes(self, idx: int) -> np.ndarray:
        return self._datetimes[idx]

    def get_was_missing(self, idx: int) -> np.ndarray:
        return self._was_missing[idx]

    def get_station_id(self, idx: int) -> int:
        return int(self._station_ids[idx])

    def get_target_hour(self, idx: int) -> int:
        return int(self._current_hours[idx][0])

    def get_target_datetime(self, idx: int) -> np.datetime64:
        return self._target_datetimes[idx]


def temporal_train_val_indices(
    dataset: CrowdDataset,
    train_ratio: float = 0.8,
) -> tuple[list[int], list[int]]:
    """
    Split samples by time within each station: earlier windows -> train, later -> val.
    """
    train_indices: list[int] = []
    val_indices: list[int] = []

    for sid in np.unique(dataset._station_ids):
        station_idx = np.where(dataset._station_ids == sid)[0]
        order = station_idx[np.argsort(dataset._target_datetimes[station_idx])]
        split_at = int(len(order) * train_ratio)

        if split_at <= 0 or split_at >= len(order):
            raise ValueError(
                f"Invalid temporal split for station {sid}: "
                f"{len(order)} samples, train_ratio={train_ratio}"
            )

        train_indices.extend(order[:split_at].tolist())
        val_indices.extend(order[split_at:].tolist())

    return train_indices, val_indices


def create_dataloader(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = CrowdDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
