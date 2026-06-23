"""Preprocess station population data from data.csv."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path(__file__).parent / "data.csv"
OUTPUT_PATH = Path(__file__).parent / "processed_data.csv"
MAX_GAP_HOURS = 3


def merge_datetime(df: pd.DataFrame) -> pd.Series:
    """Merge date and hour; hour 24 means 00:00 on the next day."""
    base = pd.to_datetime(df["date"], format="%Y/%m/%d")
    hour = df["hour"]
    return base + pd.to_timedelta(hour.where(hour != 24, 0), unit="h") + pd.to_timedelta(
        (hour == 24).astype(int), unit="D"
    )


def assign_is_fake(was_missing: pd.Series, max_gap_hours: int = MAX_GAP_HOURS) -> pd.Series:
    """
    Mark interpolated rows from long gaps as fake.

    Rows originally present -> is_fake=0.
    Interpolated rows in gaps of <= max_gap_hours -> is_fake=0.
    Interpolated rows in gaps of > max_gap_hours -> is_fake=1.
    """
    is_fake = pd.Series(0, index=was_missing.index, dtype=int)
    if not was_missing.any():
        return is_fake

    missing = was_missing.astype(bool)
    run_id = (missing != missing.shift(fill_value=False)).cumsum()
    for _, group in missing.groupby(run_id):
        if group.iloc[0] and len(group) > max_gap_hours:
            is_fake.loc[group.index] = 1
    return is_fake


def main() -> None:
    df = pd.read_csv(DATA_PATH, na_values=["NA"])

    df["datetime"] = merge_datetime(df)
    df = df.sort_values(["station_name", "datetime"]).reset_index(drop=True)

    df["was_missing"] = df["population"].isna().astype(int)

    df["population"] = df.groupby("station_name")["population"].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    df["is_fake"] = df.groupby("station_name")["was_missing"].transform(assign_is_fake)

    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["station_name"] = LabelEncoder().fit_transform(df["station_name"])

    # Keep raw (interpolated) population; scaling is fit on train split in train.py.

    output_cols = [
        "datetime",
        "station_name",
        "population",
        "hour_of_day",
        "day_of_week",
        "is_fake",
        "was_missing",
    ]
    df[output_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    fake_rows = int(df["is_fake"].sum())
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Rows marked is_fake=1 (long-gap interpolation): {fake_rows}")


if __name__ == "__main__":
    main()
