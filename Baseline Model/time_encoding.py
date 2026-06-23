"""Cyclic sin/cos encodings for hour-of-day and day-of-week."""

import numpy as np

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
TIME_FEATURE_DIM = 4  # hour_sin, hour_cos, dow_sin, dow_cos


def encode_hour(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * np.asarray(hour, dtype=float) / HOURS_PER_DAY
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def encode_day_of_week(day_of_week: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * np.asarray(day_of_week, dtype=float) / DAYS_PER_WEEK
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def encode_time_features(hour: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
    """Stack hour and day-of-week sin/cos encodings along the last axis."""
    hour_sin, hour_cos = encode_hour(hour)
    dow_sin, dow_cos = encode_day_of_week(day_of_week)
    return np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=-1)
