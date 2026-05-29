from __future__ import annotations

import numpy as np
import pandas as pd


def add_segment_id(df: pd.DataFrame, max_gap_seconds: int) -> pd.DataFrame:
    out = df.sort_values(["flight_id", "time"]).reset_index(drop=True)
    gap = out.groupby("flight_id")["time"].diff()
    breaks = (gap > max_gap_seconds).fillna(False)
    seg_num = breaks.groupby(out["flight_id"]).cumsum().astype(int)
    out["segment_id"] = out["flight_id"].astype(str) + "_seg" + seg_num.astype(str)
    return out


def resample_uniform(
    df: pd.DataFrame, seconds: int, value_columns: list[str], group_column: str = "segment_id"
) -> pd.DataFrame:
    rule = f"{seconds}s"
    parts = []
    for group_id, group in df.groupby(group_column, sort=False):
        group = group.sort_values("time").drop_duplicates("time")
        index = pd.to_datetime(group["time"].to_numpy(), unit="s")
        grid = (
            group.set_index(index)[value_columns]
            .resample(rule)
            .mean()
            .interpolate(limit_direction="both")
        )
        grid.insert(0, group_column, group_id)
        parts.append(grid.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)


def make_windows(
    df: pd.DataFrame,
    length: int,
    stride: int,
    feature_columns: list[str],
    group_column: str = "segment_id",
) -> np.ndarray:
    windows = []
    for _, group in df.groupby(group_column, sort=False):
        track = group[feature_columns].to_numpy(dtype="float32")
        for start in range(0, len(track) - length + 1, stride):
            windows.append(track[start : start + length])
    if not windows:
        return np.empty((0, length, len(feature_columns)), dtype="float32")
    return np.stack(windows)
