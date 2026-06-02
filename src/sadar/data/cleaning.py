from __future__ import annotations

import numpy as np
import pandas as pd


def drop_short_flights(df: pd.DataFrame, min_points: int) -> pd.DataFrame:
    sizes = df["flight_id"].map(df.groupby("flight_id").size())
    return df[sizes >= min_points].reset_index(drop=True)


def is_go_around(
    group: pd.DataFrame,
    near_runway_m: float = 5000.0,
    descend_vertrate: float = -1.0,
    climb_vertrate: float = 2.0,
    min_climb_points: int = 3,
) -> bool:
    g = group.sort_values("time")
    near = g["dist_to_runway_m"] < near_runway_m
    if not near.any():
        return False
    idx = np.where(near.to_numpy())[0]
    before = g.iloc[: idx[-1] + 1]
    after = g.iloc[idx[0] :]
    descended = (before["vertrate"].fillna(0) < descend_vertrate).any()
    climbed_after = (after["vertrate"].fillna(0) > climb_vertrate).sum() >= min_climb_points
    phase_climb = after["flight_phase"].isin(["climb", "takeoff"]).any()
    return bool(descended and climbed_after and phase_climb)


def has_airborne_gap(group: pd.DataFrame, threshold_seconds: int) -> bool:
    air = group.loc[group["onground"].eq(False), "time"]
    if len(air) < 2:
        return False
    return bool((air.sort_values().diff().dropna() > threshold_seconds).any())


def go_around_flight_ids(df: pd.DataFrame, **thresholds) -> set:
    flags = df.groupby("flight_id").apply(
        lambda g: is_go_around(g, **thresholds), include_groups=False
    )
    return set(flags[flags].index)


def emergency_flight_ids(df: pd.DataFrame, codes: list[int]) -> set:
    return set(df.loc[df["squawk"].isin(codes), "flight_id"].unique())


def remove_emergency_flights(df: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    flagged = emergency_flight_ids(df, codes)
    return df[~df["flight_id"].isin(flagged)].reset_index(drop=True)


def remove_go_around_flights(df: pd.DataFrame, **thresholds) -> pd.DataFrame:
    flagged = go_around_flight_ids(df, **thresholds)
    return df[~df["flight_id"].isin(flagged)].reset_index(drop=True)


def handle_nulls(df: pd.DataFrame, columns: list[str], max_missing_fraction: float) -> pd.DataFrame:
    worst = df.groupby("flight_id")[columns].apply(lambda g: g.isna().mean().max())
    keep = worst[worst <= max_missing_fraction].index
    out = df[df["flight_id"].isin(keep)].copy()
    out[columns] = out.groupby("flight_id")[columns].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    return out.reset_index(drop=True)
