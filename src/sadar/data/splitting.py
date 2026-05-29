from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_year(
    df: pd.DataFrame,
    train_years: list[int],
    eval_years: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year = df["source_date"].str[:4].astype(int)
    train = df[year.isin(train_years)].reset_index(drop=True)

    eval_pool = df[year.isin(eval_years)]
    eval_flight_ids = eval_pool["flight_id"].unique()
    shuffled = np.random.default_rng(seed).permutation(eval_flight_ids)
    n_val = int(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val])

    val = eval_pool[eval_pool["flight_id"].isin(val_ids)].reset_index(drop=True)
    test = eval_pool[~eval_pool["flight_id"].isin(val_ids)].reset_index(drop=True)
    return train, val, test
