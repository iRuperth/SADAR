from __future__ import annotations

import glob
import os

import pandas as pd


def list_parquet_files(raw_dir: str, file_glob: str) -> list[str]:
    return sorted(glob.glob(os.path.join(raw_dir, file_glob)))


def add_source_date(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    source_date = os.path.basename(file_path).split("__")[0].replace("lemd_", "")
    return df.assign(source_date=source_date)


def load_raw(raw_dir: str, file_glob: str) -> pd.DataFrame:
    files = list_parquet_files(raw_dir, file_glob)
    if not files:
        raise FileNotFoundError(f"no parquet files matching {file_glob!r} in {raw_dir!r}")
    frames = [add_source_date(pd.read_parquet(path), path) for path in files]
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(["flight_id", "time"]).reset_index(drop=True)
