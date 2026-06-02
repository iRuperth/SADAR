from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


def add_runway_relative_coords(
    df: pd.DataFrame, ref_lat: float, ref_lon: float, projection_epsg: int
) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{projection_epsg}", always_xy=True)
    ref_x, ref_y = transformer.transform(ref_lon, ref_lat)
    x, y = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df.assign(x_rel=x - ref_x, y_rel=y - ref_y)


def add_heading_sincos(df: pd.DataFrame) -> pd.DataFrame:
    radians = np.deg2rad(df["heading"].to_numpy())
    return df.assign(sin_hdg=np.sin(radians), cos_hdg=np.cos(radians))


def select_feature_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns]
