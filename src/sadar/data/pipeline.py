from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import yaml

from sadar.data import cleaning, features, scaling, splitting, windowing
from sadar.data import io as data_io


@dataclass
class PipelineArtifacts:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    anomalies: np.ndarray
    scaler: scaling.StandardScaler3D


_ENV_PATTERN = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


def _resolve_env(value: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1).strip()
        default = match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_PATTERN.sub(replace, value)


def _resolve(node):
    if isinstance(node, dict):
        return {key: _resolve(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve(value) for value in node]
    if isinstance(node, str):
        return _resolve_env(node)
    return node


def load_config(path: str) -> dict:
    with open(path) as handle:
        return _resolve(yaml.safe_load(handle))


def _split_normal_anomalies(df, config):
    cleaning_cfg = config["cleaning"]
    df = cleaning.drop_short_flights(df, cleaning_cfg["min_points_per_flight"])

    anomaly_ids = set()
    if cleaning_cfg["remove_emergency_squawks"]:
        anomaly_ids |= cleaning.emergency_flight_ids(df, cleaning_cfg["emergency_squawks"])
    if cleaning_cfg["remove_go_arounds"]:
        anomaly_ids |= cleaning.go_around_flight_ids(df, **cleaning_cfg["go_around"])

    columns = cleaning_cfg["interpolate_columns"]
    max_missing = cleaning_cfg["max_missing_fraction"]
    is_anomaly = df["flight_id"].isin(anomaly_ids)
    normal = cleaning.handle_nulls(
        df[~is_anomaly].reset_index(drop=True), columns, max_missing
    )
    anomalies = cleaning.handle_nulls(
        df[is_anomaly].reset_index(drop=True), columns, max_missing
    )
    return normal, anomalies


def _add_features(df, config):
    runway = config["runway"]
    df = features.add_runway_relative_coords(
        df, runway["ref_lat"], runway["ref_lon"], runway["projection_epsg"]
    )
    return features.add_heading_sincos(df)


def _window_split(df, config) -> np.ndarray:
    feature_columns = config["features"]["columns"]
    segmented = windowing.add_segment_id(df, config["cleaning"]["max_gap_seconds"])
    resampled = windowing.resample_uniform(
        segmented, config["resample"]["seconds"], feature_columns
    )
    return windowing.make_windows(
        resampled,
        config["windowing"]["length"],
        config["windowing"]["stride"],
        feature_columns,
    )


def run(config: dict) -> PipelineArtifacts:
    df = data_io.load_raw(config["data"]["raw_dir"], config["data"]["file_glob"])
    normal_df, anomaly_df = _split_normal_anomalies(df, config)
    normal_df = _add_features(normal_df, config)
    anomaly_df = _add_features(anomaly_df, config)

    split_cfg = config["split"]
    train_df, val_df, test_df = splitting.split_by_year(
        normal_df,
        split_cfg["train_years"],
        split_cfg["eval_years"],
        split_cfg["val_fraction"],
        config["seed"],
    )

    train_windows = _window_split(train_df, config)
    val_windows = _window_split(val_df, config)
    test_windows = _window_split(test_df, config)
    anomaly_windows = _window_split(anomaly_df, config)

    scaler = scaling.StandardScaler3D().fit(train_windows)
    return PipelineArtifacts(
        train=scaler.transform(train_windows),
        val=scaler.transform(val_windows),
        test=scaler.transform(test_windows),
        anomalies=scaler.transform(anomaly_windows),
        scaler=scaler,
    )


def save_artifacts(artifacts: PipelineArtifacts, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "train.npy"), artifacts.train)
    np.save(os.path.join(output_dir, "val.npy"), artifacts.val)
    np.save(os.path.join(output_dir, "test.npy"), artifacts.test)
    np.save(os.path.join(output_dir, "anomalies.npy"), artifacts.anomalies)
    artifacts.scaler.save(os.path.join(output_dir, "scaler.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready trajectory tensors.")
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    artifacts = run(config)
    save_artifacts(artifacts, config["output"]["dir"])

    print(
        f"train {artifacts.train.shape} "
        f"val {artifacts.val.shape} "
        f"test {artifacts.test.shape} "
        f"anomalies {artifacts.anomalies.shape} "
        f"-> {config['output']['dir']}"
    )


if __name__ == "__main__":
    main()
