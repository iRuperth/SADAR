from __future__ import annotations

import os

import numpy as np
import torch
from pyproj import Transformer

from sadar.data.pipeline import load_config
from sadar.data.scaling import StandardScaler3D
from sadar.eval import synthetic
from sadar.models.training import (
    load_autoencoder,
    per_step_error,
    reconstruction_error,
    resolve_device,
)


class ConformanceService:
    def __init__(self, serve_config_path: str) -> None:
        serve = load_config(serve_config_path)
        config = load_config(serve["eval_config"])

        self.device = resolve_device(config["model"]["device"])
        processed_dir = config["data"]["processed_dir"]

        preprocessing = load_config(config["preprocessing_config"])
        self.feature_columns = preprocessing["features"]["columns"]
        self.indices = {name: i for i, name in enumerate(self.feature_columns)}
        self.step_seconds = preprocessing["resample"]["seconds"]

        self.scaler = StandardScaler3D.load(os.path.join(processed_dir, "scaler.npz"))
        self.test = np.load(os.path.join(processed_dir, "test.npy"))
        val = np.load(os.path.join(processed_dir, "val.npy"))

        self.model = load_autoencoder(serve["checkpoint"], self.device)
        self.model.eval()

        runway = preprocessing["runway"]
        forward = Transformer.from_crs(
            "EPSG:4326", f"EPSG:{runway['projection_epsg']}", always_xy=True
        )
        self.ref_x, self.ref_y = forward.transform(runway["ref_lon"], runway["ref_lat"])
        self.inverse = Transformer.from_crs(
            f"EPSG:{runway['projection_epsg']}", "EPSG:4326", always_xy=True
        )

        self.window_scores = reconstruction_error(self.model, self.test, self.device)
        val_scores = reconstruction_error(self.model, val, self.device)
        self.threshold = float(np.percentile(val_scores, config["threshold"]["val_percentile"]))
        val_step = per_step_error(self.model, val, self.device)
        self.step_threshold = float(
            np.percentile(val_step, config["latency"]["normal_step_percentile"])
        )

    def _unscale(self, scaled: np.ndarray) -> np.ndarray:
        return synthetic.unscale(scaled, self.scaler.mean_, self.scaler.std_)

    def _rescale(self, unscaled: np.ndarray) -> np.ndarray:
        return synthetic.rescale(unscaled, self.scaler.mean_, self.scaler.std_)

    def _to_path(self, unscaled_window: np.ndarray) -> list[dict]:
        x = unscaled_window[:, self.indices["x_rel"]] + self.ref_x
        y = unscaled_window[:, self.indices["y_rel"]] + self.ref_y
        lon, lat = self.inverse.transform(x, y)
        altitude = unscaled_window[:, self.indices["baroaltitude"]]
        return [
            {"lat": float(a), "lon": float(b), "alt": float(c), "t": i * self.step_seconds}
            for i, (a, b, c) in enumerate(zip(lat, lon, altitude))
        ]

    def _reconstruct(self, scaled_window: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(scaled_window[None].astype(np.float32)).to(self.device)
            return self.model(tensor)[0].cpu().numpy()

    def _latency(self, step_scores: np.ndarray, onset_idx: int) -> float | None:
        over = np.where(step_scores[onset_idx:] >= self.step_threshold)[0]
        if len(over) == 0:
            return None
        return float(over[0] * self.step_seconds)

    def list_flights(self, limit: int) -> list[dict]:
        order = np.argsort(self.window_scores)[::-1][:limit]
        return [
            {
                "id": int(i),
                "score": float(self.window_scores[i]),
                "anomalous": bool(self.window_scores[i] >= self.threshold),
            }
            for i in order
        ]

    def flight_detail(self, flight_id: int) -> dict:
        scaled = self.test[flight_id]
        unscaled = self._unscale(scaled[None])[0]
        reconstructed = self._unscale(self._reconstruct(scaled)[None])[0]
        step_scores = per_step_error(self.model, scaled[None], self.device)[0]
        return {
            "id": flight_id,
            "path": self._to_path(unscaled),
            "reconstructed": self._to_path(reconstructed),
            "scores": [float(s) for s in step_scores],
            "window_score": float(self.window_scores[flight_id]),
            "threshold": self.threshold,
            "step_threshold": self.step_threshold,
        }

    def simulate(self, flight_id: int, kind: str, magnitude: float, onset: float) -> dict:
        unscaled = self._unscale(self.test[flight_id : flight_id + 1])
        rng = np.random.default_rng(0)
        idx = self.indices

        if kind == "route_deviation":
            perturbed, _ = synthetic.route_deviation(
                unscaled, idx["x_rel"], idx["y_rel"], magnitude, onset, rng
            )
        elif kind == "altitude":
            perturbed, _ = synthetic.altitude_anomaly(
                unscaled, idx["baroaltitude"], magnitude, onset, rng,
                idx.get("vertrate"), self.step_seconds,
            )
        elif kind == "speed":
            perturbed, _ = synthetic.speed_anomaly(unscaled, idx["velocity"], magnitude, onset, rng)
        elif kind == "holding":
            perturbed, _ = synthetic.holding_pattern(
                unscaled, idx["x_rel"], idx["y_rel"], idx["sin_hdg"], idx["cos_hdg"],
                idx["velocity"], onset, magnitude, rng, idx.get("vertrate"), self.step_seconds,
            )
        elif kind == "freeze":
            perturbed, _ = synthetic.sensor_freeze(unscaled, onset, rng)
        else:
            raise ValueError(f"unknown anomaly kind: {kind}")

        scaled = self._rescale(perturbed)
        step_scores = per_step_error(self.model, scaled, self.device)[0]
        onset_idx = synthetic.onset_index(self.test.shape[1], onset)
        return {
            "id": flight_id,
            "kind": kind,
            "path": self._to_path(perturbed[0]),
            "scores": [float(s) for s in step_scores],
            "window_score": float(reconstruction_error(self.model, scaled, self.device)[0]),
            "threshold": self.threshold,
            "step_threshold": self.step_threshold,
            "onset_index": onset_idx,
            "latency_seconds": self._latency(step_scores, onset_idx),
        }
