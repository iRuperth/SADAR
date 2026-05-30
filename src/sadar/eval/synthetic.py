from __future__ import annotations

import numpy as np

KINDS = ["route_deviation", "altitude", "speed", "holding", "freeze"]


def feature_indices(feature_columns: list[str], names: list[str]) -> list[int]:
    return [feature_columns.index(name) for name in names]


def unscale(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return windows.astype(np.float64) * std + mean


def rescale(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)


def onset_index(length: int, onset_fraction: float) -> int:
    return int(np.clip(round(length * onset_fraction), 0, length - 1))


def _ramp(length: int, start: int) -> np.ndarray:
    ramp = np.zeros(length, dtype=np.float64)
    if start >= length - 1:
        ramp[length - 1 :] = 1.0
    else:
        ramp[start:] = np.linspace(0.0, 1.0, length - start)
    return ramp


def _mask_from(length: int, start: int, count: int) -> np.ndarray:
    mask = np.zeros((count, length), dtype=bool)
    mask[:, start:] = True
    return mask


def route_deviation(
    batch: np.ndarray,
    x_index: int,
    y_index: int,
    magnitude_m: float,
    onset_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    out = batch.copy()
    n, length, _ = out.shape
    start = onset_index(length, onset_fraction)
    ramp = _ramp(length, start)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    out[:, :, x_index] += (np.cos(angle) * magnitude_m)[:, None] * ramp[None, :]
    out[:, :, y_index] += (np.sin(angle) * magnitude_m)[:, None] * ramp[None, :]
    return out, _mask_from(length, start, n)


def altitude_anomaly(
    batch: np.ndarray,
    altitude_index: int,
    magnitude_m: float,
    onset_fraction: float,
    rng: np.random.Generator,
    vertrate_index: int | None = None,
    step_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    out = batch.copy()
    n, length, _ = out.shape
    start = onset_index(length, onset_fraction)
    ramp = _ramp(length, start)
    signs = rng.choice([-1.0, 1.0], size=n)
    offset = signs[:, None] * magnitude_m * ramp[None, :]
    out[:, :, altitude_index] += offset
    if vertrate_index is not None:
        change = np.zeros_like(offset)
        change[:, 1:] = np.diff(offset, axis=1) / step_seconds
        out[:, :, vertrate_index] += change
    return out, _mask_from(length, start, n)


def speed_anomaly(
    batch: np.ndarray,
    velocity_index: int,
    factor: float,
    onset_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    out = batch.copy()
    n, length, _ = out.shape
    start = onset_index(length, onset_fraction)
    out[:, start:, velocity_index] = np.clip(out[:, start:, velocity_index] * factor, 0.0, None)
    return out, _mask_from(length, start, n)


def sensor_freeze(
    batch: np.ndarray,
    onset_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    out = batch.copy()
    n, length, _ = out.shape
    start = onset_index(length, onset_fraction)
    out[:, start:, :] = out[:, start : start + 1, :]
    return out, _mask_from(length, start, n)


def holding_pattern(
    batch: np.ndarray,
    x_index: int,
    y_index: int,
    sin_index: int,
    cos_index: int,
    velocity_index: int,
    onset_fraction: float,
    turn_period_seconds: float,
    rng: np.random.Generator,
    vertrate_index: int | None = None,
    step_seconds: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    out = batch.copy()
    n, length, _ = out.shape
    start = onset_index(length, onset_fraction)
    span = length - start

    speed = np.clip(out[:, start, velocity_index], 0.0, None)
    heading0 = np.arctan2(out[:, start, sin_index], out[:, start, cos_index])
    omega = 2.0 * np.pi / turn_period_seconds
    elapsed = np.arange(span) * step_seconds
    heading = heading0[:, None] + omega * elapsed[None, :]

    east_step = speed[:, None] * step_seconds * np.sin(heading)
    north_step = speed[:, None] * step_seconds * np.cos(heading)
    zero = np.zeros((n, 1))
    east = np.concatenate([zero, np.cumsum(east_step[:, :-1], axis=1)], axis=1)
    north = np.concatenate([zero, np.cumsum(north_step[:, :-1], axis=1)], axis=1)

    out[:, start:, x_index] = out[:, start, x_index][:, None] + east
    out[:, start:, y_index] = out[:, start, y_index][:, None] + north
    out[:, start:, sin_index] = np.sin(heading)
    out[:, start:, cos_index] = np.cos(heading)
    if vertrate_index is not None:
        out[:, start:, vertrate_index] = 0.0
    return out, _mask_from(length, start, n)


def build_cases(
    normal_unscaled: np.ndarray,
    indices: dict,
    onset_fraction: float,
    step_seconds: float,
    rng: np.random.Generator,
    synthetic_cfg: dict,
) -> list[tuple]:
    cases = []
    for magnitude in synthetic_cfg["route_deviation"]["magnitudes_m"]:
        windows, mask = route_deviation(
            normal_unscaled, indices["x_rel"], indices["y_rel"], magnitude, onset_fraction, rng
        )
        cases.append(("route_deviation", f"{magnitude:g} m", windows, mask))
    for magnitude in synthetic_cfg["altitude"]["magnitudes_m"]:
        windows, mask = altitude_anomaly(
            normal_unscaled, indices["baroaltitude"], magnitude, onset_fraction, rng,
            indices.get("vertrate"), step_seconds,
        )
        cases.append(("altitude", f"{magnitude:g} m", windows, mask))
    for factor in synthetic_cfg["speed"]["factors"]:
        windows, mask = speed_anomaly(
            normal_unscaled, indices["velocity"], factor, onset_fraction, rng
        )
        cases.append(("speed", f"x{factor:g}", windows, mask))
    for period in synthetic_cfg["holding"]["turn_periods_s"]:
        windows, mask = holding_pattern(
            normal_unscaled, indices["x_rel"], indices["y_rel"], indices["sin_hdg"],
            indices["cos_hdg"], indices["velocity"], onset_fraction, period, rng,
            indices.get("vertrate"), step_seconds,
        )
        cases.append(("holding", f"{period:g} s/turn", windows, mask))
    if synthetic_cfg["freeze"]["enabled"]:
        windows, mask = sensor_freeze(normal_unscaled, onset_fraction, rng)
        cases.append(("freeze", "stuck", windows, mask))
    return cases
