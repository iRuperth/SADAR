from __future__ import annotations

import numpy as np


class StandardScaler3D:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, windows: np.ndarray) -> "StandardScaler3D":
        self.mean_ = windows.mean(axis=(0, 1))
        self.std_ = windows.std(axis=(0, 1))
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return ((windows - self.mean_) / self.std_).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> "StandardScaler3D":
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.std_ = data["std"]
        return scaler
