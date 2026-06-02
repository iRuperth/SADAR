from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from sadar.data.pipeline import load_config


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    # Isolation Forest is not sequential, so we describe each window by
    # per-feature statistics instead of feeding it the raw time series.
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    minimum = windows.min(axis=1)
    maximum = windows.max(axis=1)
    return np.concatenate([mean, std, minimum, maximum], axis=1)


class IsolationForestBaseline:
    # Reference detector: a tree ensemble that isolates rare points. It learns
    # only from normal flights and scores how anomalous each window looks.
    def __init__(self, n_estimators: int, max_samples, contamination, seed: int) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=seed,
        )

    def fit(self, windows: np.ndarray) -> "IsolationForestBaseline":
        self.model.fit(summarize_windows(windows))
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        # Flip the sign so a higher score means more anomalous, matching the
        # reconstruction-error convention the deep models will use later.
        return -self.model.score_samples(summarize_windows(windows))


def _load(processed_dir: str, name: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, name))


def run(config: dict) -> dict:
    processed_dir = config["data"]["processed_dir"]
    train = _load(processed_dir, "train.npy")
    val = _load(processed_dir, "val.npy")
    test = _load(processed_dir, "test.npy")
    anomalies = _load(processed_dir, "anomalies.npy")

    forest_cfg = config["isolation_forest"]
    baseline = IsolationForestBaseline(
        n_estimators=forest_cfg["n_estimators"],
        max_samples=forest_cfg["max_samples"],
        contamination=forest_cfg["contamination"],
        seed=config["seed"],
    ).fit(train)

    val_scores = baseline.score(val)
    test_scores = baseline.score(test)
    anomaly_scores = baseline.score(anomalies)

    # The alert threshold is set on the validation normals only: the chosen
    # percentile fixes the target false-alarm rate before touching the test set.
    percentile = config["threshold"]["val_percentile"]
    threshold = float(np.percentile(val_scores, percentile))

    # Quantify separation on held-out normal windows (label 0) versus the real
    # anomalous flights (label 1). PR-AUC is the headline number given how rare
    # the anomalies are.
    labels = np.concatenate([np.zeros(len(test_scores)), np.ones(len(anomaly_scores))])
    scores = np.concatenate([test_scores, anomaly_scores])

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "threshold": threshold,
        "detection_rate": float((anomaly_scores >= threshold).mean()),
        "false_positive_rate": float((test_scores >= threshold).mean()),
        "median_normal": float(np.median(test_scores)),
        "median_anomaly": float(np.median(anomaly_scores)),
        "n_test": int(len(test_scores)),
        "n_anomalies": int(len(anomaly_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate the Isolation Forest baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    metrics = run(load_config(args.config))

    print(
        f"ROC-AUC {metrics['roc_auc']:.3f} "
        f"PR-AUC {metrics['pr_auc']:.3f} | "
        f"detection {metrics['detection_rate']:.1%} at "
        f"FPR {metrics['false_positive_rate']:.1%} | "
        f"median score normal {metrics['median_normal']:.3f} "
        f"vs anomaly {metrics['median_anomaly']:.3f} "
        f"({metrics['n_test']} normal / {metrics['n_anomalies']} anomaly windows)"
    )


if __name__ == "__main__":
    main()
