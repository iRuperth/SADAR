from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from sadar.data.pipeline import load_config
from sadar.data.scaling import StandardScaler3D
from sadar.eval import synthetic
from sadar.models.baseline import IsolationForestBaseline
from sadar.models.training import (
    load_autoencoder,
    per_step_error,
    reconstruction_error,
    resolve_device,
)


def _load(processed_dir: str, name: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, name))


def _latency_seconds(step_error, onset_idx, step_threshold, step_seconds):
    exceed = step_error[:, onset_idx:] >= step_threshold
    detected = exceed.any(axis=1)
    first = np.argmax(exceed, axis=1).astype(float)
    first[~detected] = np.nan
    median = float(np.nanmedian(first) * step_seconds) if detected.any() else float("nan")
    return median, float(detected.mean())


def run(config: dict) -> list[dict]:
    device = resolve_device(config["model"]["device"])
    processed_dir = config["data"]["processed_dir"]

    preprocessing = load_config(config["preprocessing_config"])
    feature_columns = preprocessing["features"]["columns"]
    step_seconds = preprocessing["resample"]["seconds"]
    indices = {
        name: feature_columns.index(name)
        for name in feature_columns
    }

    scaler = StandardScaler3D.load(os.path.join(processed_dir, "scaler.npz"))
    mean, std = scaler.mean_, scaler.std_

    train = _load(processed_dir, "train.npy")
    val = _load(processed_dir, "val.npy")
    test = _load(processed_dir, "test.npy")

    model = load_autoencoder(config["model"]["checkpoint"], device)

    baseline_cfg = load_config(config["baseline_config"])["isolation_forest"]
    baseline = IsolationForestBaseline(
        n_estimators=baseline_cfg["n_estimators"],
        max_samples=baseline_cfg["max_samples"],
        contamination=baseline_cfg["contamination"],
        seed=config["seed"],
    ).fit(train)

    val_scores = reconstruction_error(model, val, device)
    threshold = float(np.percentile(val_scores, config["threshold"]["val_percentile"]))

    rng = np.random.default_rng(config["seed"])
    sample_size = min(config["synthetic"]["sample_size"], len(test))
    sample = test[rng.choice(len(test), size=sample_size, replace=False)]

    normal_scores = reconstruction_error(model, sample, device)
    base_normal_scores = baseline.score(sample)
    normal_step = per_step_error(model, sample, device)
    step_threshold = float(
        np.percentile(normal_step, config["latency"]["normal_step_percentile"])
    )
    normal_unscaled = synthetic.unscale(sample, mean, std)

    onset = config["synthetic"]["onset_fraction"]
    onset_idx = synthetic.onset_index(sample.shape[1], onset)
    cases = synthetic.build_cases(
        normal_unscaled, indices, onset, step_seconds, rng, config["synthetic"]
    )

    results = []
    for kind, label, perturbed, _mask in cases:
        scaled = synthetic.rescale(perturbed, mean, std)
        scores = reconstruction_error(model, scaled, device)
        base_scores = baseline.score(scaled)

        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(scores))])
        model_roc = roc_auc_score(labels, np.concatenate([normal_scores, scores]))
        base_roc = roc_auc_score(labels, np.concatenate([base_normal_scores, base_scores]))

        step_error = per_step_error(model, scaled, device)
        latency, coverage = _latency_seconds(step_error, onset_idx, step_threshold, step_seconds)

        results.append({
            "kind": kind,
            "intensity": label,
            "model_roc_auc": float(model_roc),
            "baseline_roc_auc": float(base_roc),
            "model_detection": float((scores >= threshold).mean()),
            "median_normal": float(np.median(normal_scores)),
            "median_anomaly": float(np.median(scores)),
            "latency_seconds": latency,
            "detected_within_window": coverage,
        })

    fpr = float((normal_scores >= threshold).mean())
    for row in results:
        row["normal_fpr"] = fpr
    return results


def _print_table(results: list[dict]) -> None:
    header = (
        f"{'anomaly':<16}{'intensity':<12}{'model ROC':>10}{'base ROC':>9}"
        f"{'detect':>8}{'latency':>9}{'within':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        latency = "n/a" if np.isnan(row["latency_seconds"]) else f"{row['latency_seconds']:.0f}s"
        print(
            f"{row['kind']:<16}{row['intensity']:<12}"
            f"{row['model_roc_auc']:>10.3f}{row['baseline_roc_auc']:>9.3f}"
            f"{row['model_detection']:>8.1%}{latency:>9}{row['detected_within_window']:>8.1%}"
        )
    print(f"\nnormal false-alarm rate at threshold: {results[0]['normal_fpr']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained autoencoder against synthetic anomalies."
    )
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint is not None:
        config["model"]["checkpoint"] = args.checkpoint

    results = run(config)
    _print_table(results)


if __name__ == "__main__":
    main()
