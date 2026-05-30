from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from sadar.data.pipeline import load_config
from sadar.data.scaling import StandardScaler3D
from sadar.eval import synthetic
from sadar.models.baseline import IsolationForestBaseline
from sadar.models.training import load_autoencoder, reconstruction_error, resolve_device


def _load(processed_dir: str, name: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, name))


def _real_metrics(score_fn, test, anomalies):
    labels = np.concatenate([np.zeros(len(test)), np.ones(len(anomalies))])
    scores = np.concatenate([score_fn(test), score_fn(anomalies)])
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _synthetic_metrics(score_fn, normal, scaled_cases):
    normal_scores = score_fn(normal)
    per_type = {}
    for kind, label, scaled in scaled_cases:
        scores = score_fn(scaled)
        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(scores))])
        per_type[f"{kind} {label}"] = float(
            roc_auc_score(labels, np.concatenate([normal_scores, scores]))
        )
    return per_type, float(np.mean(list(per_type.values())))


def run(config: dict) -> dict:
    eval_cfg = load_config(config["eval_config"])
    device = resolve_device(eval_cfg["model"]["device"])
    processed_dir = eval_cfg["data"]["processed_dir"]

    preprocessing = load_config(eval_cfg["preprocessing_config"])
    feature_columns = preprocessing["features"]["columns"]
    step_seconds = preprocessing["resample"]["seconds"]
    indices = {name: feature_columns.index(name) for name in feature_columns}

    scaler = StandardScaler3D.load(os.path.join(processed_dir, "scaler.npz"))
    mean, std = scaler.mean_, scaler.std_

    train = _load(processed_dir, "train.npy")
    test = _load(processed_dir, "test.npy")
    anomalies = _load(processed_dir, "anomalies.npy")

    baseline_cfg = load_config(eval_cfg["baseline_config"])["isolation_forest"]
    baseline = IsolationForestBaseline(
        n_estimators=baseline_cfg["n_estimators"],
        max_samples=baseline_cfg["max_samples"],
        contamination=baseline_cfg["contamination"],
        seed=eval_cfg["seed"],
    ).fit(train)

    rng = np.random.default_rng(eval_cfg["seed"])
    sample_size = min(eval_cfg["synthetic"]["sample_size"], len(test))
    sample = test[rng.choice(len(test), size=sample_size, replace=False)]
    normal_unscaled = synthetic.unscale(sample, mean, std)
    onset = eval_cfg["synthetic"]["onset_fraction"]
    cases = synthetic.build_cases(
        normal_unscaled, indices, onset, step_seconds, rng, eval_cfg["synthetic"]
    )
    scaled_cases = [
        (kind, label, synthetic.rescale(perturbed, mean, std))
        for kind, label, perturbed, _mask in cases
    ]

    scorers = [("Baseline", baseline.score)]
    for spec in config["models"]:
        model = load_autoencoder(spec["checkpoint"], device)
        scorers.append((spec["name"], lambda w, m=model: reconstruction_error(m, w, device)))

    results = []
    for name, score_fn in scorers:
        real_roc, real_pr = _real_metrics(score_fn, test, anomalies)
        per_type, synthetic_mean = _synthetic_metrics(score_fn, sample, scaled_cases)
        results.append({
            "model": name,
            "real_roc_auc": real_roc,
            "real_pr_auc": real_pr,
            "synthetic_mean_roc_auc": synthetic_mean,
            "synthetic_per_type": per_type,
        })

    selected = max(results, key=lambda row: row["real_pr_auc"])["model"]
    return {"selected_model": selected, "results": results}


def _print_report(report: dict) -> None:
    header = f"{'model':<14}{'real ROC':>10}{'real PR':>9}{'syn mean ROC':>14}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        print(
            f"{row['model']:<14}{row['real_roc_auc']:>10.3f}"
            f"{row['real_pr_auc']:>9.3f}{row['synthetic_mean_roc_auc']:>14.3f}"
        )
    print(f"\nselected final model (by real PR-AUC): {report['selected_model']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare every detector on both benches and select the final model."
    )
    parser.add_argument("--config", default="configs/compare.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    report = run(config)
    _print_report(report)

    output_path = config["output"]["path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
