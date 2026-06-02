from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

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


def _zscore_calibrator(val_scores: np.ndarray):
    # Each model scores on a different scale, so before averaging or taking the
    # max we map every scorer to the same z-score using the validation normals.
    mean = float(val_scores.mean())
    std = float(val_scores.std()) or 1.0
    return lambda scores: (scores - mean) / std


def _build_ensemble(calibrated_scorers, reducer):
    # Combine z-scored model outputs into a single score per window. "mean" is
    # the consensus reducer (smooths false positives), "max" is the sensitive
    # one (any model raising a flag is enough).
    def score(windows: np.ndarray) -> np.ndarray:
        stacked = np.stack([fn(windows) for fn in calibrated_scorers], axis=0)
        return reducer(stacked, axis=0)
    return score


def _real_metrics(score_fn, test, anomalies):
    labels = np.concatenate([np.zeros(len(test)), np.ones(len(anomalies))])
    scores = np.concatenate([score_fn(test), score_fn(anomalies)])
    return labels, scores


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


def _step_score_factory(model, device):
    # Wraps the per-step reconstruction error so latency can reuse it without
    # knowing which deep model produced the score.
    return lambda windows: per_step_error(model, windows, device)


def _latency_per_type(step_fn, normal_sample, scaled_cases, onset_idx, step_seconds, percentile):
    # Median seconds between the injected onset and the first time the per-step
    # error crosses the normal-derived threshold. Reported per anomaly type.
    normal_step = step_fn(normal_sample)
    threshold = float(np.percentile(normal_step, percentile))
    per_type = {}
    for kind, label, scaled in scaled_cases:
        step = step_fn(scaled)
        exceed = step[:, onset_idx:] >= threshold
        detected = exceed.any(axis=1)
        first = np.argmax(exceed, axis=1).astype(float)
        first[~detected] = np.nan
        median = (
            float(np.nanmedian(first) * step_seconds) if detected.any() else float("nan")
        )
        per_type[f"{kind} {label}"] = {
            "latency_seconds": median,
            "detected_within_window": float(detected.mean()),
        }
    return per_type, threshold


def _save_curves(curves_dir: str, name: str, labels, scores) -> dict:
    # Persist PR/ROC curves both as PNG (for the report) and as JSON arrays (so
    # the dashboard can render them without recomputing).
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    os.makedirs(curves_dir, exist_ok=True)
    safe = name.lower().replace(" ", "_").replace("/", "_")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR curve - {name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(curves_dir, f"pr_{safe}.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC curve - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(curves_dir, f"roc_{safe}.png"), dpi=120)
    plt.close(fig)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def _save_confusion(curves_dir: str, name: str, labels, scores, threshold) -> dict:
    import matplotlib.pyplot as plt

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels.astype(int), predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "anomaly"])
    ax.set_yticklabels(["normal", "anomaly"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion - {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    safe = name.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(curves_dir, f"cm_{safe}.png"), dpi=120)
    plt.close(fig)

    return {
        "threshold": float(threshold),
        "matrix": matrix.tolist(),
    }


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
    val = _load(processed_dir, "val.npy")
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
    onset_idx = synthetic.onset_index(sample.shape[1], onset)
    cases = synthetic.build_cases(
        normal_unscaled, indices, onset, step_seconds, rng, eval_cfg["synthetic"]
    )
    scaled_cases = [
        (kind, label, synthetic.rescale(perturbed, mean, std))
        for kind, label, perturbed, _mask in cases
    ]

    val_percentile = eval_cfg["threshold"]["val_percentile"]
    latency_percentile = eval_cfg["latency"]["normal_step_percentile"]

    scorers = []
    step_scorers = {}

    base_val = baseline.score(val)
    base_calibrator = _zscore_calibrator(base_val)
    base_threshold = float(np.percentile(base_val, val_percentile))
    scorers.append({
        "name": "Baseline",
        "score_fn": baseline.score,
        "calibrated": lambda w: base_calibrator(baseline.score(w)),
        "threshold": base_threshold,
    })

    def _make_model_scorer(m):
        return lambda w: reconstruction_error(m, w, device)

    def _make_calibrated(fn, calibrator):
        return lambda w: calibrator(fn(w))

    dl_calibrated = []
    for spec in config["models"]:
        model = load_autoencoder(spec["checkpoint"], device)
        score_fn = _make_model_scorer(model)
        val_scores = score_fn(val)
        calibrator = _zscore_calibrator(val_scores)
        calibrated = _make_calibrated(score_fn, calibrator)
        threshold = float(np.percentile(val_scores, val_percentile))
        scorers.append({
            "name": spec["name"],
            "score_fn": score_fn,
            "calibrated": calibrated,
            "threshold": threshold,
        })
        step_scorers[spec["name"]] = _step_score_factory(model, device)
        dl_calibrated.append(calibrated)

    ensemble_cfg = config.get("ensemble", {})
    if ensemble_cfg.get("enabled", True) and len(dl_calibrated) >= 2:
        for reducer_name in ensemble_cfg.get("reducers", ["mean", "max"]):
            reducer = {"mean": np.mean, "max": np.max}[reducer_name]
            ensemble_fn = _build_ensemble(dl_calibrated, reducer)
            ensemble_val = ensemble_fn(val)
            ensemble_threshold = float(np.percentile(ensemble_val, val_percentile))
            scorers.append({
                "name": f"Ensemble-{reducer_name}",
                "score_fn": ensemble_fn,
                "calibrated": ensemble_fn,
                "threshold": ensemble_threshold,
            })

    curves_dir = config["output"].get("curves_dir")
    results = []
    for entry in scorers:
        name = entry["name"]
        score_fn = entry["score_fn"]
        labels, scores = _real_metrics(score_fn, test, anomalies)
        per_type, synthetic_mean = _synthetic_metrics(score_fn, sample, scaled_cases)

        row = {
            "model": name,
            "real_roc_auc": float(roc_auc_score(labels, scores)),
            "real_pr_auc": float(average_precision_score(labels, scores)),
            "synthetic_mean_roc_auc": synthetic_mean,
            "synthetic_per_type": per_type,
            "threshold": entry["threshold"],
        }

        if name in step_scorers:
            latency, latency_threshold = _latency_per_type(
                step_scorers[name],
                sample,
                scaled_cases,
                onset_idx,
                step_seconds,
                latency_percentile,
            )
            row["latency_per_type"] = latency
            row["latency_step_threshold"] = latency_threshold
            valid = [v["latency_seconds"] for v in latency.values() if not np.isnan(v["latency_seconds"])]
            row["latency_seconds_median"] = float(np.median(valid)) if valid else float("nan")

        if curves_dir is not None:
            row["curves"] = _save_curves(curves_dir, name, labels, scores)
            row["confusion"] = _save_confusion(curves_dir, name, labels, scores, entry["threshold"])

        results.append(row)

    selected = max(results, key=lambda r: r["real_pr_auc"])["model"]
    return {"selected_model": selected, "results": results}


def _print_report(report: dict) -> None:
    header = f"{'model':<18}{'real ROC':>10}{'real PR':>9}{'syn mean ROC':>14}{'latency':>10}"
    print(header)
    print("-" * len(header))
    for row in report["results"]:
        latency = row.get("latency_seconds_median")
        latency_str = "n/a" if latency is None or np.isnan(latency) else f"{latency:.0f}s"
        print(
            f"{row['model']:<18}{row['real_roc_auc']:>10.3f}"
            f"{row['real_pr_auc']:>9.3f}{row['synthetic_mean_roc_auc']:>14.3f}"
            f"{latency_str:>10}"
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
