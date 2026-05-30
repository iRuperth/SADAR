from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from sadar.data.pipeline import load_config
from sadar.data.scaling import StandardScaler3D
from sadar.eval import synthetic
from sadar.models.training import (
    fit,
    load_windows,
    reconstruction_error,
    resolve_device,
    set_seeds,
)
from sadar.models.vae_lstm import VAELSTM


def _elbo(model, batch):
    return model.elbo_loss(batch)


def _synthetic_mean_roc(model, normal, mean, std, indices, onset, step_seconds, synthetic_cfg, rng, device):
    # The tuning objective is detection power, not reconstruction loss: a lower
    # reconstruction loss does not imply better detection. We inject the synthetic
    # anomalies into the validation normals only, so the test set and the real
    # anomalies stay untouched during the search.
    normal_scores = reconstruction_error(model, normal, device)
    normal_unscaled = synthetic.unscale(normal, mean, std)
    cases = synthetic.build_cases(normal_unscaled, indices, onset, step_seconds, rng, synthetic_cfg)
    rocs = []
    for _kind, _label, perturbed, _mask in cases:
        scores = reconstruction_error(model, synthetic.rescale(perturbed, mean, std), device)
        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(scores))])
        rocs.append(roc_auc_score(labels, np.concatenate([normal_scores, scores])))
    return float(np.mean(rocs))


def run(config: dict) -> optuna.Study:
    device = resolve_device(config["training"]["device"])
    processed_dir = config["data"]["processed_dir"]
    train = load_windows(processed_dir, "train.npy")
    val = load_windows(processed_dir, "val.npy")
    n_features = train.shape[2]

    preprocessing = load_config(config["preprocessing_config"])
    feature_columns = preprocessing["features"]["columns"]
    step_seconds = preprocessing["resample"]["seconds"]
    indices = {name: feature_columns.index(name) for name in feature_columns}

    scaler = StandardScaler3D.load(os.path.join(processed_dir, "scaler.npz"))
    mean, std = scaler.mean_, scaler.std_

    space = config["search_space"]
    synthetic_cfg = config["synthetic"]
    onset = synthetic_cfg["onset_fraction"]

    sample_size = min(synthetic_cfg["sample_size"], len(val))
    sample = val[np.random.default_rng(config["seed"]).choice(len(val), size=sample_size, replace=False)]

    def objective(trial: optuna.Trial) -> float:
        # Every trial trains from the same seed so the only thing that changes is
        # the hyperparameters being searched.
        set_seeds(config["seed"])
        model = VAELSTM(
            n_features=n_features,
            hidden_size=trial.suggest_categorical("hidden_size", space["hidden_size"]),
            latent_size=trial.suggest_categorical("latent_size", space["latent_size"]),
            num_layers=trial.suggest_categorical("num_layers", space["num_layers"]),
            dropout=trial.suggest_float("dropout", *space["dropout"]),
            beta=trial.suggest_float("beta", *space["beta"], log=True),
        )
        training_cfg = {
            **config["training"],
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        }
        fit(model, train, val, training_cfg, None, {}, device, loss_fn=_elbo)
        return _synthetic_mean_roc(
            model, sample, mean, std, indices, onset, step_seconds, synthetic_cfg,
            np.random.default_rng(config["seed"]), device,
        )

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"])
    )
    study.optimize(objective, n_trials=config["n_trials"])
    return study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the VAE-LSTM with Optuna on validation synthetic anomalies."
    )
    parser.add_argument("--config", default="configs/tune.yaml")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.trials is not None:
        config["n_trials"] = args.trials

    study = run(config)
    print(f"best synthetic mean ROC {study.best_value:.4f}")
    print(f"best params {study.best_params}")

    output_path = config["output"]["path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, handle, indent=2)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
