from __future__ import annotations

import argparse
import os

from sadar.data.pipeline import load_config
from sadar.models.lstm_autoencoder import LSTMAutoencoder
from sadar.models.training import fit, load_windows, resolve_device, set_seeds


def train(config: dict) -> dict:
    set_seeds(config["seed"])
    device = resolve_device(config["training"]["device"])

    processed_dir = config["data"]["processed_dir"]
    train_windows = load_windows(processed_dir, "train.npy")
    val_windows = load_windows(processed_dir, "val.npy")
    n_features = train_windows.shape[2]

    model_cfg = config["model"]
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=model_cfg["hidden_size"],
        latent_size=model_cfg["latent_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    )

    output_cfg = config["output"]
    checkpoint_path = os.path.join(output_cfg["dir"], output_cfg["checkpoint"])
    meta = {"arch": "lstm", "model": model_cfg, "n_features": n_features}
    best_val = fit(
        model, train_windows, val_windows, config["training"], checkpoint_path, meta, device
    )
    return {"best_val_loss": best_val, "checkpoint": checkpoint_path, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM autoencoder on normal flights.")
    parser.add_argument("--config", default="configs/lstm.yaml")
    args = parser.parse_args()

    result = train(load_config(args.config))
    print(
        f"best val loss {result['best_val_loss']:.5f} on {result['device']} "
        f"-> saved {result['checkpoint']}"
    )


if __name__ == "__main__":
    main()
