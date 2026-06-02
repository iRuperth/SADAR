from __future__ import annotations

import argparse
import os

from sadar.data.pipeline import load_config
from sadar.models.training import fit, load_windows, resolve_device, set_seeds
from sadar.models.transformer_autoencoder import TransformerAutoencoder


def train(config: dict) -> dict:
    set_seeds(config["seed"])
    device = resolve_device(config["training"]["device"])

    processed_dir = config["data"]["processed_dir"]
    train_windows = load_windows(processed_dir, "train.npy")
    val_windows = load_windows(processed_dir, "val.npy")
    n_features = train_windows.shape[2]

    model_cfg = config["model"]
    model = TransformerAutoencoder(
        n_features=n_features,
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        latent_size=model_cfg["latent_size"],
        dropout=model_cfg["dropout"],
        max_len=model_cfg["max_len"],
    )

    output_cfg = config["output"]
    checkpoint_path = os.path.join(output_cfg["dir"], output_cfg["checkpoint"])
    meta = {"arch": "transformer", "model": model_cfg, "n_features": n_features}
    best_val = fit(
        model, train_windows, val_windows, config["training"], checkpoint_path, meta, device
    )
    return {"best_val_loss": best_val, "checkpoint": checkpoint_path, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Transformer autoencoder on normal flights."
    )
    parser.add_argument("--config", default="configs/transformer.yaml")
    args = parser.parse_args()

    result = train(load_config(args.config))
    print(
        f"best val loss {result['best_val_loss']:.5f} on {result['device']} "
        f"-> saved {result['checkpoint']}"
    )


if __name__ == "__main__":
    main()
