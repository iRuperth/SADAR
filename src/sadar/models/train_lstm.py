from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sadar.data.pipeline import load_config
from sadar.models.lstm_autoencoder import LSTMAutoencoder


def resolve_device(requested: str) -> torch.device:
    # "auto" prefers Apple's Metal backend (MPS), then CUDA, then CPU.
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load(processed_dir: str, name: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, name))


def _loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = torch.from_numpy(windows.astype(np.float32))
    return DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=shuffle)


def reconstruction_error(
    model: nn.Module, windows: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    # Per-window mean squared error between input and reconstruction. A higher
    # value means the window was harder to rebuild, i.e. more anomalous. This is
    # the score the detector thresholds on.
    model.eval()
    scores = []
    with torch.no_grad():
        for (batch,) in _loader(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            reconstructed = model(batch)
            error = ((batch - reconstructed) ** 2).mean(dim=(1, 2))
            scores.append(error.cpu().numpy())
    return np.concatenate(scores)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    # One pass over the data. With an optimizer we train; without one we only
    # measure the loss, which is how the validation pass reuses this.
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    return total / count


def train(config: dict) -> dict:
    set_seeds(config["seed"])
    device = resolve_device(config["training"]["device"])

    processed_dir = config["data"]["processed_dir"]
    train_windows = _load(processed_dir, "train.npy")
    val_windows = _load(processed_dir, "val.npy")
    n_features = train_windows.shape[2]

    training_cfg = config["training"]
    train_loader = _loader(train_windows, training_cfg["batch_size"], shuffle=True)
    val_loader = _loader(val_windows, training_cfg["batch_size"], shuffle=False)

    model_cfg = config["model"]
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=model_cfg["hidden_size"],
        latent_size=model_cfg["latent_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=training_cfg["lr_factor"], patience=training_cfg["lr_patience"]
    )

    output_cfg = config["output"]
    os.makedirs(output_cfg["dir"], exist_ok=True)
    checkpoint_path = os.path.join(output_cfg["dir"], output_cfg["checkpoint"])

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")

        # Keep the best weights seen so far and stop once the validation loss
        # stops improving for a while (early stopping).
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_cfg["patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save the weights plus the shape information needed to rebuild the model.
    torch.save(
        {"state_dict": model.state_dict(), "model": model_cfg, "n_features": n_features},
        checkpoint_path,
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
