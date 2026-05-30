from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sadar.models.lstm_autoencoder import LSTMAutoencoder
from sadar.models.transformer_autoencoder import TransformerAutoencoder


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


def load_windows(processed_dir: str, name: str) -> np.ndarray:
    return np.load(os.path.join(processed_dir, name))


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
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
        for (batch,) in make_loader(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            error = ((batch - model(batch)) ** 2).mean(dim=(1, 2))
            scores.append(error.cpu().numpy())
    return np.concatenate(scores)


def per_step_error(
    model: nn.Module, windows: np.ndarray, device: torch.device, batch_size: int = 512
) -> np.ndarray:
    # Reconstruction error at each timestep (averaged over features only). Keeps
    # the time axis so we can see how soon within a window the model reacts to an
    # injected anomaly, which is what detection latency measures.
    model.eval()
    chunks = []
    with torch.no_grad():
        for (batch,) in make_loader(windows, batch_size, shuffle=False):
            batch = batch.to(device)
            error = ((batch - model(batch)) ** 2).mean(dim=2)
            chunks.append(error.cpu().numpy())
    return np.concatenate(chunks)


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
            loss = criterion(model(batch), batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    return total / count


def fit(
    model: nn.Module,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    training_cfg: dict,
    checkpoint_path: str,
    model_meta: dict,
    device: torch.device,
) -> float:
    # Shared autoencoder training loop reused by every deep model: Adam with
    # weight decay, MSE reconstruction loss, a plateau learning-rate scheduler
    # and early stopping on the validation loss. The best weights are saved with
    # the metadata needed to rebuild the model later.
    train_loader = make_loader(train_windows, training_cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_windows, training_cfg["batch_size"], shuffle=False)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=training_cfg["lr_factor"], patience=training_cfg["lr_patience"]
    )

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")

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

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {"state_dict": model.state_dict(), **model_meta}
    torch.save(payload, checkpoint_path)
    return best_val


def _build_lstm(n_features: int, cfg: dict) -> LSTMAutoencoder:
    return LSTMAutoencoder(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )


def _build_transformer(n_features: int, cfg: dict) -> TransformerAutoencoder:
    return TransformerAutoencoder(
        n_features=n_features,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        latent_size=cfg["latent_size"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
    )


_BUILDERS = {"lstm": _build_lstm, "transformer": _build_transformer}


def load_autoencoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    # Rebuild a trained autoencoder of any supported architecture from its
    # checkpoint (the saved metadata records which one) and move it onto the
    # target device for scoring.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint.get("arch", "lstm")
    model = _BUILDERS[arch](checkpoint["n_features"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
