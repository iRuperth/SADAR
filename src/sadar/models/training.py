from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sadar.models.lstm_autoencoder import LSTMAutoencoder
from sadar.models.transformer_autoencoder import TransformerAutoencoder
from sadar.models.vae_lstm import VAELSTM


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


def reconstruction_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    # Default training objective: rebuild the input and penalise the squared
    # error. Models with their own objective (the VAE adds a KL term) pass a
    # different loss function to fit().
    return nn.functional.mse_loss(model(batch), batch)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
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
            loss = loss_fn(model, batch)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    return total / count


def _start_tracking(model_meta: dict, training_cfg: dict):
    # Open an MLflow run for a tagged training and record its configuration.
    # Returns the mlflow module while tracking, or None to skip it (the Optuna
    # search trains without a tag so it stays out of the experiment log).
    arch = model_meta.get("arch")
    if arch is None:
        return None
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:mlruns"))
    mlflow.set_experiment(os.environ.get("SADAR_MLFLOW_EXPERIMENT", "sadar"))
    mlflow.start_run(run_name=arch)
    params = {f"train.{key}": value for key, value in training_cfg.items()}
    params.update({f"model.{key}": value for key, value in model_meta.get("model", {}).items()})
    params["arch"] = arch
    params["n_features"] = model_meta.get("n_features")
    mlflow.log_params(params)
    return mlflow


def fit(
    model: nn.Module,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    training_cfg: dict,
    checkpoint_path: str,
    model_meta: dict,
    device: torch.device,
    loss_fn=None,
) -> float:
    # Shared autoencoder training loop reused by every deep model: Adam with
    # weight decay, a plateau learning-rate scheduler and early stopping on the
    # validation loss. The objective defaults to reconstruction error; the VAE
    # passes its own loss. The best weights are saved with the metadata needed to
    # rebuild the model later.
    if loss_fn is None:
        loss_fn = reconstruction_loss

    train_loader = make_loader(train_windows, training_cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_windows, training_cfg["batch_size"], shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=training_cfg["lr_factor"], patience=training_cfg["lr_patience"]
    )

    tracker = _start_tracking(model_meta, training_cfg)
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0

    try:
        for epoch in range(1, training_cfg["epochs"] + 1):
            train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_loss = _run_epoch(model, val_loader, loss_fn, device)
            scheduler.step(val_loss)
            print(f"epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")
            if tracker is not None:
                tracker.log_metric("train_loss", train_loss, step=epoch)
                tracker.log_metric("val_loss", val_loss, step=epoch)

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

        # Hyperparameter search reuses this loop with checkpoint_path=None to train
        # a candidate without writing it to disk.
        if checkpoint_path is not None:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            payload = {"state_dict": model.state_dict(), **model_meta}
            torch.save(payload, checkpoint_path)

        if tracker is not None:
            tracker.log_metric("best_val_loss", best_val)
            if checkpoint_path is not None:
                tracker.log_artifact(checkpoint_path)
        return best_val
    finally:
        if tracker is not None:
            tracker.end_run()


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


def _build_vae(n_features: int, cfg: dict) -> VAELSTM:
    return VAELSTM(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        beta=cfg["beta"],
    )


_BUILDERS = {"lstm": _build_lstm, "transformer": _build_transformer, "vae": _build_vae}


def load_autoencoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    # Rebuild a trained autoencoder of any supported architecture from its
    # checkpoint (the saved metadata records which one) and move it onto the
    # target device for scoring.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    arch = checkpoint.get("arch", "lstm")
    model = _BUILDERS[arch](checkpoint["n_features"], checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
