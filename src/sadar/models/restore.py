from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


CHECKPOINTS = ("lstm_autoencoder.pt", "transformer_autoencoder.pt", "vae_lstm.pt")


def _latest_artifact(mlruns_root: Path, filename: str) -> Path | None:
    candidates = sorted(
        mlruns_root.glob(f"*/*/artifacts/{filename}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def restore(models_dir: Path, mlruns_root: Path) -> list[tuple[str, str]]:
    models_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str]] = []
    for name in CHECKPOINTS:
        target = models_dir / name
        if target.exists():
            results.append((name, "ok"))
            continue
        source = _latest_artifact(mlruns_root, name)
        if source is None:
            results.append((name, "missing"))
            continue
        shutil.copy2(source, target)
        results.append((name, f"restored from {source.relative_to(mlruns_root.parent)}"))
    return results


def main() -> int:
    models_dir = Path(os.environ.get("SADAR_MODELS_DIR", "models"))
    mlruns_root = Path(os.environ.get("MLFLOW_TRACKING_DIR", "mlruns"))
    if not mlruns_root.exists():
        print(f"no mlruns directory at {mlruns_root}")
        return 1
    statuses = restore(models_dir, mlruns_root)
    missing = [name for name, status in statuses if status == "missing"]
    for name, status in statuses:
        print(f"{name}: {status}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
