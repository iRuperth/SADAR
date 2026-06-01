<div align="center">

<img src="docs/assets/logo.png" alt="SADAR" width="220" />

<h1>SADAR</h1>

<p><b>Smart Anomaly Detection for Aviation Routes</b></p>

<p>Early detection of non-conforming flight behavior over Madrid-Barajas (LEMD).</p>

<p>
  <img src="docs/assets/qr-placeholder.png" alt="QR" width="220" onerror="this.style.display='none'" />
</p>

<p>
  <a href="README.md"><img src="https://img.shields.io/badge/English-0d1a1f?style=flat&logoColor=7fd1c6&labelColor=0a1014" alt="English" /></a>
  ·
  <a href="README.es.md"><img src="https://img.shields.io/badge/Espa%C3%B1ol-0d1a1f?style=flat&logoColor=7fd1c6&labelColor=0a1014" alt="Español" /></a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-0a1014?logo=python&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-0a1014?logo=pytorch&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/FastAPI-0.136-0a1014?logo=fastapi&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/React-18-0a1014?logo=react&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/Vite-5-0a1014?logo=vite&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/MLflow-3.x-0a1014?logo=mlflow&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/Optuna-4.x-0a1014?logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/uv-package%20mgr-0a1014?logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-0a1014?logo=huggingface&logoColor=7fd1c6&labelColor=0d1a1f" />
  <img src="https://img.shields.io/badge/Docker-compose-0a1014?logo=docker&logoColor=7fd1c6&labelColor=0d1a1f" />
</p>

<p>
  <a href="https://medium.com/@devrup404/when-the-most-complex-model-is-not-the-one-that-detects-best-aebafe115ea2"><img src="https://img.shields.io/badge/Medium-Read%20the%20article-0a1014?logo=medium&logoColor=7fd1c6&labelColor=0d1a1f" alt="Read on Medium" /></a>
</p>

</div>

---

### Project description

**SADAR** (Smart Anomaly Detection for Aviation Routes) is a deep-learning system that learns what a **normal** approach or departure looks like at Madrid-Barajas (LEMD) and flags any trajectory that deviates from the learned pattern: go-arounds, anomalous routes, transponder dropouts or maneuvers consistent with an emergency.

It is built on top of three deep autoencoders trained on ~20.000 real ADS-B flights from the OpenSky Network: an **LSTM autoencoder**, a **Transformer autoencoder** and a **VAE-LSTM**. The three models are trained under the same conditions and compared head-to-head; the **VAE-LSTM** is the one served in production because it combines the best PR-AUC with the lowest detection latency.

> **Scope.** SADAR is trained exclusively on normal operations at LEMD and works as a trajectory conformance monitor: it learns the normal pattern and surfaces any deviation for human review. It is not a predictor of catastrophic events.

### Highlights

- **One-class deep learning** on 7 dynamic features (position, altitude, velocity, sin/cos heading, vertical rate).
- **Comparative model selection**: Isolation Forest baseline vs LSTM vs Transformer vs VAE-LSTM under identical preprocessing and metrics.
- **Synthetic anomaly bench** with five injection types (route deviation, altitude bust, speed anomaly, holding, transponder freeze) at several intensities, measures **PR-AUC** and **detection latency**.
- **Tower-console dashboard** built with React + Vite: live radar, ranked anomaly list, simulator with sliders, metrics page.
- **Fully reproducible**: `uv` for Python, `pnpm` for the frontend, YAML configs, fixed seeds.

### Tech stack

| Layer | Tools |
|---|---|
| Models | PyTorch · scikit-learn (baseline) · Optuna · MLflow |
| Data | pandas · pyarrow · pyproj · NumPy |
| Backend | FastAPI · uvicorn · uv |
| Frontend | React 18 · Vite 5 · TypeScript · framer-motion |
| Deploy | Docker · Hugging Face Spaces |

### Live demo screenshots

<details>
<summary><b>1. Tower console - live radar over LEMD</b></summary>

![Live radar over Madrid-Barajas](docs/assets/img1.png)

Real-time scope centered on Madrid-Barajas. Aircraft are color-coded by status (tracked / alert) and the side panel lists every flight currently in the airspace with its callsign and state.

</details>

<details>
<summary><b>2. Simulator - inject an anomaly and watch the alert fire</b></summary>

![Anomaly simulator](docs/assets/img2.png)

The star of the demo. Pick a normal flight, choose an anomaly type (route deviation, altitude bust, speed anomaly, holding, transponder cut), set intensity and onset, and inject it live. The score on the right crosses the threshold and the latency is reported in seconds.

</details>

<details>
<summary><b>3. Metrics - model comparison and selected detector</b></summary>

![Model comparison page](docs/assets/img3.png)

The page reports PR-AUC and per-anomaly detection performance for the baseline, LSTM, Transformer and VAE-LSTM, plus the rationale for selecting the **VAE-LSTM** as the production model.

</details>

<details>
<summary><b>4. Metrics - per-anomaly breakdown</b></summary>

![Per-anomaly metrics breakdown](docs/assets/img4.png)

Detailed PR-AUC per anomaly type and intensity. Useful for justifying which detector handles each failure mode best.

</details>

<details>
<summary><b>5. Tower console - full LEMD traffic snapshot</b></summary>

![Full LEMD traffic](docs/assets/img5.png)

A wider view of the radar with the full visible fleet, flight-data block and the live anomaly score histogram at the bottom-right.

</details>

<details>
<summary><b>6. Tower console - alert raised on a tracked flight</b></summary>

![Alert raised on a flight](docs/assets/img6.png)

A flagged aircraft turns red and the controller gets the score, the threshold and the suggested classification in the right-hand panel.

</details>

### Architecture

```
        ┌──────────────────┐         ┌──────────────────┐
        │  ADS-B parquet   │         │  Synthetic       │
        │  (OpenSky / LEMD)│         │  anomaly bench   │
        └────────┬─────────┘         └────────┬─────────┘
                 │                            │
        ┌────────▼─────────────────┐          │
        │  Preprocessing           │          │
        │  · clean / dedupe        │          │
        │  · resample 10s          │          │
        │  · runway-relative XY    │          │
        │  · sin/cos hdg           │          │
        │  · per-feature           │          │
        │    standardize           │          │
        └────────┬─────────────────┘          │
                 │                            │
        ┌────────▼────────────────────────────▼─────────┐
        │     Training (normals only, one-class)        │
        │  ┌──────────┐ ┌─────────────┐ ┌────────────┐  │
        │  │ Isolation│ │ LSTM-AE     │ │ Transformer│  │
        │  │ Forest   │ │             │ │ AE         │  │
        │  └──────────┘ └─────────────┘ └────────────┘  │
        │            ┌─────────────────┐                │
        │            │  VAE-LSTM ★      │  ← selected   │
        │            └─────────────────┘                │
        └────────┬──────────────────────────────────────┘
                 │  checkpoint + scaler + threshold
        ┌────────▼─────────┐
        │  FastAPI service │  /api/health · /api/scene
        │  (uvicorn)       │  /api/flights · /api/metrics
        └────────┬─────────┘  /api/simulate
                 │
        ┌────────▼─────────┐
        │  React + Vite    │  Radar · Simulator · Metrics
        │  dashboard       │
        └──────────────────┘
```

### Run it locally

Prerequisites: **Python 3.11**, [`uv`](https://docs.astral.sh/uv/) for Python deps, **Node 20+** and [`pnpm`](https://pnpm.io/) for the frontend.

Dashboard at `http://localhost:5180`, API at `http://localhost:8000`.

<details>
<summary><b>macOS / Linux - with Makefile (recommended)</b></summary>

```bash
uv sync
pnpm -C frontend install
make dev
```

`make dev` boots the FastAPI backend and the Vite frontend together with hot reload.

</details>

<details>
<summary><b>macOS / Linux - without Makefile</b></summary>

Two terminals:

```bash
# terminal 1: backend
uv sync
uv run uvicorn sadar.serve.app:app --port 8000

# terminal 2: frontend
pnpm -C frontend install
pnpm -C frontend dev
```

</details>

<details>
<summary><b>Windows (PowerShell)</b></summary>

PowerShell does not ship with `make`; run the two services in two terminals.

```powershell
# install once
uv sync
pnpm -C frontend install

# terminal 1: backend
uv run uvicorn sadar.serve.app:app --port 8000

# terminal 2: frontend
pnpm -C frontend dev
```

If `pnpm` is missing: `npm install -g pnpm`. If `uv` is missing: `winget install astral-sh.uv`.

</details>

<details>
<summary><b>Windows with Make (Chocolatey / scoop / WSL)</b></summary>

```powershell
choco install make    # or: scoop install make
make dev
```

Or run everything inside WSL Ubuntu and follow the macOS / Linux instructions above.

</details>

### Run it with Docker

Same on every OS, only Docker Desktop / Docker Engine required.

<details>
<summary><b>One container with Docker Compose</b></summary>

```bash
docker compose up --build
```

Open `http://localhost:5180`. Models, preprocessed tensors and reports are mounted read-only from `./models`, `./data/processed` and `./reports`.

</details>

<details>
<summary><b>Single image (same as production / Hugging Face Spaces)</b></summary>

```bash
docker build -t sadar .
docker run --rm -p 7860:7860 sadar
```

Open `http://localhost:7860`. Models, tensors and reports are baked into the image; no volumes needed.

</details>

### Deploy to Hugging Face Spaces

The repository ships with a [Dockerfile](Dockerfile) ready for HF Spaces (single image, frontend served by FastAPI on port 7860).

```bash
git lfs install
git clone https://huggingface.co/spaces/<your-user>/sadar
cd sadar
# copy the SADAR repository files into this folder
git lfs track "*.npy" "*.npz" "*.pt"
git add . && git commit -m "feat: initial deploy" && git push
```

### Repository layout

```
SADAR/
├── src/sadar/                           Python package (importable as `sadar`)
│   ├── data/                            Data pipeline (raw parquet → tensors)
│   │   ├── pipeline.py                  Entry point: `sadar-preprocess`
│   │   ├── io.py                        Parquet loading, dedupe, manifests
│   │   ├── cleaning.py                  Null handling, gap filtering, go-around removal
│   │   ├── features.py                  Runway-relative XY, sin/cos heading
│   │   ├── scaling.py                   Per-feature standardization (train-only fit)
│   │   ├── splitting.py                 Leak-free split by flight_id and date
│   │   └── windowing.py                 Fixed-length window slicing
│   ├── models/                          Models and trainers
│   │   ├── baseline.py                  Isolation Forest baseline
│   │   ├── lstm_autoencoder.py          LSTM-AE architecture
│   │   ├── transformer_autoencoder.py   Transformer-AE architecture
│   │   ├── vae_lstm.py                  VAE-LSTM architecture (selected)
│   │   ├── training.py                  Shared training loop, device, checkpoint I/O
│   │   ├── train_lstm.py                Entry: `sadar-train-lstm`
│   │   ├── train_transformer.py         Entry: `sadar-train-transformer`
│   │   ├── train_vae.py                 Entry: `sadar-train-vae`
│   │   └── tune_vae.py                  Optuna search: `sadar-tune-vae`
│   ├── eval/                            Evaluation and synthetic bench
│   │   ├── synthetic.py                 Anomaly injection (5 types, several intensities)
│   │   ├── evaluate.py                  Per-model evaluation: `sadar-evaluate`
│   │   └── compare.py                   Comparative report: `sadar-compare`
│   └── serve/                           Production inference
│       ├── app.py                       FastAPI app (also serves the SPA in prod)
│       └── inference.py                 ConformanceService (loads model + scaler)
│
├── frontend/                            React + Vite dashboard (pnpm)
│   ├── src/
│   │   ├── pages/                       Dashboard, Simulator, Metrics, Presentation
│   │   ├── components/                  RadarScope, panels, charts
│   │   ├── api.ts                       Typed client for /api/*
│   │   └── i18n.tsx                     EN / ES translations
│   ├── public/                          Logo and static assets
│   ├── vite.config.ts                    Dev proxy /api → :8000
│   └── package.json
│
├── configs/                              YAML configuration
│   ├── preprocessing.yaml
│   ├── baseline.yaml
│   ├── lstm.yaml · transformer.yaml · vae.yaml
│   ├── tune.yaml · eval.yaml · compare.yaml
│   └── serve.yaml                         Selects production checkpoint
│
├── models/                               Trained checkpoints (Git LFS)
│   ├── lstm_autoencoder.pt
│   ├── transformer_autoencoder.pt
│   └── vae_lstm.pt                      ★ served in production
│
├── data/
│   ├── raw/                           ADS-B parquet (not in repository, $SADAR_DATA_DIR)
│   └── processed/                     Standardized tensors (Git LFS)
│       ├── scaler.npz
│       ├── train.npy · val.npy · test.npy
│       └── anomalies.npy
│
├── reports/                         Generated metrics
│   ├── model_comparison.json
│   └── vae_tuning.json
│
├── notebooks/                       EDA
├── docs/assets/                     Screenshots and logo for the README
│
├── Dockerfile                       Production image (HF Spaces, single container)
├── docker-compose.yml               Local two-container stack
├── pyproject.toml · uv.lock         Python deps managed by uv
├── Makefile                         `make dev`, `make preprocess`, ...
└── README.md · README.es.md         This file (EN / ES)
```

### Limitations

- No real catastrophic events in the dataset, SADAR validates a methodology, not historical incidents.
- No flight plans are available, "intended route" is approximated as "learned normal pattern".
- Transponder gaps are mostly receiver coverage, used as a feature, not as a ground-truth label.
- Only four real emergency-squawk flights, used to validate, never to train.

---

<div align="center">
  <sub><b>SADAR</b> · Smart Anomaly Detection for Aviation Routes</sub>
  <br />
  <sub>Deep-learning flight conformance monitoring over Madrid-Barajas (LEMD) · 2026</sub>
</div>
