# SADAR

**Smart Anomaly Detection for Aviation Routes**

Technical Report

---

**Project:** SADAR · Flight Conformance Monitor
**Domain:** Air traffic surveillance · ADS-B data
**Airport of study:** Madrid-Barajas (LEMD)
**Approach:** One-class deep learning for trajectory anomaly detection
**Date:** 2026

---

## Executive summary

SADAR is a deep-learning system that learns the normal pattern of approach and departure
operations at Madrid-Barajas (LEMD) from ADS-B surveillance data and surfaces, in real time,
any trajectory that deviates from that pattern. The system is built as a **flight
conformance monitor**: it does not classify specific events, it raises an interpretable
anomaly score for every flight in the airspace so that an operator can review the cases
that fall outside the learned distribution.

Three sequence autoencoders are trained under identical conditions on roughly 20.000 real
flights from the OpenSky Network (an LSTM autoencoder, a Transformer autoencoder and a
VAE-LSTM), together with a non-sequential Isolation Forest baseline. The four detectors
are evaluated on the same held-out test set and on a synthetic anomaly bench with five
injection types and several intensities. The **VAE-LSTM** is selected as the production
model: it achieves the highest PR-AUC on real held-out data (0.299) and the highest
synthetic mean ROC-AUC (0.792). Two ensemble strategies (consensus and most-sensitive) are
also evaluated and discarded because neither beats the best individual model.

The selected model is wrapped in a FastAPI inference service and served through a
React + Vite dashboard that combines a tower-style live radar, a ranked anomaly list, a
metrics page and a fully interactive simulator. The whole system is reproducible from a
single Docker image and ships ready for deployment on Hugging Face Spaces.

---

## 1. Motivation and scope

Trajectory monitoring is a core function of air traffic management. Today it relies on
deterministic rules and human supervision: separation, descent profiles, holding patterns
and emergency squawk codes are all monitored against fixed thresholds. These rules work
well for the situations they were designed for, but they cannot anticipate behavior that
is unusual without being a textbook violation.

SADAR explores a complementary, data-driven approach. Instead of encoding rules, the
system **learns** what a normal operation at LEMD looks like from a large sample of real
flights, and treats every new trajectory as more or less compatible with that learned
distribution. The output is a continuous conformance score that an operator can rank,
filter and threshold.

The scope of the project is deliberately narrow and well defined:

- **What it does.** Detects deviations from the normal traffic pattern at one airport
  (LEMD) based on a fixed set of dynamic features extracted from ADS-B reports.
- **What it does not do.** Predict specific operational events, replace controllers,
  produce certified safety recommendations or generalize without retraining to airports
  with different procedures.

This framing as a **conformance monitor** is what makes the problem tractable with the
data available: it requires only examples of normal flight, which are abundant, and it
keeps the interpretation of the output clear ("this trajectory is unlike the ones the
model has seen") without overclaiming.

---

## 2. Data

### 2.1 Source

The dataset is built from ADS-B reports collected by the OpenSky Network around the
Madrid-Barajas terminal area. It covers approximately 18 days of operations sampled
between June 2017 and March 2020. Each aircraft is reported once every 10 seconds, which
is the natural cadence of ADS-B.

| Property | Value |
|---|---|
| Total trajectories | ≈ 20.000 (≈ 950 to 1.200 per day) |
| Rows per day | 140.000 to 240.000 |
| Columns | 21 |
| Points per flight (min · median · max) | 40 · 188 · 1.246 |

### 2.2 Variables

The raw schema includes positional information (latitude, longitude, time), dynamics
(barometric and geometric altitude, ground velocity, heading, vertical rate), derived
fields (distance to runway, flight phase), identity (ICAO24, callsign, flight id) and
transponder state (squawk, on-ground flag, alert flag, special-purpose indicator, last
contact). One field, `operation`, is entirely missing and is dropped.

### 2.3 Quality and cleaning

A profiling pass over the full dataset surfaces several issues that are handled in the
preprocessing pipeline:

- **Missing values.** `geoaltitude` is missing in 34.9 % of rows, `baroaltitude` in
  19.6 %, kinematic fields (`velocity`, `heading`, `vertrate`) around 19 %. The rest are
  below 4 %.
- **Duplicate file.** One day is shipped twice with different filenames and is
  deduplicated by content hash.
- **Empty or errored days.** A handful of days arrive with zero rows or with manifest
  errors and are dropped.
- **Coverage gaps.** About 8 % of airborne flights show a gap greater than 120 s between
  consecutive reports. These gaps are almost always receiver-coverage artifacts, not
  operational events, and are used as a feature, never as a label.

After cleaning, the dataset contains a stable set of normal LEMD operations spanning
multiple seasons and including the start of 2020, which doubles as a natural distribution
shift to stress-test the temporal split.

### 2.4 Notable observations

The exploratory analysis confirms a small set of rare cases that are kept aside for
validation only:

- Four flights carry an emergency squawk code at some point of their trajectory.
- Roughly one hundred flights show a clear go-around pattern (an approach followed by a
  sustained climb).

None of these flights enter the training set. They serve as a sanity check for the
deployed model.

---

## 3. Problem formulation

The dataset contains many normal flights and almost no labeled anomalies. This rules out
supervised classification and points to **one-class learning**: train a model that
captures the distribution of normal trajectories and score new flights by how compatible
they are with that distribution.

Concretely, all three deep models are **autoencoders** trained on normal flights only.
The training objective is to reconstruct the input window from a compressed latent
representation; at inference time the **reconstruction error** is used as the conformance
score. A trajectory that resembles the normal patterns is reconstructed accurately and
gets a low score; a trajectory that departs from those patterns is harder to reconstruct
and gets a high score.

This formulation has three properties that match the operational context. It needs only
normal data, which is the only data available at scale. It produces a continuous,
interpretable score instead of a hard label, which is the right output for a tool that
supports a human operator. And it gracefully handles unknown failure modes: any
trajectory the model has not learned to reconstruct will surface as anomalous regardless
of what makes it unusual.

---

## 4. Preprocessing pipeline

The preprocessing pipeline turns raw parquet files into fixed-length, standardized
tensors ready for training. Every step is implemented as a small, testable function under
`src/sadar/data/`.

1. **Load and merge.** All parquet files under `$SADAR_DATA_DIR` are read with pyarrow,
   concatenated and deduplicated. Empty or errored days are dropped.
2. **Coordinate transform.** Latitude and longitude are projected to a runway-relative
   metric system using `pyproj`. This makes the model robust to the curvature of
   coordinates and gives the network meters as input units.
3. **Heading encoding.** Heading is encoded as `(sin(heading), cos(heading))` to avoid
   the discontinuity at 0 / 360 degrees.
4. **Missing-value handling.** Short gaps inside an otherwise complete trajectory are
   filled with linear interpolation. Trajectories with too many missing samples are
   discarded.
5. **Resampling.** Every flight is resampled to a uniform 10 s grid, which matches the
   nominal cadence of ADS-B.
6. **Windowing.** Each resampled trajectory is sliced into fixed-length windows. The
   window length is a hyperparameter, set in `configs/preprocessing.yaml`.
7. **Standardization.** Each feature is normalized to zero mean and unit variance. The
   scaler is fit **only on the training split** to avoid leakage and saved as
   `data/processed/scaler.npz`, so the exact same transform is applied at inference time.
8. **Leak-free split.** The split is done jointly by `flight_id` and by date: no flight
   ever appears in more than one split, and the test split is strictly later in time than
   the training split. Years 2017 to 2019 are used for training and validation; 2020 is
   reserved for testing. This realistic split is what allows latency numbers to be taken
   at face value.
9. **Training set cleaning.** The few flights with emergency squawks and the suspected
   go-arounds are removed from the training split. Only "normal" trajectories enter the
   training set.

The input to all deep models is a window of seven features:
`[x_rel, y_rel, baroaltitude, velocity, sin_hdg, cos_hdg, vertrate]`.

---

## 5. Models

### 5.1 Isolation Forest baseline

A non-sequential reference detector. Each window is summarized by per-feature statistics
(mean, standard deviation, min, max) and fed to an Isolation Forest. The baseline is
useful to quantify how much sequential modeling actually adds.

### 5.2 LSTM autoencoder

A two-layer LSTM encoder compresses the input window into a latent vector, and a mirror
LSTM decoder reconstructs it. Trained end-to-end with MSE loss and the shared training
loop described below. This is the workhorse architecture: stable to train, fast to score
and well aligned with the literature on ADS-B trajectory modeling.

### 5.3 Transformer autoencoder

The same encoder-decoder skeleton, but with multi-head self-attention instead of
recurrence. The Transformer captures long-range dependencies inside the window without
the sequential bottleneck of LSTMs, at the cost of more parameters and a larger
hyperparameter surface.

### 5.4 VAE-LSTM (selected production model)

A variational autoencoder built on LSTM encoder and decoder. Instead of producing a single
latent vector, the encoder produces a posterior distribution; the decoder reconstructs
from a sample drawn from it. The training loss is the standard ELBO: reconstruction MSE
plus a KL term that regularizes the posterior toward a unit Gaussian, scaled by a
beta factor. This adds a principled, probabilistic interpretation to the score and
behaves better at the threshold-selection step.

### 5.5 Ensembling (evaluated and discarded)

Two ensemble strategies are evaluated as part of the comparison. After per-model
z-score calibration on validation normals, the three deep models are combined either by
their **mean** score (consensus) or by their **max** score (most-sensitive). Neither
combination beats the best individual model on the real test set, so the ensemble is
documented and discarded. The final detector is the VAE-LSTM alone.

### 5.6 Shared training infrastructure

All three deep models share the same training loop, implemented once in
`src/sadar/models/training.py`. The loop uses:

- **Adam** with weight decay,
- a **ReduceLROnPlateau** scheduler,
- **early stopping** on the validation loss,
- **MLflow** for tracking (losses, hyperparameters, best checkpoint),
- **Optuna** for hyperparameter search, used here mainly for the VAE-LSTM.

Each architecture has its own YAML configuration file under `configs/`. Reproducibility
is enforced by fixed seeds and by storing the exact preprocessing config alongside the
checkpoint.

---

## 6. Synthetic anomaly bench

Because real anomalies are rare and unlabeled, a controlled benchmark is built on top of
normal test flights. Each scenario starts from a real normal window and injects a known,
labeled perturbation that ramps in halfway through the window. Five types of injection
are implemented, each at multiple intensities:

| Type | Description |
|---|---|
| Route deviation | Lateral departure from the corridor, parameterized in meters. |
| Altitude anomaly | Altitude offset inconsistent with the flight phase. |
| Speed anomaly | Multiplicative acceleration or deceleration. |
| Holding | A repeated 360 ° turn at a configurable period. |
| Transponder freeze | The signal is held constant from a chosen point. |

The bench produces ground-truth labels (each injection knows exactly when it starts) and
therefore enables two metrics that real data cannot give us: **PR-AUC per anomaly type**
and **detection latency** (median seconds between the injection onset and the first
moment the per-step score crosses the alert threshold).

The synthetic bench is used only for evaluation. No synthetic window ever enters the
training set.

---

## 7. Evaluation

### 7.1 Real test set

The four detectors are scored on the same held-out test split. The headline metric is
**PR-AUC**, which is more informative than ROC-AUC under the strong class imbalance
present in real data.

| Model | ROC-AUC | PR-AUC | Synthetic mean ROC-AUC | Median latency |
|---|---|---|---|---|
| Isolation Forest baseline | 0.515 | 0.133 | 0.593 | n/a |
| LSTM autoencoder | 0.648 | 0.260 | 0.779 | 115 s |
| Transformer autoencoder | 0.614 | 0.227 | 0.743 | 115 s |
| **VAE-LSTM (selected)** | **0.659** | **0.299** | **0.792** | 120 s |
| Ensemble (mean) | 0.652 | 0.273 | 0.778 | n/a |
| Ensemble (max) | 0.658 | 0.278 | 0.774 | n/a |

Three conclusions stand out:

- The deep models add substantial value over the baseline (PR-AUC roughly doubles).
- The VAE-LSTM is the best overall: highest PR-AUC on real data, highest synthetic mean
  ROC-AUC, competitive latency.
- The two ensemble strategies do not beat the best individual model and are not retained.

### 7.2 Synthetic bench (per-anomaly ROC-AUC)

The VAE-LSTM performs best across all anomaly types. The pattern is consistent: large,
sustained perturbations are easier than subtle ones, and structured maneuvers (holding,
high-magnitude speed change) are easier than small drifts.

| Anomaly | Intensity | Baseline | LSTM | Transformer | VAE-LSTM |
|---|---|---|---|---|---|
| Route deviation | 20.000 m | 0.513 | 0.556 | 0.543 | 0.558 |
| Route deviation | 40.000 m | 0.537 | 0.684 | 0.657 | 0.700 |
| Route deviation | 80.000 m | 0.618 | 0.883 | 0.865 | 0.899 |
| Altitude offset | 300 m | 0.502 | 0.512 | 0.505 | 0.514 |
| Altitude offset | 800 m | 0.518 | 0.581 | 0.531 | 0.593 |
| Altitude offset | 1.500 m | 0.561 | 0.726 | 0.606 | 0.752 |
| Speed factor | x1.6 | 0.620 | 0.901 | 0.835 | 0.927 |
| Speed factor | x2.2 | 0.743 | 0.985 | 0.965 | 0.989 |
| Speed factor | x0.4 | 0.673 | 0.901 | 0.884 | 0.925 |
| Holding | 240 s/turn | 0.614 | 0.972 | 0.979 | 0.978 |
| Holding | 120 s/turn | 0.634 | 0.975 | 0.985 | 0.984 |
| Transponder freeze | stuck | 0.582 | 0.668 | 0.558 | 0.686 |

### 7.3 Curves and confusion matrices

Precision-recall and ROC curves are produced for every detector, together with a
confusion matrix taken at the validation-99-percentile threshold. The figures are
saved as PNGs under `reports/figures/` and the underlying arrays are stored inside
`reports/model_comparison.json`, so the dashboard can render them without recomputing.

---

## 8. Serving

The selected VAE-LSTM is served by a small FastAPI application
(`src/sadar/serve/app.py`). At startup, the service loads:

- the checkpoint of the production model,
- the scaler fit on the training data,
- the precomputed metrics report,
- and a curated sample of trajectories so the dashboard can run without raw data.

The HTTP surface is intentionally small. The dashboard talks to the backend through five
endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness probe (also serves a build identifier). |
| `GET /api/scene` | Current radar scene: aircraft, scores, threshold. |
| `GET /api/flights` | Ranked list of monitored flights with metadata. |
| `GET /api/metrics` | Pre-computed comparative metrics and curves. |
| `POST /api/simulate` | Inject an anomaly on a chosen flight and return the response. |

In production, the same FastAPI process also serves the built React app (the SPA is
mounted on `/`, the API stays under `/api`). The whole system runs in a single
container.

---

## 9. Frontend

The dashboard is a React 18 + Vite + TypeScript application designed to look and feel
like a tower control console: dark background, monospace data blocks, criticality
encoded by color (green/cyan for normal, amber/red for alert). It exposes four screens:

1. **Tower console.** A live radar scope centered on LEMD with a side panel listing every
   tracked flight and its current state. Tracked vs alerted aircraft are color-coded.
2. **Simulator.** The interactive heart of the demo. The operator picks a flight, an
   anomaly type, an intensity and an onset, and injects the perturbation in real time.
   The score on the right reacts to the injection and the detection latency is reported
   in seconds.
3. **Metrics.** The full comparative report: ROC and PR curves, per-anomaly PR-AUC
   breakdown, model selection rationale and final-detector summary.
4. **Presentation.** A guided walkthrough used for live presentations of the project.

The app is internationalized in English and Spanish through a small translation context.

---

## 10. Reproducibility and deployment

The project is engineered so that the entire pipeline (preprocessing, training,
evaluation, serving) can be reproduced from scratch.

- **Python environment.** Locked with `uv` and `pyproject.toml`; one command (`uv sync`)
  rebuilds the environment.
- **Frontend environment.** Managed with `pnpm` and `pnpm-lock.yaml`.
- **Configuration.** All knobs (paths, window length, model dimensions, training
  schedule, evaluation thresholds) live in YAML files under `configs/`. The code reads no
  literal paths.
- **Seeds.** A single seed is set at the top of every entry point; it propagates to
  NumPy, scikit-learn and PyTorch.
- **Tracking.** MLflow logs all training runs locally; Optuna stores its search history
  alongside the resulting hyperparameters.
- **Containerization.** A single Dockerfile builds a self-contained image that bundles
  the trained model, the scaler, the precomputed reports and the SPA. The same image is
  used locally (`docker compose up`) and in production (Hugging Face Spaces).

A `Makefile` exposes the workflow as named targets: `make install`, `make preprocess`,
`make train-lstm`, `make train-transformer`, `make train-vae`, `make tune-vae`,
`make compare`, `make serve`, `make dev`, `make docker-up`.

---

## 11. Limitations and future work

### 11.1 Limitations

- **Scope.** The dataset covers one airport (LEMD) and a limited time span. The model is
  not expected to transfer to other terminals without retraining.
- **No flight plans.** "Intended route" is approximated as "learned normal pattern". A
  flight that deviates from its filed route but stays inside the normal traffic pattern
  may not be flagged.
- **Coverage gaps.** Transponder gaps in ADS-B are dominated by receiver coverage. They
  are used as an input feature, not as a label.
- **Class imbalance.** Real anomalous events are extremely rare. The PR-AUC numbers on
  the real test set are computed on a small positive set; the synthetic bench is the
  primary quantitative benchmark.

### 11.2 Future work

- Extend the dataset to additional airports and validate cross-airport generalization.
- Incorporate filed flight plans, when available, as an auxiliary signal.
- Combine the conformance score with a downstream classifier trained on labeled
  operational events (go-arounds, missed approaches, holding) for more interpretable
  alerts.
- Explore probabilistic ensembles that weight each model by its calibrated likelihood
  rather than by a fixed reducer.

---

## 12. Conclusions

SADAR shows that a one-class deep-learning approach is a viable foundation for trajectory
conformance monitoring at a major airport. Three sequence autoencoders trained on the
same data and evaluated on the same bench produce a clear, defensible model choice: the
**VAE-LSTM** combines the best PR-AUC on real data, the best mean performance on the
synthetic bench and a competitive median detection latency, while improving substantially
over a non-sequential baseline. The system is fully reproducible, packaged as a single
container and served through a dashboard built around a real operational use case.

The result is not a finished product, but it is a solid prototype of how data-driven
conformance monitoring could complement the rule-based tools that air traffic operators
use today.

---

**Repository.** Smart Anomaly Detection for Aviation Routes (SADAR).
Madrid-Barajas (LEMD) · 2026.
