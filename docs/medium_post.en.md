# When the most complex model is not the one that detects best

*A comparison of LSTM, Transformer and VAE-LSTM autoencoders over 19,000 real flights at Madrid-Barajas. And why validation loss is a poor advisor.*

![Control tower with air traffic over the runway at night.](../docs/assets/tower-night.png)

> **Code:** [github.com/iRuperth/SADAR](https://github.com/iRuperth/SADAR)
> **Live demo:** [huggingface.co/spaces/devrup404/sadar](https://huggingface.co/spaces/devrup404/sadar)
>
> The repository is public and open to collaborations, suggestions and forks. Any technical discussion, *issue* or *pull request* is welcome.

---

The most interesting data point in the dataset is not a hijacking. It is an aircraft that pretended to be one for thirty seconds, by mistake.

On July 31, 2017, an Iberia Airbus flew over Madrid with the hijack code active on its transponder during three points of its trajectory. Then it went back to normal. The flight was IBE2845, the transmitted code was 7500, and in the secondary surveillance system that code means unlawful interference.

It was not a hijacking. It was almost certainly a transponder fault: three points out of the 171 that make up the complete flight record, 800 meters from the runway, with no other operational anomaly. That flight, logged in a parquet file of 3.4 million rows, marks the beginning of this article. And it also marks the technical question I took into the project: can a model learn what a normal flight looks like and raise a flag when one stops looking like one?

This article presents SADAR, an anomaly detection system for flight trajectories trained on eighteen days of real traffic from Adolfo Suárez Madrid-Barajas airport. It compares three deep autoencoders against a classical baseline, evaluates the system on a synthetic anomaly bench and discusses with candor where it works and where it does not.

Let us start with the problem.

## Problem statement

Incident detection in civil aviation can rarely be framed as a supervised problem: the data is dominated by normal operation and the few labeled events are ambiguous. The standard alternative is **conformance monitoring**: learn the distribution of normal behavior and flag trajectories that deviate from it. This is the framing SADAR adopts.

The distinction matters. Saying "this flight deviates from the learned pattern" is defensible with data. Saying "this flight is an incident" requires operational context that ADS-B does not provide. SADAR operates strictly under the first statement.

## Data

The data comes from OpenSky Network, a collaborative network of ADS-B receivers. The set covers 18 non-consecutive days between June 2017 and March 2020, centered on the airspace of Barajas. Each aircraft reports a position every 10 seconds, with 21 columns per point: position and dynamics (latitude, longitude, barometric and geometric altitude, velocity, heading, vertical rate), identity (callsign, icao24, flight_id) and transponder state (squawk, alert, onground).

The raw figures after merging and deduplication:

- 3,429,638 rows.
- 19,057 distinct flights.
- Median of 175 points per flight, maximum 1,606.
- Median duration of 30.2 minutes.

The first finding of the exploratory analysis was the gaps. GPS altitude (`geoaltitude`) shows 24.79 % missing values. Velocity and heading, around 13 %. These gaps are not errors: they correspond to areas of lower ground receiver coverage. I discarded GPS altitude and kept barometric altitude, more complete, treating the gaps as a property of the problem rather than a fault to correct.

![Sampled trajectories over the airspace of LEMD. The approach and departure corridors are visually identifiable before any projection. In red, the points of the four flights with an emergency squawk.](../docs/assets/eda/trajectories_map.png)

The second finding came from the heading histogram. Plotted on a polar chart, the two parallel runways of Barajas appear as two opposite lobes.

![Polar histogram of heading over the full dataset. The two opposite lobes correspond to the parallel runways of Barajas. This circular property motivated the encoding of heading as the (sin, cos) pair.](../docs/assets/eda/heading_polar.png)

The airport is identifiable in the data before projecting it onto a map.

That circular property of heading motivated its encoding as a sine and cosine pair: for an autoencoder, 359° and 1° must remain adjacent, and a linear encoding prevents that.

![Distribution of rows by flight phase in the dataset. The descent phase dominates, followed by climb and ground operations, which conditions which patterns the model learns best.](../docs/assets/eda/flight_phases.png)

## Real events in the set

The dataset contains only four flights with an emergency code on the transponder. They deserve detailed description because their scarcity determines the framing of the problem:

- **IBE2845** (Iberia, July 31, 2017). Squawk 7500 for three points out of 171, transmitted 800 meters from the runway. Consistent with a transponder fault.
- **ELY395** (El Al, October 2, 2017). Squawk 7700 for five points, declared 800 meters from the runway. The aircraft landed. The most plausible case of a real emergency.
- **RYR61AD** (Ryanair, January 28, 2019). Squawk 7600 (communications failure). One point. Landed.
- **SWR202Y** (Swiss, February 3, 2020). Squawk 7700. One point. Landed.

Four labels in a universe of non-labels. This proportion rules out supervised learning: no classifier is trained on four positives. The four flights were set aside as a qualitative validation set, grouped in 1,076 labeled windows in `data/processed/anomalies.npy`, and do not appear at any point of training.

## Methodology

### One-class learning

Without enough labels, the framing is **one-class learning** through autoencoders. The network learns to reconstruct normal trajectories. The reconstruction error operates as an anomaly score: an input similar to those in training produces a low error; an unusual input raises it.

I trained four detectors under identical conditions (same preprocessing, same split, same threshold) to allow a rigorous comparison:

| # | Model | Idea | Weights |
|---|---|---|---|
| 0 | Isolation Forest | Per-window summary statistics, not sequential | baseline |
| 1 | LSTM autoencoder | Sequential reconstruction | 224 KB |
| 2 | Transformer autoencoder | Attention over the full window | 635 KB |
| 3 | VAE-LSTM | Probabilistic encoding of the latent space | 222 KB |

### Preprocessing

Four preprocessing decisions deserve specific mention because they determine the final quality of a trajectory detector.

**Metric coordinates relative to the runway.** Latitude and longitude in degrees are inadequate for modeling movement over an airport: distances depend on latitude, and the model does not incorporate geodesy. The pipeline projects coordinates to UTM zone 30 (EPSG:32630), centered on the runway of Barajas, and operates in meters.

**Heading as the (sin, cos) pair.** A circular variable that linear encoding does not represent faithfully. Trigonometric encoding ensures that values close on the circle are also close in feature space.

**Split by date and by flight_id.** Training on 2017-2019, validation and test on 2020. Each flight appears in a single split. This blocks both temporal leakage and instance leakage simultaneously.

**Cleaning of the training set.** I excluded from training the 100 flights with a go-around pattern and the four with an emergency squawk. The training set must represent the most conservative version of "normal flight". Leaving go-arounds in makes the model learn to reconstruct them without difficulty, nullifying their value as anomalies.

The final windows have 60 steps of 10 seconds (10 minutes per window), with 50 % overlap. Seven features per step:

```
[x_rel, y_rel, baroaltitude, velocity, sin_hdg, cos_hdg, vertrate]
```

The resulting set: 61,008 training windows, 7,679 validation, 7,788 test and 1,076 labeled as anomalous. The scaler standardizes features by fitting only on the training set.

### Architectures

**Isolation Forest** over summary statistics (mean, standard deviation, minimum and maximum per feature). A non-sequential baseline whose function is to establish the reference line that deep learning must surpass.

**LSTM autoencoder**: encoder with a single LSTM layer (hidden 64), linear bottleneck to a latent of dimension 16, symmetric decoder that unrolls from the repeated latent.

**Transformer autoencoder**: two encoder layers with four attention heads (d_model 64), sinusoidal positional encoding, same bottleneck to latent 16.

**VAE-LSTM**: LSTM encoder that produces the mean and log-variance, reparameterization so that gradients flow through the sampling, LSTM decoder from z. The loss combines reconstruction error (MSE) with the Kullback-Leibler divergence weighted by a beta coefficient of 0.0001193, selected by Optuna.

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

All models share the same training loop: Adam optimizer with weight decay, ReduceLROnPlateau scheduler, early stopping with patience 7, fixed seed 42, complete tracking with MLflow.

### Synthetic anomaly bench

Four real events do not allow stable quantitative evaluation. To address this limitation, controlled anomalies are injected over normal flights from the test set. Five families, all with a linear ramp at onset to allow the measurement of **detection latency** (seconds from the start of the perturbation to the first threshold crossing):

- **Route deviation**: lateral deformation, magnitudes of 20, 40 and 80 km.
- **Altitude anomaly**: offset of 300, 800 or 1,500 m.
- **Speed anomaly**: multiplicative factors of 0.4x, 1.6x and 2.2x.
- **Holding**: continuous turns with a period of 120 or 240 seconds.
- **Transponder freeze**: the ADS-B retains the last value for the rest of the window.

Validation against an in-house simulator might seem circular. It is standard practice in *safety engineering*: the simulator was designed before the models and describes well-defined failure modes, not patterns learned from the model. The set produces 12 variants (combinations of type and intensity) and 24,000 synthetic windows per evaluated model.

So much for the setup. Next, whether it works.

## Results

Results on the 2020 test set:

| Model | Real ROC-AUC | Real PR-AUC | Synthetic ROC | Median latency |
|---|---|---|---|---|
| Isolation Forest | 0.515 | 0.133 | 0.593 | n/a |
| LSTM-AE | 0.648 | 0.260 | 0.779 | 115 s |
| Transformer-AE | 0.614 | 0.227 | 0.743 | 115 s |
| **VAE-LSTM** | **0.659** | **0.299** | **0.792** | 120 s |

The deep models nearly double the PR-AUC of the baseline. This difference justifies the investment in deep learning: if the LSTM had not beaten the Isolation Forest, the rest of the work would not have made sense.

![Precision-recall curve of the VAE-LSTM on the real test set. The PR-AUC of 0.299 doubles the baseline under heavy imbalance.](../reports/figures/pr_vae-lstm.png)

![ROC curve of the VAE-LSTM on the same set. ROC-AUC of 0.659.](../reports/figures/roc_vae-lstm.png)

One observation deserves attention.

The Transformer won the validation loss (0.038 against 0.053 for the VAE and 0.072 for the LSTM). And lost the PR-AUC. Something did not add up.

The explanation is direct: an architecture with enough capacity also reconstructs anomalies with some fidelity, which reduces the separability between normal and anomalous as measured by reconstruction error. In reconstruction-based detection, validation loss is a poor proxy for performance. We know it and keep looking at it anyway.

I also evaluated two ensembles, by mean and by maximum of the z-normalized scores. Neither beat the individual VAE (PR-AUC 0.273 and 0.278 against 0.299), and I discarded them. The ensemble was an attempt to close with a push. It did not deliver. It is honest to report it.

The numbers tell one part. The other lies in what does not show up in the table.

## Discussion

The results show a detection profile that is **heterogeneous by anomaly type**. The final model easily detects loud perturbations: ROC-AUC 0.984 in holding with a 120 s period, 0.989 in acceleration with factor 2.2, 0.899 in route deviation of 80 km. Subtle perturbations (300 m of altitude, small drifts) require close to two minutes and fall into the 0.51-0.55 range.

The most relevant case of the discussion is the **transponder freeze**.

The VAE detects it in 1.2 % of cases. One out of every eighty attempts.

The explanation is structural: an aircraft in stable cruise whose values freeze produces a pattern very close to that of an aircraft in stable cruise that keeps transmitting. The reconstruction error remains low and the score does not cross the threshold. It is a limitation of the reconstruction approach, not of training.

![Confusion matrix of the VAE-LSTM with the threshold set at the 99th percentile of validation. The balance favors precision over recall, consistent with a target false alarm rate of 1 %.](../reports/figures/cm_vae-lstm.png)

This heterogeneity defines the scope of the system. SADAR is a detector of **anomalous maneuvers**, not a universal incident detector. Its operational value lies in those deviations that the traditional rule-based network (systems such as STCA or MSAW) does not cover, not in replacing those systems.

Before closing, it is worth being explicit about what the system does not do.

## Limitations

The system assumes certain limitations documented explicitly in the technical report.

- **Absence of flight plans**. SADAR does not have the planned route for each flight. The "deviation from the expected route" is approximated by "deviation from the mean pattern". A flight that departs from its assigned plan but stays within the mean pattern will not be flagged.
- **Temporal coverage**. The set consists of 18 non-consecutive days between 2017 and 2020. Modeling demand, seasonality or continuous temporal prediction falls outside the scope.
- **Transponder freeze**. Within-window detection rate of 1.2 %. Any serious extension of the system must address this case specifically (derivative signals, change-point detection, dedicated model).
- **Coverage gaps**. 24.2 % of in-air flights present gaps longer than 120 seconds in reception. These gaps are receiver artifacts, not operational incidents. Treating them as positive labels would generate systematic noise.

## Deployed system

The final model is served through an API built on FastAPI that mounts a React + Vite + TypeScript frontend on the same port, allowing a *single-image* deployment on Hugging Face Spaces. The dashboard consists of three screens:

1. **Tower console**: live radar over Barajas with the day's flights and their scores, visual alert when one crosses the threshold.
2. **Simulator**: controlled injection of anomalies over a reference flight, with visualization of the deformed trajectory, the score and the latency.
3. **Metrics**: model comparison table, precision-recall and ROC curves, confusion matrices, breakdown by anomaly type.

## Stack and reproducibility

Python 3.11, PyTorch, FastAPI, MLflow and Optuna on the backend. Vite, React 18 and TypeScript on the frontend. Packaged in a *single-image* Docker container. Reproducibility guaranteed by fixed seeds, YAML configurations that resolve environment variables, and checkpoints stored also as MLflow artifacts.

## Conclusions and future work

SADAR shows that conformance monitoring of flight trajectories is tractable with deep autoencoders trained exclusively on normal operation, with defensible metrics and a reproducible evaluation protocol. The comparison between three model families under identical conditions allows the choice of VAE-LSTM to be justified not only by its final PR-AUC, but by its overall detection and latency profile.

The Transformer was not the right choice for this problem. I leave it documented because discarded options also count.

Immediate future work points to three directions. First, replace the Transformer with a TCN or a convolutional autoencoder to bound capacity without losing temporal modeling. Second, incorporate derivatives and change-point detection to address the transponder freeze. Third, evaluate the incorporation of flight plans when available, which would reframe the problem as plan-conditioned prediction rather than approximation to the mean pattern.

IBE2845 landed without incident on July 31, 2017 and its three red points received no further attention. Glitch, anomaly or incident, reality rarely arrives labeled, and building systems that operate in that terrain begins by accepting it.
