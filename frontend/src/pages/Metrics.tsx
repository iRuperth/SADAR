import { useEffect, useMemo, useState } from "react";

import { getMetrics, type Metrics as MetricsData, type MetricRow } from "../api";
import { useT } from "../i18n";

interface AnomalyGroup {
  family: string;
  label: string;
  rows: { intensity: string; values: Record<string, number> }[];
}

function groupAnomalies(rows: MetricRow[]): AnomalyGroup[] {
  const families: Record<string, string> = {
    route_deviation: "ROUTE DEVIATION",
    altitude: "ALTITUDE BUST",
    speed: "SPEED ANOMALY",
    holding: "HOLDING TURNS",
    freeze: "TRANSPONDER CUT",
  };
  const grouped: Record<string, AnomalyGroup> = {};
  if (rows.length === 0) return [];
  const keys = Object.keys(rows[0].synthetic_per_type);
  for (const key of keys) {
    const family = Object.keys(families).find((f) => key.startsWith(f));
    if (!family) continue;
    const intensity = key.slice(family.length).trim() || "—";
    if (!grouped[family]) {
      grouped[family] = { family, label: families[family], rows: [] };
    }
    const values: Record<string, number> = {};
    for (const row of rows) values[row.model] = row.synthetic_per_type[key];
    grouped[family].rows.push({ intensity, values });
  }
  return Object.values(grouped);
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function delta(target: number, baseline: number): string {
  if (baseline <= 0) return "—";
  const change = ((target - baseline) / baseline) * 100;
  const sign = change >= 0 ? "+" : "";
  return `${sign}${change.toFixed(0)}%`;
}

const METHODOLOGY = [
  { k: "TRAIN SPLIT", v: "2017 – 2019 / 61,008 windows" },
  { k: "VALIDATION", v: "2020 Q1 / 7,679 windows" },
  { k: "TEST (HELD-OUT)", v: "2020 Q1–Q2 / 7,788 windows" },
  { k: "WINDOW LENGTH", v: "60 steps × 10 s = 10 min" },
  { k: "FEATURES (7)", v: "x_rel, y_rel, alt, velocity, sin/cos hdg, vertrate" },
  { k: "SCALING", v: "Standard scaler fit on train only" },
  { k: "THRESHOLD RULE", v: "99th percentile of validation error" },
  { k: "SYN. SAMPLE", v: "2,000 windows × 12 anomaly variants" },
  { k: "FRAMEWORK", v: "PyTorch (MPS) · Optuna · MLflow" },
];

const MODEL_NOTES: Record<string, string> = {
  Baseline: "Isolation Forest on summary features. Reference to prove DL adds value.",
  LSTM: "Sequence-to-sequence LSTM autoencoder. Strong baseline for time series.",
  Transformer: "Self-attention encoder/decoder. Captures long-range dependencies.",
  "VAE-LSTM": "Variational LSTM autoencoder. Probabilistic, principled threshold.",
};

export default function Metrics() {
  const t = useT();
  const [data, setData] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics().then(setData).catch((reason) => setError(String(reason)));
  }, []);

  const groups = useMemo(() => (data ? groupAnomalies(data.results) : []), [data]);

  if (error) return <div className="status-alert">{t.monitor.offline}</div>;
  if (!data || data.results.length === 0) return <div className="label">{t.metrics.none}</div>;

  const baseline = data.results.find((row) => row.model === "Baseline");
  const winner = data.results.find((row) => row.model === data.selected_model) ?? data.results[0];
  const maxPr = Math.max(...data.results.map((row) => row.real_pr_auc));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px", gap: 16, alignItems: "start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 16, minWidth: 0 }}>
        <div className="panel" style={{ padding: 18 }}>
          <div className="label">FINAL MODEL</div>
          <div
            style={{
              fontSize: 32,
              letterSpacing: "0.1em",
              color: "var(--normal)",
              marginTop: 4,
              fontWeight: 700,
            }}
          >
            {winner.model.toUpperCase()}
          </div>
          <div style={{ color: "var(--label)", fontSize: 11, marginTop: 6, maxWidth: 720 }}>
            Selected after a head-to-head evaluation of 4 detectors on identical preprocessing, splits and metrics.
            Held-out PR-AUC and synthetic robustness drive the choice; the other three remain as documented baselines.
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 12 }}>
          {[
            { label: "REAL ROC-AUC", value: pct(winner.real_roc_auc), hint: "anomaly separability" },
            { label: "REAL PR-AUC", value: pct(winner.real_pr_auc), hint: "rare-class precision/recall" },
            { label: "SYNTHETIC ROC", value: pct(winner.synthetic_mean_roc_auc), hint: "mean across 12 variants" },
            {
              label: "vs BASELINE",
              value: baseline ? delta(winner.real_pr_auc, baseline.real_pr_auc) : "—",
              hint: "PR-AUC improvement over IF",
            },
          ].map((kpi) => (
            <div key={kpi.label} className="panel" style={{ padding: 14 }}>
              <div className="label">{kpi.label}</div>
              <div style={{ fontSize: 26, color: "var(--info)", marginTop: 4, letterSpacing: "0.05em" }}>{kpi.value}</div>
              <div style={{ color: "var(--muted)", fontSize: 10, marginTop: 4 }}>{kpi.hint}</div>
            </div>
          ))}
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span className="label">HEAD-TO-HEAD COMPARISON</span>
            <span className="label" style={{ color: "var(--muted)" }}>
              held-out test · 2020 · {data.results.length} detectors
            </span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10, fontFamily: "var(--mono)" }}>
            <thead>
              <tr className="label" style={{ color: "var(--label)" }}>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>MODEL</th>
                <th style={{ textAlign: "right", padding: "6px 8px" }}>REAL ROC</th>
                <th style={{ textAlign: "right", padding: "6px 8px" }}>REAL PR-AUC</th>
                <th style={{ textAlign: "right", padding: "6px 8px" }}>SYN ROC</th>
                <th style={{ textAlign: "right", padding: "6px 8px" }}>Δ vs BASELINE</th>
                <th style={{ width: "26%", padding: "6px 8px" }}>PR-AUC</th>
              </tr>
            </thead>
            <tbody>
              {data.results.map((row) => {
                const chosen = row.model === data.selected_model;
                return (
                  <tr
                    key={row.model}
                    style={{
                      borderTop: "1px solid var(--panel-edge)",
                      background: chosen ? "rgba(127, 209, 198, 0.06)" : undefined,
                      color: chosen ? "var(--normal)" : "var(--text)",
                    }}
                  >
                    <td style={{ padding: "10px 8px" }}>
                      <div style={{ fontSize: 13, letterSpacing: "0.06em" }}>
                        {row.model}
                        {chosen ? "  ★" : ""}
                      </div>
                      <div style={{ color: "var(--muted)", fontSize: 10, marginTop: 2 }}>
                        {MODEL_NOTES[row.model] ?? ""}
                      </div>
                    </td>
                    <td style={{ textAlign: "right", padding: "10px 8px" }}>{pct(row.real_roc_auc)}</td>
                    <td style={{ textAlign: "right", padding: "10px 8px" }}>{pct(row.real_pr_auc)}</td>
                    <td style={{ textAlign: "right", padding: "10px 8px" }}>{pct(row.synthetic_mean_roc_auc)}</td>
                    <td
                      style={{
                        textAlign: "right",
                        padding: "10px 8px",
                        color: baseline && row.real_pr_auc >= baseline.real_pr_auc ? "var(--info)" : "var(--muted)",
                      }}
                    >
                      {baseline ? delta(row.real_pr_auc, baseline.real_pr_auc) : "—"}
                    </td>
                    <td style={{ padding: "10px 8px" }}>
                      <div style={{ height: 10, background: "var(--panel-edge)" }}>
                        <div
                          style={{
                            height: 10,
                            width: `${(row.real_pr_auc / maxPr) * 100}%`,
                            background: chosen ? "var(--normal)" : "var(--info)",
                          }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <span className="label">SYNTHETIC ANOMALY PERFORMANCE</span>
            <span className="label" style={{ color: "var(--muted)" }}>ROC-AUC by anomaly family and intensity</span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 10, fontFamily: "var(--mono)" }}>
            <thead>
              <tr className="label" style={{ color: "var(--label)" }}>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>ANOMALY</th>
                <th style={{ textAlign: "left", padding: "6px 8px" }}>INTENSITY</th>
                {data.results.map((row) => (
                  <th
                    key={row.model}
                    style={{
                      textAlign: "right",
                      padding: "6px 8px",
                      color: row.model === data.selected_model ? "var(--normal)" : "var(--label)",
                    }}
                  >
                    {row.model}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((group) =>
                group.rows.map((row, idx) => (
                  <tr
                    key={`${group.family}-${row.intensity}`}
                    style={{ borderTop: idx === 0 ? "1px solid var(--panel-edge)" : "1px solid rgba(26,47,54,0.4)" }}
                  >
                    <td style={{ padding: "6px 8px", color: idx === 0 ? "var(--text)" : "var(--muted)" }}>
                      {idx === 0 ? group.label : ""}
                    </td>
                    <td style={{ padding: "6px 8px", color: "var(--label)", fontSize: 10 }}>{row.intensity}</td>
                    {data.results.map((mdl) => {
                      const value = row.values[mdl.model];
                      const isWinner =
                        Math.max(...Object.values(row.values)) === value && data.results.length > 1;
                      return (
                        <td
                          key={mdl.model}
                          style={{
                            textAlign: "right",
                            padding: "6px 8px",
                            color: mdl.model === data.selected_model
                              ? "var(--normal)"
                              : isWinner
                                ? "var(--info)"
                                : "var(--text)",
                          }}
                        >
                          {pct(value)}
                        </td>
                      );
                    })}
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>

      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div className="panel" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 10 }}>METHODOLOGY</div>
          <div style={{ display: "grid", gap: 8 }}>
            {METHODOLOGY.map((entry) => (
              <div key={entry.k} style={{ borderTop: "1px solid var(--panel-edge)", paddingTop: 6 }}>
                <div style={{ color: "var(--muted)", fontSize: 10, letterSpacing: "0.1em" }}>{entry.k}</div>
                <div style={{ fontSize: 12, marginTop: 2 }}>{entry.v}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 10 }}>WHY THIS MODEL</div>
          <div style={{ fontSize: 12, color: "var(--text)", lineHeight: 1.55 }}>
            The <span style={{ color: "var(--normal)" }}>{winner.model}</span> reaches the highest
            held-out PR-AUC ({pct(winner.real_pr_auc)}) while maintaining strong synthetic robustness
            ({pct(winner.synthetic_mean_roc_auc)} mean ROC). Probabilistic latent space gives a
            principled threshold and uncertainty estimates, which matters more than raw ROC when
            anomalies are rare.
          </div>
        </div>

        <div className="panel" style={{ padding: 16 }}>
          <div className="label" style={{ marginBottom: 10 }}>METRIC PRIMER</div>
          <div style={{ fontSize: 11, color: "var(--label)", lineHeight: 1.55 }}>
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: "var(--info)" }}>ROC-AUC</span> — probability a random anomaly
              scores higher than a random normal. 50% = random, 100% = perfect.
            </div>
            <div style={{ marginBottom: 6 }}>
              <span style={{ color: "var(--info)" }}>PR-AUC</span> — area under
              precision/recall. More informative than ROC when anomalies are rare (our case).
            </div>
            <div>
              <span style={{ color: "var(--info)" }}>HELD-OUT</span> — test data the model never
              saw during training or hyperparameter tuning. The honest measure of generalization.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
