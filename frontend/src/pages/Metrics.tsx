import { useEffect, useState } from "react";

import { getMetrics, type Metrics as MetricsData } from "../api";
import { useT } from "../i18n";

export default function Metrics() {
  const t = useT();
  const [data, setData] = useState<MetricsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setData)
      .catch((reason) => setError(String(reason)));
  }, []);

  if (error) {
    return <div className="status-alert">{t.monitor.offline}</div>;
  }
  if (!data || data.results.length === 0) {
    return <div className="label">{t.metrics.none}</div>;
  }

  const maxPr = Math.max(...data.results.map((row) => row.real_pr_auc));

  return (
    <div style={{ maxWidth: 760 }}>
      <div className="label">{t.metrics.title}</div>
      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 12 }}>
        <thead>
          <tr className="label">
            <th style={{ textAlign: "left", padding: "6px 8px" }}>{t.metrics.model}</th>
            <th style={{ textAlign: "right", padding: "6px 8px" }}>{t.metrics.realRoc}</th>
            <th style={{ textAlign: "right", padding: "6px 8px" }}>{t.metrics.realPr}</th>
            <th style={{ textAlign: "right", padding: "6px 8px" }}>{t.metrics.synRoc}</th>
            <th style={{ width: 180 }} />
          </tr>
        </thead>
        <tbody>
          {data.results.map((row) => {
            const chosen = row.model === data.selected_model;
            return (
              <tr
                key={row.model}
                style={{ borderTop: "1px solid var(--panel-edge)", color: chosen ? "var(--normal)" : "var(--text)" }}
              >
                <td style={{ padding: "8px" }}>
                  {row.model}
                  {chosen ? " *" : ""}
                </td>
                <td style={{ textAlign: "right", padding: "8px" }}>{row.real_roc_auc.toFixed(3)}</td>
                <td style={{ textAlign: "right", padding: "8px" }}>{row.real_pr_auc.toFixed(3)}</td>
                <td style={{ textAlign: "right", padding: "8px" }}>
                  {row.synthetic_mean_roc_auc.toFixed(3)}
                </td>
                <td style={{ padding: "8px" }}>
                  <div style={{ height: 8, background: "var(--panel-edge)" }}>
                    <div
                      style={{
                        height: 8,
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
      <div className="label" style={{ marginTop: 14 }}>
        {t.metrics.selected}: <span className="status-normal">{data.selected_model}</span>
      </div>
    </div>
  );
}
