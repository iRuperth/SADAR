import { useEffect, useState } from "react";

import { getFlight, getFlights, type FlightDetail, type FlightSummary } from "../api";
import AlertBanner from "../components/AlertBanner";
import RadarPlot from "../components/RadarPlot";
import ScoreTimeline from "../components/ScoreTimeline";

export default function Monitor() {
  const [flights, setFlights] = useState<FlightSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<FlightDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getFlights(14)
      .then((rows) => {
        setFlights(rows);
        if (rows.length) setSelected(rows[0].id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (selected == null) return;
    getFlight(selected)
      .then(setDetail)
      .catch((reason) => setError(String(reason)));
  }, [selected]);

  if (error) {
    return <div className="status-alert">backend offline. start it with: make serve</div>;
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", gap: 20 }}>
      <div>
        <div className="label">Most anomalous windows</div>
        <div style={{ marginTop: 10, display: "grid", gap: 1, background: "var(--panel-edge)" }}>
          {flights.map((flight) => (
            <button
              key={flight.id}
              onClick={() => setSelected(flight.id)}
              style={{
                textAlign: "left",
                border: "none",
                padding: "8px 12px",
                background: flight.id === selected ? "var(--bg-deep)" : "var(--panel)",
                display: "flex",
                justifyContent: "space-between",
              }}
            >
              <span>WIN {String(flight.id).padStart(5, "0")}</span>
              <span className={flight.anomalous ? "status-alert" : "status-normal"}>
                {flight.score.toFixed(2)}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "grid", gap: 16 }}>
        {detail && <AlertBanner scores={detail.scores} stepThreshold={detail.step_threshold} />}
        <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
          {detail && (
            <RadarPlot
              tracks={[
                { points: detail.path, color: "var(--info)", label: "actual" },
                { points: detail.reconstructed, color: "var(--muted)", label: "reconstructed", dashed: true },
              ]}
            />
          )}
          {detail && <ScoreTimeline scores={detail.scores} threshold={detail.step_threshold} />}
        </div>
      </div>
    </div>
  );
}
