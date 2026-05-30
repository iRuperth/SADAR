interface Props {
  scores: number[];
  stepThreshold: number;
  latency?: number | null;
}

export default function AlertBanner({ scores, stepThreshold, latency }: Props) {
  const peak = scores.length ? Math.max(...scores) : 0;
  const alert = peak >= stepThreshold;
  return (
    <div
      className="panel"
      style={{
        padding: "14px 18px",
        borderColor: alert ? "var(--alert)" : "var(--normal)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <span
        className={alert ? "status-alert" : "status-normal"}
        style={{ fontSize: 18, letterSpacing: "0.24em" }}
      >
        {alert ? "ALERT" : "NOMINAL"}
      </span>
      <span className="label">
        peak {peak.toFixed(3)} / thr {stepThreshold.toFixed(3)}
        {alert && latency != null ? ` / latency ${latency.toFixed(0)}s` : ""}
      </span>
    </div>
  );
}
