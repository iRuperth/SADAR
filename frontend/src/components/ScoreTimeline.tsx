import { useT } from "../i18n";

interface Props {
  scores: number[];
  threshold: number;
  onsetIndex?: number;
  stepSeconds?: number;
  width?: number;
  height?: number;
}

export default function ScoreTimeline({
  scores,
  threshold,
  onsetIndex,
  stepSeconds = 10,
  width = 600,
  height = 180,
}: Props) {
  const t = useT();
  const pad = 28;
  const count = scores.length;
  const maxScore = Math.max(threshold, ...scores) * 1.15 || 1;
  const x = (i: number) => pad + (count > 1 ? (i / (count - 1)) * (width - 2 * pad) : 0);
  const y = (s: number) => height - pad - (s / maxScore) * (height - 2 * pad);
  const line = scores
    .map((s, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(s).toFixed(1)}`)
    .join(" ");
  const thresholdY = y(threshold);

  return (
    <svg width={width} height={height} className="panel" style={{ display: "block" }}>
      <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="var(--panel-edge)" />
      <line
        x1={pad}
        y1={thresholdY}
        x2={width - pad}
        y2={thresholdY}
        stroke="var(--alert)"
        strokeDasharray="4 3"
      />
      <text x={width - pad} y={thresholdY - 5} textAnchor="end" fontSize={10} fill="var(--alert)">
        {t.timeline.threshold}
      </text>
      {onsetIndex != null && (
        <line
          x1={x(onsetIndex)}
          y1={pad}
          x2={x(onsetIndex)}
          y2={height - pad}
          stroke="var(--warn)"
          strokeDasharray="2 3"
        />
      )}
      <path d={line} fill="none" stroke="var(--info)" strokeWidth={1.7} />
      {scores.map((s, i) =>
        s >= threshold ? <circle key={i} cx={x(i)} cy={y(s)} r={2.4} fill="var(--alert)" /> : null,
      )}
      <text x={pad} y={height - 8} fontSize={10} fill="var(--muted)">
        0s
      </text>
      <text x={width - pad} y={height - 8} textAnchor="end" fontSize={10} fill="var(--muted)">
        {(count - 1) * stepSeconds}s
      </text>
    </svg>
  );
}
