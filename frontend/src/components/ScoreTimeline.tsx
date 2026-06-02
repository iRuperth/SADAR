import { useEffect, useRef, useState } from "react";

import { useT } from "../i18n";

interface Props {
  scores: number[];
  threshold: number;
  onsetIndex?: number;
  stepSeconds?: number;
  width?: number;
  height?: number;
}

const NARROW_TIMELINE_WIDTH = 420;
const NARROW_TIMELINE_PADDING = 22;
const DESKTOP_TIMELINE_PADDING = 28;

export default function ScoreTimeline({
  scores,
  threshold,
  onsetIndex,
  stepSeconds = 10,
  width,
  height,
}: Props) {
  const t = useT();
  const ref = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 600, h: 180 });

  useEffect(() => {
    if (width != null && height != null) return;
    if (!ref.current) return;
    const node = ref.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w <= 0 || h <= 0) continue;
        setBox({ w: Math.max(280, w), h: Math.max(140, h) });
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [width, height]);

  const w = width ?? box.w;
  const h = height ?? box.h;
  const pad = w < NARROW_TIMELINE_WIDTH ? NARROW_TIMELINE_PADDING : DESKTOP_TIMELINE_PADDING;
  const count = scores.length;
  const maxScore = Math.max(threshold, ...scores) * 1.15 || 1;
  const x = (i: number) => pad + (count > 1 ? (i / (count - 1)) * (w - 2 * pad) : 0);
  const y = (s: number) => h - pad - (s / maxScore) * (h - 2 * pad);
  const line = scores
    .map((s, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(s).toFixed(1)}`)
    .join(" ");
  const thresholdY = y(threshold);

  return (
    <div ref={ref} style={{ width: "100%", height: "100%" }}>
      <svg width={w} height={h} className="panel" style={{ display: "block" }}>
        <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="var(--panel-edge)" />
        <line
          x1={pad}
          y1={thresholdY}
          x2={w - pad}
          y2={thresholdY}
          stroke="var(--alert)"
          strokeDasharray="4 3"
        />
        <text x={w - pad} y={thresholdY - 5} textAnchor="end" fontSize={10} fill="var(--alert)">
          {t.timeline.threshold}
        </text>
        {onsetIndex != null && (
          <line
            x1={x(onsetIndex)}
            y1={pad}
            x2={x(onsetIndex)}
            y2={h - pad}
            stroke="var(--warn)"
            strokeDasharray="2 3"
          />
        )}
        <path d={line} fill="none" stroke="var(--info)" strokeWidth={1.7} />
        {scores.map((s, i) =>
          s >= threshold ? <circle key={i} cx={x(i)} cy={y(s)} r={2.4} fill="var(--alert)" /> : null,
        )}
        <text x={pad} y={h - 8} fontSize={10} fill="var(--muted)">
          0s
        </text>
        <text x={w - pad} y={h - 8} textAnchor="end" fontSize={10} fill="var(--muted)">
          {(count - 1) * stepSeconds}s
        </text>
      </svg>
    </div>
  );
}
