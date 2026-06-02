import { useEffect, useRef, useState } from "react";

import type { PathPoint } from "../api";
import { projectTracks, toPath } from "./geo";

export interface RadarTrack {
  points: PathPoint[];
  color: string;
  label: string;
  dashed?: boolean;
}

const NARROW_WIDTH_THRESHOLD = 380;
const NARROW_HORIZONTAL_PADDING = 16;
const LEGEND_RESERVED_HEIGHT = 26;
const MIN_RADAR_DIM = 220;
const DEFAULT_RADAR_DIM = 380;

export default function RadarPlot({ tracks, size }: { tracks: RadarTrack[]; size?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [autoSize, setAutoSize] = useState(DEFAULT_RADAR_DIM);

  useEffect(() => {
    if (size != null || !ref.current) return;
    const node = ref.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width <= 0 || height <= 0) continue;
        const widthCap = width < NARROW_WIDTH_THRESHOLD ? width - NARROW_HORIZONTAL_PADDING : width;
        const next = Math.max(MIN_RADAR_DIM, Math.min(widthCap, height - LEGEND_RESERVED_HEIGHT));
        setAutoSize(next);
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [size]);

  const dim = size ?? autoSize;
  const projected = projectTracks(
    tracks.map((track) => track.points),
    dim,
    dim,
  );
  const center = dim / 2;
  const maxRadius = dim / 2 - 8;

  return (
    <div ref={ref} style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
      <svg width={dim} height={dim} className="panel" style={{ display: "block" }}>
        {[0.25, 0.5, 0.75, 1].map((ratio) => (
          <circle
            key={ratio}
            cx={center}
            cy={center}
            r={maxRadius * ratio}
            fill="none"
            stroke="var(--panel-edge)"
          />
        ))}
        <line x1={center} y1={8} x2={center} y2={dim - 8} stroke="var(--panel-edge)" />
        <line x1={8} y1={center} x2={dim - 8} y2={center} stroke="var(--panel-edge)" />
        {tracks.map((track, index) => (
          <path
            key={track.label}
            d={toPath(projected[index])}
            fill="none"
            stroke={track.color}
            strokeWidth={1.7}
            strokeDasharray={track.dashed ? "5 3" : undefined}
          />
        ))}
        {tracks.map((track, index) =>
          projected[index].length ? (
            <circle
              key={`start-${track.label}`}
              cx={projected[index][0].x}
              cy={projected[index][0].y}
              r={3}
              fill={track.color}
            />
          ) : null,
        )}
      </svg>
      <div style={{ display: "flex", gap: 18, marginTop: 8 }}>
        {tracks.map((track) => (
          <span key={track.label} className="label" style={{ color: track.color }}>
            {track.dashed ? "- - " : "___ "}
            {track.label}
          </span>
        ))}
      </div>
    </div>
  );
}
