import type { PathPoint } from "../api";
import { projectTracks, toPath } from "./geo";

export interface RadarTrack {
  points: PathPoint[];
  color: string;
  label: string;
  dashed?: boolean;
}

export default function RadarPlot({ tracks, size = 380 }: { tracks: RadarTrack[]; size?: number }) {
  const projected = projectTracks(
    tracks.map((track) => track.points),
    size,
    size,
  );
  const center = size / 2;
  const maxRadius = size / 2 - 8;

  return (
    <div>
      <svg width={size} height={size} className="panel" style={{ display: "block" }}>
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
        <line x1={center} y1={8} x2={center} y2={size - 8} stroke="var(--panel-edge)" />
        <line x1={8} y1={center} x2={size - 8} y2={center} stroke="var(--panel-edge)" />
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
