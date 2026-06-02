import type { PathPoint } from "../api";

export interface XY {
  x: number;
  y: number;
}

export function projectTracks(
  tracks: PathPoint[][],
  width: number,
  height: number,
  pad = 28,
): XY[][] {
  const all = tracks.flat();
  if (all.length === 0) return tracks.map(() => []);

  const lat0 = all.reduce((sum, p) => sum + p.lat, 0) / all.length;
  const k = Math.cos((lat0 * Math.PI) / 180);

  const xs = all.map((p) => p.lon * k);
  const ys = all.map((p) => p.lat);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1e-6;
  const spanY = maxY - minY || 1e-6;
  const scale = Math.min((width - 2 * pad) / spanX, (height - 2 * pad) / spanY);
  const offX = (width - scale * spanX) / 2;
  const offY = (height - scale * spanY) / 2;

  return tracks.map((track) =>
    track.map((p) => ({
      x: offX + (p.lon * k - minX) * scale,
      y: height - (offY + (p.lat - minY) * scale),
    })),
  );
}

export function toPath(points: XY[]): string {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}
