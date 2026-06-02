import { useEffect, useMemo, useRef, useState } from "react";

import type { PathPoint, SceneFlight } from "../api";
import { classify, type AlertClassification } from "./classify";

export interface ActiveAlert {
  id: number;
  callsign: string;
  classification: AlertClassification;
}

const KM_PER_NM = 1.852;
const SCOPE_RANGE_NM = 40;
const SCOPE_RANGE_KM = SCOPE_RANGE_NM * KM_PER_NM;
const COMPACT_SCOPE_WIDTH = 600;
const TINY_SCOPE_WIDTH = 420;

const LEMD_RUNWAYS: Array<{ name: string; lat1: number; lon1: number; lat2: number; lon2: number }> = [
  { name: "18R/36L", lat1: 40.531799, lon1: -3.574850, lat2: 40.492599, lon2: -3.574630 },
  { name: "18L/36R", lat1: 40.532600, lon1: -3.559380, lat2: 40.501099, lon2: -3.559210 },
  { name: "14L/32R", lat1: 40.494900, lon1: -3.557870, lat2: 40.470001, lon2: -3.532580 },
  { name: "14R/32L", lat1: 40.484901, lon1: -3.576010, lat2: 40.455700, lon2: -3.546380 },
];

interface Props {
  flights: ExtendedFlight[];
  stepThreshold: number;
  stepSeconds: number;
  simTime: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
  center: { lat: number; lon: number };
  onAlerts?: (alerts: ActiveAlert[]) => void;
}

export interface ExtendedFlight extends SceneFlight {
  highlighted?: boolean;
}

interface Projected {
  x: number;
  y: number;
}

interface ActiveAircraft {
  flight: ExtendedFlight;
  lat: number;
  lon: number;
  alt: number;
  speedKt: number;
  headingDeg: number;
  score: number;
  alert: boolean;
  classification: AlertClassification | null;
  history: { lat: number; lon: number }[];
}

function project(
  lat: number,
  lon: number,
  center: { lat: number; lon: number },
  width: number,
  height: number,
  pxPerKm: number,
): Projected {
  const dLat = lat - center.lat;
  const dLon = lon - center.lon;
  const kmY = dLat * 111.32;
  const kmX = dLon * 111.32 * Math.cos((center.lat * Math.PI) / 180);
  return {
    x: width / 2 + kmX * pxPerKm,
    y: height / 2 - kmY * pxPerKm,
  };
}

function interpolatePoint(path: PathPoint[], localT: number): {
  point: PathPoint;
  next: PathPoint;
  fraction: number;
} | null {
  if (path.length === 0) return null;
  if (localT <= path[0].t) return { point: path[0], next: path[Math.min(1, path.length - 1)], fraction: 0 };
  if (localT >= path[path.length - 1].t) return null;
  for (let i = 0; i < path.length - 1; i++) {
    if (localT >= path[i].t && localT < path[i + 1].t) {
      const span = path[i + 1].t - path[i].t || 1;
      const fraction = (localT - path[i].t) / span;
      const point: PathPoint = {
        lat: path[i].lat + (path[i + 1].lat - path[i].lat) * fraction,
        lon: path[i].lon + (path[i + 1].lon - path[i].lon) * fraction,
        alt: path[i].alt + (path[i + 1].alt - path[i].alt) * fraction,
        t: localT,
      };
      return { point, next: path[i + 1], fraction };
    }
  }
  return null;
}

function headingBetween(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  const brng = (Math.atan2(y, x) * 180) / Math.PI;
  return (brng + 360) % 360;
}

function kmBetween(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export default function RadarScope({
  flights,
  stepThreshold,
  stepSeconds,
  simTime,
  selectedId,
  onSelect,
  center,
  onAlerts,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });

  useEffect(() => {
    if (!containerRef.current) return;
    const node = containerRef.current;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 0 && h > 0) setSize({ w, h });
      }
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const pxPerKm = useMemo(() => {
    const usable = Math.min(size.w, size.h) - 60;
    return usable / 2 / SCOPE_RANGE_KM;
  }, [size.w, size.h]);

  const active: ActiveAircraft[] = useMemo(() => {
    const list: ActiveAircraft[] = [];
    for (const flight of flights) {
      const localT = simTime - flight.start_offset;
      const interp = interpolatePoint(flight.path, localT);
      if (!interp) continue;
      const stepIndex = Math.max(0, Math.min(flight.scores.length - 1, Math.floor(localT / stepSeconds)));
      const score = flight.scores[stepIndex];
      const alert = score >= stepThreshold;
      const headingDeg = headingBetween(interp.point, interp.next);
      const distKm = kmBetween(interp.point, interp.next);
      const speedKt = (distKm / KM_PER_NM) * (3600 / stepSeconds);
      const history: { lat: number; lon: number }[] = [];
      const steps = 6;
      for (let i = 1; i <= steps; i++) {
        const past = interpolatePoint(flight.path, localT - i * stepSeconds);
        if (past) history.push({ lat: past.point.lat, lon: past.point.lon });
      }
      const classification = alert ? classify(flight.path, stepIndex) : null;
      list.push({
        flight,
        lat: interp.point.lat,
        lon: interp.point.lon,
        alt: interp.point.alt,
        speedKt,
        headingDeg,
        score,
        alert,
        classification,
        history,
      });
    }
    return list;
  }, [flights, simTime, stepThreshold, stepSeconds]);

  useEffect(() => {
    if (!onAlerts) return;
    const alerts: ActiveAlert[] = active
      .filter((a) => a.alert && a.classification)
      .map((a) => ({ id: a.flight.id, callsign: a.flight.callsign, classification: a.classification! }));
    onAlerts(alerts);
  }, [active, onAlerts]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#102125";
    ctx.fillRect(0, 0, size.w, size.h);

    const cx = size.w / 2;
    const cy = size.h / 2;

    ctx.strokeStyle = "#183137";
    ctx.lineWidth = 1;
    for (let nm = 5; nm <= SCOPE_RANGE_NM; nm += 5) {
      const r = nm * KM_PER_NM * pxPerKm;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.strokeStyle = "#286058";
    ctx.fillStyle = "#286058";
    ctx.font = "9px 'B612 Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const outerR = SCOPE_RANGE_NM * KM_PER_NM * pxPerKm;
    for (let deg = 0; deg < 360; deg += 5) {
      const rad = ((deg - 90) * Math.PI) / 180;
      const inner = deg % 30 === 0 ? outerR - 14 : deg % 10 === 0 ? outerR - 8 : outerR - 4;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(rad) * inner, cy + Math.sin(rad) * inner);
      ctx.lineTo(cx + Math.cos(rad) * outerR, cy + Math.sin(rad) * outerR);
      ctx.stroke();
      if (deg % 30 === 0) {
        const lx = cx + Math.cos(rad) * (outerR - 26);
        const ly = cy + Math.sin(rad) * (outerR - 26);
        ctx.fillText(String(deg).padStart(3, "0"), lx, ly);
      }
    }

    ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
    ctx.lineWidth = 1.4;
    LEMD_RUNWAYS.forEach((rw) => {
      const a = project(rw.lat1, rw.lon1, center, size.w, size.h, pxPerKm);
      const b = project(rw.lat2, rw.lon2, center, size.w, size.h, pxPerKm);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    ctx.strokeStyle = "rgba(127, 209, 198, 0.18)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    LEMD_RUNWAYS.forEach((rw) => {
      const a = project(rw.lat1, rw.lon1, center, size.w, size.h, pxPerKm);
      const b = project(rw.lat2, rw.lon2, center, size.w, size.h, pxPerKm);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const ext = 10 * KM_PER_NM * pxPerKm;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(a.x - (dx / len) * ext, a.y - (dy / len) * ext);
      ctx.moveTo(b.x, b.y);
      ctx.lineTo(b.x + (dx / len) * ext, b.y + (dy / len) * ext);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    const lemd = project(center.lat, center.lon, center, size.w, size.h, pxPerKm);
    ctx.fillStyle = "#7fd1c6";
    ctx.font = "10px 'B612 Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("LEMD", lemd.x + 6, lemd.y - 6);

    active.forEach((a) => {
      const isInjected = a.flight.id >= 100000;
      a.history.forEach((h, i) => {
        const p = project(h.lat, h.lon, center, size.w, size.h, pxPerKm);
        const op = (1 - i / a.history.length) * 0.7;
        ctx.fillStyle = isInjected
          ? `rgba(217, 107, 217, ${op})`
          : a.alert
            ? `rgba(224, 128, 128, ${op})`
            : `rgba(193, 218, 205, ${op})`;
        ctx.fillRect(p.x - 1.5, p.y - 1.5, 3, 3);
      });
    });

    active.forEach((a) => {
      const p = project(a.lat, a.lon, center, size.w, size.h, pxPerKm);
      const selected = a.flight.id === selectedId;
      const isInjected = a.flight.id >= 100000;
      const color = isInjected
        ? "#d96bd9"
        : a.alert
          ? "#e08080"
          : selected
            ? "#ffffff"
            : a.flight.highlighted
              ? "#e6a23c"
              : "#c1dacd";
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = selected ? 1.6 : 1.2;
      const r = selected ? 4 : 3;
      ctx.strokeRect(p.x - r, p.y - r, r * 2, r * 2);
      const rad = ((a.headingDeg - 90) * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + Math.cos(rad) * 12, p.y + Math.sin(rad) * 12);
      ctx.stroke();
    });
  }, [size, pxPerKm, active, center, selectedId]);

  const compact = size.w < COMPACT_SCOPE_WIDTH;
  const tiny = size.w < TINY_SCOPE_WIDTH;
  const blockOffsetX = compact ? 8 : 12;
  const blockOffsetY = compact ? 4 : 6;
  const blockFontSize = compact ? 8 : 10;
  const blockLineHeight = compact ? 1.05 : 1.2;
  const compassSize = compact ? 48 : 64;
  const compassCenter = compassSize / 2;
  const compassRadius = compassSize / 2 - 4;
  const compassTop = tiny ? 14 : 36;

  return (
    <div ref={containerRef} className="scope-root">
      <canvas ref={canvasRef} className="scope-canvas" style={{ width: size.w, height: size.h }} />
      {active.map((a) => {
        const p = project(a.lat, a.lon, center, size.w, size.h, pxPerKm);
        const isInjected = a.flight.id >= 100000;
        const cls = ["data-block"];
        if (isInjected) cls.push("data-block--injected");
        if (a.flight.id === selectedId) cls.push("data-block--selected");
        if (a.alert && !isInjected) cls.push("data-block--alert", "atc-blink");
        if (a.classification?.critical && !isInjected) cls.push("data-block--emergency");
        const fl = `FL${String(Math.round(a.alt / 100 * 3.281 / 10)).padStart(3, "0")}`;
        const spd = `${String(Math.round(a.speedKt)).padStart(3, "0")}`;
        const tagLine = a.alert ? (a.classification?.short ?? "ALERT") : "LEMD";
        return (
          <div
            key={a.flight.id}
            className={cls.join(" ")}
            style={{
              left: p.x + blockOffsetX,
              top: p.y + blockOffsetY,
              fontSize: blockFontSize,
              lineHeight: blockLineHeight,
            }}
            onClick={() => onSelect(a.flight.id)}
          >
            {a.flight.callsign}{"\n"}
            {fl} {spd}{"\n"}
            {tagLine}
          </div>
        );
      })}
      {!tiny && (
        <div className="scope-hud" style={{ top: 10, left: 14 }}>
          LEMD ACC // RANGE {SCOPE_RANGE_NM} NM
        </div>
      )}
      <div
        className="scope-hud"
        style={{ top: compassTop, left: 14, pointerEvents: "none" }}
      >
        <svg width={compassSize} height={compassSize} style={{ display: "block" }}>
          <circle cx={compassCenter} cy={compassCenter} r={compassRadius} fill="none" stroke="var(--panel-edge)" />
          <line x1={compassCenter} y1={compassCenter - compassRadius + 2} x2={compassCenter} y2={compassCenter + compassRadius - 2} stroke="var(--map-feature)" />
          <line x1={compassCenter - compassRadius + 2} y1={compassCenter} x2={compassCenter + compassRadius - 2} y2={compassCenter} stroke="var(--map-feature)" />
          <polygon
            points={`${compassCenter},${compassCenter - compassRadius - 2} ${compassCenter - 4},${compassCenter - compassRadius + 8} ${compassCenter},${compassCenter - compassRadius + 5} ${compassCenter + 4},${compassCenter - compassRadius + 8}`}
            fill="var(--info)"
          />
          <text x={compassCenter} y={compassCenter - compassRadius + 14} textAnchor="middle" fontSize={9} fill="var(--info)" fontFamily="var(--mono)">N</text>
          <text x={compassCenter} y={compassCenter + compassRadius - 4} textAnchor="middle" fontSize={9} fill="var(--label)" fontFamily="var(--mono)">S</text>
          <text x={compassCenter + compassRadius - 4} y={compassCenter + 3} textAnchor="middle" fontSize={9} fill="var(--label)" fontFamily="var(--mono)">E</text>
          <text x={compassCenter - compassRadius + 6} y={compassCenter + 3} textAnchor="middle" fontSize={9} fill="var(--label)" fontFamily="var(--mono)">W</text>
        </svg>
      </div>
      <div className="scope-hud" style={{ top: 10, right: 14, textAlign: "right" }}>
        TRACKS {active.length}
      </div>
    </div>
  );
}
