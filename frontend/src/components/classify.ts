import type { PathPoint } from "../api";

export type AlertKind = "hijack" | "goaround" | "holding" | "altitude" | "route" | "anomaly";

export interface AlertClassification {
  kind: AlertKind;
  label: string;
  short: string;
  reason: string;
  critical: boolean;
}

const LEMD_LAT = 40.4936;
const LEMD_LON = -3.5668;

function distanceNm(lat: number, lon: number): number {
  const R = 6371;
  const dLat = ((lat - LEMD_LAT) * Math.PI) / 180;
  const dLon = ((lon - LEMD_LON) * Math.PI) / 180;
  const a = (LEMD_LAT * Math.PI) / 180;
  const b = (lat * Math.PI) / 180;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a) * Math.cos(b) * Math.sin(dLon / 2) ** 2;
  return (2 * R * Math.asin(Math.sqrt(h))) / 1.852;
}

function headingDeg(a: PathPoint, b: PathPoint): number {
  const lat1 = (a.lat * Math.PI) / 180;
  const lat2 = (b.lat * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

function angularDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % 360;
  if (d > 180) d = 360 - d;
  return d;
}

export function classify(path: PathPoint[], currentStep: number): AlertClassification {
  const window = path.slice(Math.max(0, currentStep - 5), currentStep + 1);
  if (window.length < 2) {
    return {
      kind: "anomaly",
      label: "UNCLASSIFIED ANOMALY",
      short: "ANOM",
      reason: "insufficient history",
      critical: false,
    };
  }

  const latVar = variance(window.map((p) => p.lat));
  const lonVar = variance(window.map((p) => p.lon));
  const altVar = variance(window.map((p) => p.alt));
  const frozenSteps = window.length;
  if (latVar < 1e-9 && lonVar < 1e-9 && altVar < 1e-2 && frozenSteps >= 3) {
    return {
      kind: "hijack",
      label: "POSSIBLE HIJACK SIGNATURE",
      short: "HIJK?",
      reason: `transponder frozen ${frozenSteps} steps`,
      critical: true,
    };
  }

  const altsM = window.map((p) => p.alt);
  const minAlt = Math.min(...altsM);
  const lastAlt = altsM[altsM.length - 1];
  const altRise = lastAlt - minAlt;
  if (minAlt < 1500 && altRise > 250 && altsM[altsM.length - 1] > altsM[altsM.length - 2]) {
    return {
      kind: "goaround",
      label: "GO-AROUND PATTERN",
      short: "GA",
      reason: `climb after low pass (${Math.round(minAlt)} m)`,
      critical: false,
    };
  }

  const headings: number[] = [];
  for (let i = 1; i < window.length; i++) headings.push(headingDeg(window[i - 1], window[i]));
  let totalTurn = 0;
  for (let i = 1; i < headings.length; i++) totalTurn += angularDiff(headings[i], headings[i - 1]);
  if (totalTurn > 180) {
    return {
      kind: "holding",
      label: "HOLDING PATTERN",
      short: "HOLD",
      reason: `cumulative turn ${Math.round(totalTurn)}°`,
      critical: false,
    };
  }

  const current = path[currentStep];
  const dist = distanceNm(current.lat, current.lon);
  if (dist < 5 && current.alt > 3000) {
    return {
      kind: "altitude",
      label: "ALTITUDE ANOMALY",
      short: "ALT!",
      reason: `${Math.round(current.alt)} m at ${dist.toFixed(1)} NM`,
      critical: false,
    };
  }
  if (dist > 20 && current.alt < 500) {
    return {
      kind: "altitude",
      label: "ALTITUDE ANOMALY",
      short: "ALT!",
      reason: `${Math.round(current.alt)} m at ${dist.toFixed(1)} NM`,
      critical: false,
    };
  }

  if (headings.length >= 2) {
    const meanHeading = circularMean(headings);
    const lastHeading = headings[headings.length - 1];
    if (angularDiff(meanHeading, lastHeading) > 30) {
      return {
        kind: "route",
        label: "ROUTE DEVIATION",
        short: "ROUTE",
        reason: `heading drift ${Math.round(angularDiff(meanHeading, lastHeading))}°`,
        critical: false,
      };
    }
  }

  return {
    kind: "anomaly",
    label: "UNCLASSIFIED ANOMALY",
    short: "ANOM",
    reason: "score above threshold, no clear signature",
    critical: false,
  };
}

function variance(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
}

function circularMean(degrees: number[]): number {
  let sx = 0;
  let sy = 0;
  for (const d of degrees) {
    sx += Math.cos((d * Math.PI) / 180);
    sy += Math.sin((d * Math.PI) / 180);
  }
  return ((Math.atan2(sy / degrees.length, sx / degrees.length) * 180) / Math.PI + 360) % 360;
}
