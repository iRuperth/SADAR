import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getScene, type InjectedFlight, type Scene, type SceneFlight } from "../api";
import RadarScope, { type ActiveAlert } from "../components/RadarScope";
import { classify } from "../components/classify";
import { useT } from "../i18n";

const SPEEDS = [1, 10, 30, 60];
const AIRLINES = ["IBE", "AEA", "RYR", "VLG", "AFR", "BAW", "DLH", "KLM", "EZY", "UAE", "QTR", "AAL"];

function newCallsign(seed: number): string {
  const airline = AIRLINES[seed % AIRLINES.length];
  const number = 100 + (seed * 37) % 8900;
  return `${airline}${number}`;
}

interface MonitorProps {
  injected: InjectedFlight | null;
  onClearInjected: () => void;
}

export default function Monitor({ injected, onClearInjected }: MonitorProps) {
  const t = useT();
  const [scene, setScene] = useState<Scene | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [simTime, setSimTime] = useState(120);
  const [speed, setSpeed] = useState(1);
  const [running, setRunning] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [flights, setFlights] = useState<SceneFlight[]>([]);
  const [alerts, setAlerts] = useState<ActiveAlert[]>([]);
  const respawnSeedRef = useRef(1000);
  const handleAlerts = useCallback((next: ActiveAlert[]) => {
    setAlerts((prev) => {
      if (prev.length !== next.length) return next;
      for (let i = 0; i < prev.length; i++) {
        if (prev[i].id !== next[i].id || prev[i].classification.kind !== next[i].classification.kind) return next;
      }
      return prev;
    });
  }, []);
  const rafRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  useEffect(() => {
    getScene(25)
      .then((data) => {
        setScene(data);
        setFlights(data.flights);
        if (data.flights.length) setSelectedId(data.flights[0].id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (!running) return;
    lastTickRef.current = performance.now();
    const step = (now: number) => {
      const dt = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      setSimTime((value) => value + dt * speed);
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    };
  }, [running, speed]);

  useEffect(() => {
    if (!flights.length) return;
    const updated = flights.map((flight) => {
      const endT = flight.start_offset + flight.path[flight.path.length - 1].t;
      if (simTime <= endT + 5) return flight;
      const delay = 30 + Math.random() * 60;
      const seed = respawnSeedRef.current++;
      return {
        ...flight,
        start_offset: simTime + delay,
        callsign: newCallsign(seed),
      };
    });
    if (updated.some((flight, i) => flight !== flights[i])) {
      setFlights(updated);
    }
  }, [simTime, flights]);

  const [injectedFlight, setInjectedFlight] = useState<SceneFlight | null>(null);
  const audioPlayedRef = useRef<number | null>(null);
  useEffect(() => {
    if (injected) {
      setInjectedFlight({ ...injected, start_offset: simTime + 5 });
      audioPlayedRef.current = null;
    } else {
      setInjectedFlight(null);
    }
  }, [injected]);

  useEffect(() => {
    if (!injectedFlight) return;
    if (audioPlayedRef.current === injectedFlight.id) return;
    const localT = simTime - injectedFlight.start_offset;
    if (localT < 0) return;
    audioPlayedRef.current = injectedFlight.id;
    const sequence = ["/atc/tower1.mp3", "/atc/plane.mp3", "/atc/tower2.mp3"];
    let idx = 0;
    const playNext = () => {
      if (idx >= sequence.length) return;
      const audio = new Audio(sequence[idx]);
      audio.volume = 0.85;
      audio.addEventListener("ended", () => {
        idx++;
        setTimeout(playNext, 600);
      });
      audio.play().catch(() => {});
    };
    playNext();
  }, [simTime, injectedFlight]);

  useEffect(() => {
    if (!injectedFlight) return;
    const endT = injectedFlight.start_offset + injectedFlight.path[injectedFlight.path.length - 1].t;
    if (simTime > endT + 10) {
      setInjectedFlight(null);
      onClearInjected();
    }
  }, [simTime, injectedFlight, onClearInjected]);

  const allFlights = useMemo(
    () => (injectedFlight ? [...flights, injectedFlight] : flights),
    [flights, injectedFlight],
  );

  const stepThreshold = scene?.step_threshold ?? 0;
  const stepSeconds = scene?.step_seconds ?? 10;
  const center = useMemo(() => scene?.center ?? { lat: 40.4936, lon: -3.5668 }, [scene]);

  if (error) {
    return <div className="status-alert">{t.monitor.offline}</div>;
  }
  if (!scene) {
    return <div className="label">LOADING SCENE...</div>;
  }

  const activeIds = new Set(
    allFlights
      .filter((flight) => {
        const localT = simTime - flight.start_offset;
        return localT >= 0 && localT <= flight.path[flight.path.length - 1].t;
      })
      .map((flight) => flight.id),
  );

  const selected = allFlights.find((flight) => flight.id === selectedId);

  const minutes = Math.floor(simTime / 60);
  const seconds = Math.floor(simTime % 60);
  const clock = `T+${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) 280px",
        gap: 0,
        height: "calc(100vh - 92px)",
      }}
    >
      <div style={{ position: "relative", borderRight: "1px solid var(--panel-edge)" }}>
        <RadarScope
          flights={allFlights}
          stepThreshold={stepThreshold}
          stepSeconds={stepSeconds}
          simTime={simTime}
          selectedId={selectedId}
          onSelect={setSelectedId}
          center={center}
          onAlerts={handleAlerts}
        />
        {alerts.length > 0 && (
          <div
            style={{
              position: "absolute",
              top: 116,
              left: 14,
              maxWidth: 280,
              display: "flex",
              flexDirection: "column",
              gap: 4,
              zIndex: 5,
            }}
          >
            {alerts.map((alert) => (
              <div
                key={alert.id}
                onClick={() => setSelectedId(alert.id)}
                className={alert.classification.critical ? "atc-blink" : ""}
                style={{
                  cursor: "pointer",
                  background: "rgba(14, 26, 31, 0.92)",
                  border: `1px solid ${alert.classification.critical ? "var(--emergency)" : "var(--alert)"}`,
                  padding: "6px 8px",
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: alert.classification.critical ? "var(--emergency)" : "var(--alert)",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                  <span style={{ fontWeight: 700 }}>{alert.callsign}</span>
                  <span>{alert.classification.short}</span>
                </div>
                <div style={{ marginTop: 2, color: "var(--label)", fontSize: 10 }}>
                  {alert.classification.label}
                </div>
                <div style={{ color: "var(--muted)", fontSize: 10 }}>
                  {alert.classification.reason}
                </div>
              </div>
            ))}
          </div>
        )}
        <div
          style={{
            position: "absolute",
            bottom: 12,
            left: 12,
            display: "flex",
            gap: 6,
            alignItems: "center",
            background: "rgba(14, 26, 31, 0.85)",
            border: "1px solid var(--panel-edge)",
            padding: "6px 10px",
            fontSize: 11,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          <span style={{ color: "var(--info)" }}>{clock}</span>
          <span style={{ color: "var(--panel-edge)" }}>|</span>
          <button onClick={() => setRunning((value) => !value)} style={{ padding: "2px 8px" }}>
            {running ? "PAUSE" : "PLAY"}
          </button>
          {SPEEDS.map((value) => (
            <button
              key={value}
              onClick={() => setSpeed(value)}
              style={{
                padding: "2px 8px",
                borderColor: speed === value ? "var(--info)" : "var(--panel-edge)",
                color: speed === value ? "var(--info)" : "var(--text)",
              }}
            >
              {value}X
            </button>
          ))}
          <button onClick={() => setSimTime(120)} style={{ padding: "2px 8px" }}>
            RESET
          </button>
        </div>
        {injectedFlight && (
          <div
            style={{
              position: "absolute",
              bottom: 12,
              right: 12,
              display: "flex",
              gap: 8,
              alignItems: "center",
              background: "rgba(14, 26, 31, 0.92)",
              border: "1px solid #d96bd9",
              padding: "6px 10px",
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "#d96bd9",
            }}
          >
            <span>SIM INJECTED · {injectedFlight.callsign}</span>
            <button
              onClick={() => {
                setInjectedFlight(null);
                onClearInjected();
              }}
              style={{ padding: "2px 8px", borderColor: "#d96bd9", color: "#d96bd9" }}
            >
              CLEAR
            </button>
          </div>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", minHeight: 0, background: "var(--panel)" }}>
        <div
          style={{
            padding: "8px 12px",
            borderBottom: "1px solid var(--panel-edge)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
          }}
        >
          <span className="label">FLIGHT STRIPS</span>
          <span className="label" style={{ color: "var(--info)" }}>{activeIds.size} ACTIVE</span>
        </div>
        <div style={{ overflowY: "auto", flex: 1, minHeight: 0 }}>
          {allFlights.map((flight) => {
            const localT = simTime - flight.start_offset;
            const onScope = localT >= 0 && localT <= flight.path[flight.path.length - 1].t;
            const stepIndex = Math.max(
              0,
              Math.min(flight.scores.length - 1, Math.floor(localT / stepSeconds)),
            );
            const score = onScope ? flight.scores[stepIndex] : 0;
            const alerting = onScope && score >= stepThreshold;
            const stateLabel = !onScope
              ? localT < 0
                ? "INBOUND"
                : "CLEARED"
              : alerting
                ? "ALERT"
                : "TRACK";
            const stateColor = !onScope
              ? "var(--muted)"
              : alerting
                ? "var(--alert)"
                : "var(--track-normal)";
            return (
              <div
                key={flight.id}
                className={`scope-row${flight.id === selectedId ? " scope-row--selected" : ""}`}
                style={{ gridTemplateColumns: "1fr 56px 56px", color: stateColor }}
                onClick={() => setSelectedId(flight.id)}
              >
                <span>{flight.callsign}</span>
                <span style={{ textAlign: "right", color: "var(--muted)" }}>WIN {String(flight.id).padStart(4, "0")}</span>
                <span style={{ textAlign: "right" }}>{stateLabel}</span>
              </div>
            );
          })}
        </div>
        {selected && (() => {
          const localT = simTime - selected.start_offset;
          const onScope = localT >= 0 && localT <= selected.path[selected.path.length - 1].t;
          let alt = 0;
          let speedKt = 0;
          let headingDeg = 0;
          let lat = 0;
          let lon = 0;
          let currentScore = 0;
          if (onScope) {
            const path = selected.path;
            let i = 0;
            for (; i < path.length - 1; i++) {
              if (localT >= path[i].t && localT < path[i + 1].t) break;
            }
            const a = path[i];
            const b = path[Math.min(i + 1, path.length - 1)];
            const span = b.t - a.t || 1;
            const f = (localT - a.t) / span;
            lat = a.lat + (b.lat - a.lat) * f;
            lon = a.lon + (b.lon - a.lon) * f;
            alt = a.alt + (b.alt - a.alt) * f;
            const lat1 = (a.lat * Math.PI) / 180;
            const lat2 = (b.lat * Math.PI) / 180;
            const dLon = ((b.lon - a.lon) * Math.PI) / 180;
            headingDeg = (
              (Math.atan2(
                Math.sin(dLon) * Math.cos(lat2),
                Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon),
              ) * 180) / Math.PI + 360
            ) % 360;
            const R = 6371;
            const dLatHav = ((b.lat - a.lat) * Math.PI) / 180;
            const h = Math.sin(dLatHav / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
            const kmStep = 2 * R * Math.asin(Math.sqrt(h));
            speedKt = (kmStep / 1.852) * (3600 / stepSeconds);
            const stepIndex = Math.max(0, Math.min(selected.scores.length - 1, Math.floor(localT / stepSeconds)));
            currentScore = selected.scores[stepIndex];
          }
          const lemdLat = (center.lat * Math.PI) / 180;
          const acLat = (lat * Math.PI) / 180;
          const dLonNm = ((lon - center.lon) * Math.PI) / 180;
          const dLatNm = ((lat - center.lat) * Math.PI) / 180;
          const hNm = Math.sin(dLatNm / 2) ** 2 + Math.cos(lemdLat) * Math.cos(acLat) * Math.sin(dLonNm / 2) ** 2;
          const distKm = onScope ? 2 * 6371 * Math.asin(Math.sqrt(hNm)) : 0;
          const distNm = distKm / 1.852;
          const peakScore = Math.max(...selected.scores);
          const altFt = Math.round((alt * 3.281) / 100) * 100;
          const fl = `FL${String(Math.round((alt * 3.281) / 1000) * 10 / 10).padStart(3, "0")}`;
          const sparkW = 240;
          const sparkH = 36;
          const maxS = Math.max(stepThreshold, ...selected.scores) * 1.1 || 1;
          const sparkPath = selected.scores
            .map((s, i) => {
              const sx = (i / (selected.scores.length - 1)) * sparkW;
              const sy = sparkH - (s / maxS) * sparkH;
              return `${i === 0 ? "M" : "L"}${sx.toFixed(1)},${sy.toFixed(1)}`;
            })
            .join(" ");
          const cursorIdx = onScope
            ? Math.max(0, Math.min(selected.scores.length - 1, Math.floor(localT / stepSeconds)))
            : -1;
          const cursorX = cursorIdx >= 0 ? (cursorIdx / (selected.scores.length - 1)) * sparkW : 0;
          const thrY = sparkH - (stepThreshold / maxS) * sparkH;
          const alerting = onScope && currentScore >= stepThreshold;
          const alertClass = alerting && cursorIdx >= 0 ? classify(selected.path, cursorIdx) : null;
          return (
            <div style={{ padding: "10px 12px", borderTop: "1px solid var(--panel-edge)", fontSize: 11 }}>
              <div className="label" style={{ marginBottom: 8, display: "flex", justifyContent: "space-between" }}>
                <span>SELECTED</span>
                <span style={{ color: alerting ? "var(--alert)" : onScope ? "var(--info)" : "var(--muted)" }}>
                  {onScope ? (alerting ? "ALERT" : "TRACK") : localT < 0 ? "INBOUND" : "CLEARED"}
                </span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3 }}>
                <span style={{ color: "var(--muted)" }}>CALLSIGN</span>
                <span style={{ textAlign: "right" }}>{selected.callsign}</span>
                <span style={{ color: "var(--muted)" }}>WINDOW</span>
                <span style={{ textAlign: "right" }}>{String(selected.id).padStart(5, "0")}</span>
                <span style={{ color: "var(--muted)" }}>ALTITUDE</span>
                <span style={{ textAlign: "right" }}>{onScope ? `${fl} (${altFt} FT)` : "---"}</span>
                <span style={{ color: "var(--muted)" }}>GROUND SPD</span>
                <span style={{ textAlign: "right" }}>{onScope ? `${Math.round(speedKt)} KT` : "---"}</span>
                <span style={{ color: "var(--muted)" }}>HEADING</span>
                <span style={{ textAlign: "right" }}>{onScope ? `${String(Math.round(headingDeg)).padStart(3, "0")}°` : "---"}</span>
                <span style={{ color: "var(--muted)" }}>DIST LEMD</span>
                <span style={{ textAlign: "right" }}>{onScope ? `${distNm.toFixed(1)} NM` : "---"}</span>
                <span style={{ color: "var(--muted)" }}>SCORE (NOW)</span>
                <span style={{ textAlign: "right", color: alerting ? "var(--alert)" : "var(--text)" }}>
                  {onScope ? currentScore.toFixed(3) : "---"}
                </span>
                <span style={{ color: "var(--muted)" }}>PEAK</span>
                <span style={{ textAlign: "right" }}>{peakScore.toFixed(3)}</span>
                <span style={{ color: "var(--muted)" }}>THRESHOLD</span>
                <span style={{ textAlign: "right" }}>{stepThreshold.toFixed(3)}</span>
                {alertClass && (
                  <>
                    <span style={{ color: "var(--muted)" }}>ALERT TYPE</span>
                    <span
                      style={{
                        textAlign: "right",
                        color: alertClass.critical ? "var(--emergency)" : "var(--alert)",
                      }}
                    >
                      {alertClass.short}
                    </span>
                    <span style={{ color: "var(--muted)" }}>SIGNATURE</span>
                    <span style={{ textAlign: "right", fontSize: 10 }}>{alertClass.label}</span>
                    <span style={{ color: "var(--muted)" }}>REASON</span>
                    <span style={{ textAlign: "right", fontSize: 10, color: "var(--label)" }}>
                      {alertClass.reason} (heuristic)
                    </span>
                  </>
                )}
              </div>
              <div className="label" style={{ marginTop: 10, marginBottom: 4 }}>SCORE / 10 MIN</div>
              <svg
                viewBox={`0 0 ${sparkW} ${sparkH}`}
                preserveAspectRatio="none"
                style={{ display: "block", width: "100%", height: sparkH, background: "var(--bg-deep)", border: "1px solid var(--panel-edge)" }}
              >
                <line x1={0} y1={thrY} x2={sparkW} y2={thrY} stroke="var(--alert)" strokeDasharray="3 3" strokeWidth={1} />
                <path d={sparkPath} fill="none" stroke="var(--info)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
                {cursorIdx >= 0 && (
                  <line x1={cursorX} y1={0} x2={cursorX} y2={sparkH} stroke="var(--track-selected)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                )}
              </svg>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
