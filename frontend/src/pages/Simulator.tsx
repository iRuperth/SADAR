import { useEffect, useRef, useState } from "react";

import {
  getFlight,
  getFlights,
  simulate,
  type FlightDetail,
  type FlightSummary,
  type InjectedFlight,
  type SimulationResult,
} from "../api";
import AlertBanner from "../components/AlertBanner";
import RadarPlot from "../components/RadarPlot";
import ScoreTimeline from "../components/ScoreTimeline";
import { useT } from "../i18n";

interface SimulatorProps {
  onInject: (flight: InjectedFlight) => void;
  hasInjected: boolean;
  onClearInjected: () => void;
}

interface KindConfig {
  id: string;
  min: number;
  max: number;
  step: number;
  def: number;
  unit: string;
}

const KINDS: KindConfig[] = [
  { id: "route_deviation", min: 0, max: 80000, step: 2000, def: 40000, unit: "m" },
  { id: "altitude", min: 0, max: 2500, step: 100, def: 1200, unit: "m" },
  { id: "speed", min: 0.3, max: 2.6, step: 0.1, def: 2.2, unit: "x" },
  { id: "holding", min: 60, max: 300, step: 20, def: 160, unit: "s/turn" },
  { id: "freeze", min: 0, max: 0, step: 1, def: 0, unit: "" },
];

const MOBILE_BREAKPOINT_QUERY = "(max-width: 900px)";
const SLIDER_TRACK_HEIGHT = 26;
const INJECT_BUTTON_PADDING_MOBILE = "14px";
const INJECT_BUTTON_PADDING_DESKTOP = "10px";
const INJECT_BUTTON_FONT_MOBILE = 13;

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(MOBILE_BREAKPOINT_QUERY).matches;
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia(MOBILE_BREAKPOINT_QUERY);
    const handler = (event: MediaQueryListEvent) => setIsMobile(event.matches);
    setIsMobile(mql.matches);
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);
  return isMobile;
}

export default function Simulator({ onInject, hasInjected, onClearInjected }: SimulatorProps) {
  const t = useT();
  const isMobile = useIsMobile();
  const [bases, setBases] = useState<FlightSummary[]>([]);
  const [baseId, setBaseId] = useState<number | null>(null);
  const [baseDetail, setBaseDetail] = useState<FlightDetail | null>(null);
  const [kind, setKind] = useState("speed");
  const [magnitude, setMagnitude] = useState(2.2);
  const [onset, setOnset] = useState(0.5);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const simSeedRef = useRef(1);

  useEffect(() => {
    getFlights(20, "typical")
      .then((rows) => {
        setBases(rows);
        if (rows.length) setBaseId(rows[0].id);
      })
      .catch((reason) => setError(String(reason)));
  }, []);

  useEffect(() => {
    if (baseId == null) return;
    getFlight(baseId)
      .then(setBaseDetail)
      .catch((reason) => setError(String(reason)));
  }, [baseId]);

  useEffect(() => {
    if (baseId == null) return;
    simulate({ id: baseId, kind, magnitude, onset })
      .then(setResult)
      .catch((reason) => setError(String(reason)));
  }, [baseId, kind, magnitude, onset]);

  if (error) {
    return <div className="status-alert">{t.monitor.offline}</div>;
  }

  const config = KINDS.find((entry) => entry.id === kind)!;

  function changeKind(id: string) {
    setKind(id);
    setMagnitude(KINDS.find((entry) => entry.id === id)!.def);
  }

  const controlsPanel = (
    <div className="panel" style={{ padding: 16, display: "grid", gap: 16, alignContent: "start", overflowY: "auto", minHeight: 0 }}>
      <div>
        <div className="label">{t.simulator.baseFlight}</div>
        <select
          value={baseId ?? ""}
          onChange={(event) => setBaseId(Number(event.target.value))}
          style={{ width: "100%", marginTop: 6, background: "var(--bg-deep)", color: "var(--text)", border: "1px solid var(--panel-edge)", padding: 8, fontFamily: "var(--mono)" }}
        >
          {bases.map((flight) => (
            <option key={flight.id} value={flight.id}>
              WIN {String(flight.id).padStart(5, "0")} ({flight.score.toFixed(2)})
            </option>
          ))}
        </select>
      </div>

      <div>
        <div className="label">{t.simulator.injectedAnomaly}</div>
        <div style={{ display: "grid", gap: 4, marginTop: 6 }}>
          {KINDS.map((entry) => (
            <button
              key={entry.id}
              onClick={() => changeKind(entry.id)}
              style={{ textAlign: "left", borderColor: kind === entry.id ? "var(--alert)" : "var(--panel-edge)", color: kind === entry.id ? "var(--alert)" : "var(--text)" }}
            >
              {t.simulator.kinds[entry.id]}
            </button>
          ))}
        </div>
      </div>

      {config.max > config.min && (
        <div>
          <div className="label">
            {t.simulator.intensity}: {magnitude}
            {config.unit}
          </div>
          <input
            type="range"
            min={config.min}
            max={config.max}
            step={config.step}
            value={magnitude}
            onChange={(event) => setMagnitude(Number(event.target.value))}
            style={{ width: "100%", marginTop: 8, accentColor: "var(--alert)", height: SLIDER_TRACK_HEIGHT }}
          />
        </div>
      )}

      <div>
        <div className="label">
          {t.simulator.onset}: {(onset * 100).toFixed(0)}
          {t.simulator.onsetSuffix}
        </div>
        <input
          type="range"
          min={0.1}
          max={0.8}
          step={0.05}
          value={onset}
          onChange={(event) => setOnset(Number(event.target.value))}
          style={{ width: "100%", marginTop: 8, accentColor: "var(--warn)", height: SLIDER_TRACK_HEIGHT }}
        />
      </div>

      <div
        style={{
          borderTop: "1px solid var(--panel-edge)",
          paddingTop: 14,
          display: "flex",
          flexDirection: "column",
          gap: 8,
          position: isMobile ? "sticky" : "static",
          bottom: isMobile ? 0 : "auto",
          background: isMobile ? "var(--bg-panel)" : "transparent",
          boxShadow: isMobile ? "0 -8px 16px rgba(0, 0, 0, 0.45)" : "none",
          zIndex: 2,
        }}
      >
        <button
          disabled={!result}
          onClick={() => {
            if (!result) return;
            const seed = simSeedRef.current++;
            const injected: InjectedFlight = {
              id: 100000 + seed,
              callsign: "SDR001",
              path: result.path,
              scores: result.scores,
              anomalous: true,
              start_offset: 0,
              injected: true,
              kind: result.kind,
            };
            onInject(injected);
          }}
          style={{
            padding: isMobile ? INJECT_BUTTON_PADDING_MOBILE : INJECT_BUTTON_PADDING_DESKTOP,
            fontSize: isMobile ? INJECT_BUTTON_FONT_MOBILE : undefined,
            borderColor: "#d96bd9",
            color: "#d96bd9",
            fontWeight: 700,
            letterSpacing: "0.12em",
          }}
        >
          {"> INJECT INTO MONITOR"}
        </button>
        {hasInjected && (
          <button
            onClick={onClearInjected}
            style={{ padding: "6px", borderColor: "#d96bd9", color: "#d96bd9" }}
          >
            CLEAR INJECTED
          </button>
        )}
        <div style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.04em", lineHeight: 1.4 }}>
          Appears in the Monitor as <span style={{ color: "#d96bd9" }}>SDR001</span> within 5 s.
          Replaces any previous injection.
        </div>
      </div>
    </div>
  );

  const radarColumn = (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, minHeight: 0, minWidth: 0 }}>
      {result && (
        <AlertBanner
          scores={result.scores}
          stepThreshold={result.step_threshold}
          latency={result.latency_seconds}
        />
      )}
      {result && baseDetail && (
        <div className="panel" style={{ flex: 1, minHeight: 320, padding: 10 }}>
          <RadarPlot
            tracks={[
              { points: baseDetail.path, color: "var(--muted)", label: t.simulator.original, dashed: true },
              { points: result.path, color: "var(--alert)", label: t.simulator.injected },
            ]}
          />
        </div>
      )}
    </div>
  );

  const timelineColumn = (
    <div style={{ display: "flex", flexDirection: "column", minHeight: 260, minWidth: 0 }}>
      <div className="label" style={{ marginBottom: 6 }}>{t.monitor.scoreTitle}</div>
      {result && (
        <div style={{ flex: 1, minHeight: 220 }}>
          <ScoreTimeline
            scores={result.scores}
            threshold={result.step_threshold}
            onsetIndex={result.onset_index}
          />
        </div>
      )}
    </div>
  );

  if (isMobile) {
    return (
      <div className="simulator-layout">
        {radarColumn}
        {controlsPanel}
        {timelineColumn}
      </div>
    );
  }

  return (
    <div className="simulator-layout">
      {controlsPanel}
      {radarColumn}
      {timelineColumn}
    </div>
  );
}
