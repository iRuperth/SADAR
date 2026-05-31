import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import type { InjectedFlight } from "../api";
import DynamicBackdrop from "../components/DynamicBackdrop";
import LangToggle from "../components/LangToggle";
import { useT } from "../i18n";
import Metrics from "./Metrics";
import Monitor from "./Monitor";
import Simulator from "./Simulator";

type Tab = "monitor" | "simulator" | "metrics";

const TAB_IDS: Tab[] = ["monitor", "simulator", "metrics"];

function formatUtc(date: Date): string {
  const y = date.getUTCFullYear();
  const m = String(date.getUTCMonth() + 1).padStart(2, "0");
  const d = String(date.getUTCDate()).padStart(2, "0");
  const hh = String(date.getUTCHours()).padStart(2, "0");
  const mm = String(date.getUTCMinutes()).padStart(2, "0");
  const ss = String(date.getUTCSeconds()).padStart(2, "0");
  return `${y}-${m}-${d} · ${hh}:${mm}:${ss} UTC`;
}

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("monitor");
  const [injected, setInjected] = useState<InjectedFlight | null>(null);
  const [now, setNow] = useState(() => new Date());
  const t = useT();

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      <DynamicBackdrop light={false} />
      <div style={{ padding: "14px 56px", width: "100%", boxSizing: "border-box", position: "relative" }}>
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderBottom: "1px solid var(--panel-edge)",
            paddingBottom: 10,
            gap: 24,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 18, minWidth: 0 }}>
            <img
              src="/sadar-logo.png"
              alt="SADAR"
              style={{ height: 56, width: "auto", display: "block" }}
            />
            <div style={{ minWidth: 0 }}>
              <h1 style={{ margin: 0, fontSize: 18, letterSpacing: "0.08em", whiteSpace: "nowrap" }}>{t.appTitle}</h1>
            </div>
          </div>
          <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-end",
                gap: 2,
                paddingRight: 10,
                borderRight: "1px solid var(--panel-edge)",
                lineHeight: 1,
              }}
            >
              <div
                className="label"
                style={{ color: "#7fd1c6", fontSize: 10, letterSpacing: "0.18em" }}
              >
                STATUS · OPERATIONAL
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  color: "var(--text)",
                  letterSpacing: "0.06em",
                }}
              >
                {formatUtc(now)}
              </div>
            </div>
            {TAB_IDS.map((id) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                style={{
                  borderColor: tab === id ? "var(--info)" : "var(--panel-edge)",
                  color: tab === id ? "var(--info)" : "var(--text)",
                }}
              >
                {t.nav[id]}
              </button>
            ))}
            <Link to="/">
              <button>{t.nav.presentation}</button>
            </Link>
            <Link to="/presentation/final">
              <button style={{ borderColor: "var(--info)", color: "var(--info)" }}>
                {t.scenes.finalNav}
              </button>
            </Link>
            <LangToggle />
          </div>
        </header>

        <main style={{ marginTop: 16, position: "relative" }}>
          <div style={{ display: tab === "monitor" ? "block" : "none" }}>
            <Monitor injected={injected} onClearInjected={() => setInjected(null)} />
          </div>
          <div style={{ display: tab === "simulator" ? "block" : "none" }}>
            <Simulator
              onInject={(flight) => {
                setInjected(flight);
                setTab("monitor");
              }}
              hasInjected={injected != null}
              onClearInjected={() => setInjected(null)}
            />
          </div>
          <div style={{ display: tab === "metrics" ? "block" : "none" }}>
            <Metrics />
          </div>
          {(() => {
            const idx = TAB_IDS.indexOf(tab);
            const prev = TAB_IDS[(idx - 1 + TAB_IDS.length) % TAB_IDS.length];
            const next = TAB_IDS[(idx + 1) % TAB_IDS.length];
            return (
              <>
                <button
                  className="edge-nav edge-nav--left"
                  onClick={() => setTab(prev)}
                  aria-label={`Go to ${prev}`}
                >
                  <span className="edge-nav__chevron">{"<"}</span>
                  <span className="edge-nav__label">{t.nav[prev]}</span>
                </button>
                <button
                  className="edge-nav edge-nav--right"
                  onClick={() => setTab(next)}
                  aria-label={`Go to ${next}`}
                >
                  <span className="edge-nav__chevron">{">"}</span>
                  <span className="edge-nav__label">{t.nav[next]}</span>
                </button>
              </>
            );
          })()}
        </main>
      </div>
    </>
  );
}
