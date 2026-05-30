import { useState } from "react";
import { Link } from "react-router-dom";

import Metrics from "./Metrics";
import Monitor from "./Monitor";
import Simulator from "./Simulator";

type Tab = "monitor" | "simulator" | "metrics";

const TABS: { id: Tab; label: string }[] = [
  { id: "monitor", label: "Monitor" },
  { id: "simulator", label: "Simulator" },
  { id: "metrics", label: "Metrics" },
];

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("monitor");

  return (
    <>
      <div className="tower-backdrop" />
      <div style={{ padding: 24, maxWidth: 1320, margin: "0 auto", position: "relative" }}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          borderBottom: "1px solid var(--panel-edge)",
          paddingBottom: 12,
        }}
      >
        <div>
          <div className="label">SADAR // LEMD</div>
          <h1 style={{ margin: "4px 0 0", fontSize: 20, letterSpacing: "0.08em" }}>
            FLIGHT CONFORMANCE MONITOR
          </h1>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {TABS.map((entry) => (
            <button
              key={entry.id}
              onClick={() => setTab(entry.id)}
              style={{ borderColor: tab === entry.id ? "var(--info)" : "var(--panel-edge)", color: tab === entry.id ? "var(--info)" : "var(--text)" }}
            >
              {entry.label}
            </button>
          ))}
          <Link to="/presentation">
            <button>Presentation</button>
          </Link>
        </div>
      </header>

      <main style={{ marginTop: 24 }}>
        {tab === "monitor" && <Monitor />}
        {tab === "simulator" && <Simulator />}
        {tab === "metrics" && <Metrics />}
      </main>
      </div>
    </>
  );
}
