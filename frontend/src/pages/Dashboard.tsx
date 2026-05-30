import { useState } from "react";
import { Link } from "react-router-dom";

import LangToggle from "../components/LangToggle";
import { useT } from "../i18n";
import Metrics from "./Metrics";
import Monitor from "./Monitor";
import Simulator from "./Simulator";

type Tab = "monitor" | "simulator" | "metrics";

const TAB_IDS: Tab[] = ["monitor", "simulator", "metrics"];

export default function Dashboard() {
  const [tab, setTab] = useState<Tab>("monitor");
  const t = useT();

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
            <h1 style={{ margin: "4px 0 0", fontSize: 20, letterSpacing: "0.08em" }}>{t.appTitle}</h1>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
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
            <Link to="/presentation">
              <button>{t.nav.presentation}</button>
            </Link>
            <LangToggle />
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
