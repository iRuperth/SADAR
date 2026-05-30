import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Lang = "en" | "es";

interface Panel {
  title: string;
  body: string;
}

interface Dict {
  appTitle: string;
  nav: { monitor: string; simulator: string; metrics: string; presentation: string };
  monitor: { mostAnomalous: string; offline: string; actual: string; reconstructed: string };
  alert: { alert: string; nominal: string; peak: string; threshold: string; latency: string };
  simulator: {
    baseFlight: string;
    injectedAnomaly: string;
    intensity: string;
    onset: string;
    onsetSuffix: string;
    original: string;
    injected: string;
    kinds: Record<string, string>;
  };
  metrics: {
    title: string;
    model: string;
    realRoc: string;
    realPr: string;
    synRoc: string;
    none: string;
    selected: string;
  };
  timeline: { threshold: string };
  scenes: {
    skip: string;
    enter: string;
    terminal: { kicker: string; title: string; sub: string };
    jetbridge: { kicker: string; title: string };
    cabin: { kicker: string; title: string; sub: string };
    closing: { kicker: string; title: string };
    explainHeader: string;
    panels: Panel[];
    opening: { kicker: string; title: string };
    leaving: { kicker: string; title: string };
    ack: { kicker: string; subtitle: string };
  };
}

const EN: Dict = {
  appTitle: "FLIGHT CONFORMANCE MONITOR",
  nav: { monitor: "Monitor", simulator: "Simulator", metrics: "Metrics", presentation: "Presentation" },
  monitor: {
    mostAnomalous: "Most anomalous windows",
    offline: "backend offline. start it with: make serve",
    actual: "actual",
    reconstructed: "reconstructed",
  },
  alert: { alert: "ALERT", nominal: "NOMINAL", peak: "peak", threshold: "thr", latency: "latency" },
  simulator: {
    baseFlight: "Base flight (normal)",
    injectedAnomaly: "Injected anomaly",
    intensity: "Intensity",
    onset: "Onset",
    onsetSuffix: "% into the window",
    original: "original",
    injected: "injected",
    kinds: {
      route_deviation: "Route deviation",
      altitude: "Altitude bust",
      speed: "Speed anomaly",
      holding: "Holding turns",
      freeze: "Transponder cut",
    },
  },
  metrics: {
    title: "Model comparison on held-out data",
    model: "Model",
    realRoc: "real ROC",
    realPr: "real PR-AUC",
    synRoc: "syn ROC",
    none: "no metrics yet. generate them with: make compare",
    selected: "selected final model",
  },
  timeline: { threshold: "THRESHOLD" },
  scenes: {
    skip: "Skip",
    enter: "Enter the console",
    terminal: {
      kicker: "Madrid Barajas / LEMD",
      title: "You walk through the terminal",
      sub: "Every day, around a thousand flights follow the same quiet pattern.",
    },
    jetbridge: { kicker: "Boarding", title: "Down the jet bridge" },
    cabin: { kicker: "On board", title: "Take your seat", sub: "Lower the window shade." },
    closing: { kicker: "Doors closed", title: "The flight begins" },
    explainHeader: "Air traffic control / LEMD",
    panels: [
      {
        title: "It learns what normal looks like",
        body: "A neural network sees thousands of ordinary approaches and departures and learns their shape.",
      },
      {
        title: "It flags what deviates",
        body: "Anything it cannot reconstruct, an odd route, an altitude bust, a frozen transponder, raises the score.",
      },
      {
        title: "You can provoke it live",
        body: "Inject a deviation or cut the transponder and watch the alert fire, with its detection latency.",
      },
    ],
    opening: { kicker: "Destination", title: "Arrived safely" },
    leaving: { kicker: "Conformance confirmed", title: "You walk out" },
    ack: { kicker: "Thank you", subtitle: "Smart Anomaly Detection for Aviation Routes" },
  },
};

const ES: Dict = {
  appTitle: "MONITOR DE CONFORMIDAD DE VUELOS",
  nav: { monitor: "Monitor", simulator: "Simulador", metrics: "Métricas", presentation: "Presentación" },
  monitor: {
    mostAnomalous: "Ventanas más anómalas",
    offline: "backend apagado. arráncalo con: make serve",
    actual: "real",
    reconstructed: "reconstruido",
  },
  alert: { alert: "ALERTA", nominal: "NORMAL", peak: "pico", threshold: "umbral", latency: "latencia" },
  simulator: {
    baseFlight: "Vuelo base (normal)",
    injectedAnomaly: "Anomalía inyectada",
    intensity: "Intensidad",
    onset: "Inicio",
    onsetSuffix: "% de la ventana",
    original: "original",
    injected: "inyectado",
    kinds: {
      route_deviation: "Desvío de ruta",
      altitude: "Anomalía de altitud",
      speed: "Anomalía de velocidad",
      holding: "Giros de espera",
      freeze: "Corte de transpondedor",
    },
  },
  metrics: {
    title: "Comparativa de modelos (datos no vistos)",
    model: "Modelo",
    realRoc: "ROC real",
    realPr: "PR-AUC real",
    synRoc: "ROC sint",
    none: "sin métricas. genéralas con: make compare",
    selected: "modelo final elegido",
  },
  timeline: { threshold: "UMBRAL" },
  scenes: {
    skip: "Saltar",
    enter: "Entrar a la consola",
    terminal: {
      kicker: "Madrid Barajas / LEMD",
      title: "Cruzas la terminal",
      sub: "Cada día, unos mil vuelos siguen el mismo patrón tranquilo.",
    },
    jetbridge: { kicker: "Embarque", title: "Por el pasillo de embarque" },
    cabin: { kicker: "A bordo", title: "Toma asiento", sub: "Baja la persiana de la ventanilla." },
    closing: { kicker: "Puertas cerradas", title: "Comienza el vuelo" },
    explainHeader: "Control de tráfico aéreo / LEMD",
    panels: [
      {
        title: "Aprende qué es lo normal",
        body: "Una red neuronal ve miles de aproximaciones y despegues normales y aprende su forma.",
      },
      {
        title: "Señala lo que se desvía",
        body: "Lo que no logra reconstruir, una ruta rara, una altitud incoherente, un transpondedor congelado, sube la puntuación.",
      },
      {
        title: "Puedes provocarlo en vivo",
        body: "Inyecta un desvío o corta el transpondedor y mira saltar la alerta, con su latencia de detección.",
      },
    ],
    opening: { kicker: "Destino", title: "Has llegado" },
    leaving: { kicker: "Conformidad confirmada", title: "Sales caminando" },
    ack: { kicker: "Gracias", subtitle: "Detección inteligente de anomalías en rutas aéreas" },
  },
};

const DICTS: Record<Lang, Dict> = { en: EN, es: ES };

interface LangContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  toggle: () => void;
}

const LangContext = createContext<LangContextValue>({ lang: "en", setLang: () => {}, toggle: () => {} });

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    const saved = localStorage.getItem("sadar-lang");
    if (saved === "en" || saved === "es") return saved;
    return navigator.language.startsWith("es") ? "es" : "en";
  });

  useEffect(() => {
    localStorage.setItem("sadar-lang", lang);
  }, [lang]);

  const value = useMemo<LangContextValue>(
    () => ({ lang, setLang, toggle: () => setLang(lang === "en" ? "es" : "en") }),
    [lang],
  );

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang(): LangContextValue {
  return useContext(LangContext);
}

export function useT(): Dict {
  return DICTS[useContext(LangContext).lang];
}
