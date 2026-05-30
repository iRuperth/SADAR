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
        title: "From the ground, we look after every flight",
        body: "While you fly, the control tower keeps watch, making sure your route is the one it should be.",
      },
      {
        title: "Every route checked against the pattern",
        body: "Each approach and departure is compared with how flights normally behave, so any drift off course stands out.",
      },
      {
        title: "Deviations and incidents, caught early",
        body: "An odd route, an altitude that does not fit, a transponder that goes silent: spotted as it begins, not after.",
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
        title: "Desde tierra, cuidamos de cada vuelo",
        body: "Mientras vuelas, la torre de control vigila, asegurándose de que tu ruta es la que debe ser.",
      },
      {
        title: "Cada ruta, comparada con el patrón",
        body: "Cada aproximación y despegue se compara con cómo vuelan normalmente, así cualquier desvío del rumbo salta a la vista.",
      },
      {
        title: "Desviaciones e incidencias, a tiempo",
        body: "Una ruta rara, una altitud que no encaja, un transpondedor que se calla: se detecta en cuanto empieza, no después.",
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
