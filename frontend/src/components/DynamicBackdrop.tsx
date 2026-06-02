import { useEffect, useState } from "react";

const PHASES = ["day", "mid", "night"] as const;
const STEP_MS = 5 * 60 * 1000;

export default function DynamicBackdrop({ light }: { light: boolean }) {
  const [phase, setPhase] = useState(() => (light ? 0 : 2));

  useEffect(() => {
    setPhase(light ? 0 : 2);
  }, [light]);

  useEffect(() => {
    const id = setInterval(() => setPhase((current) => (current + 1) % PHASES.length), STEP_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="backdrop">
      {PHASES.map((name, index) => (
        <div
          key={name}
          className="backdrop__layer"
          style={{ backgroundImage: `url(/bg-${name}.jpg)`, opacity: index === phase ? 1 : 0 }}
        />
      ))}
      <div className="backdrop__veil" />
    </div>
  );
}
