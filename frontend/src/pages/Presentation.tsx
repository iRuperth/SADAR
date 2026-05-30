import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { useRef } from "react";
import { Link } from "react-router-dom";

import "./presentation.css";

type Progress = MotionValue<number>;

function useFade(progress: Progress, start: number, end: number): MotionValue<number> {
  const pad = Math.min(0.04, (end - start) / 3);
  return useTransform(progress, [start, start + pad, end - pad, end], [0, 1, 1, 0]);
}

function Terminal({ progress }: { progress: Progress }) {
  const opacity = useTransform(progress, [0, 0.12, 0.16], [1, 1, 0]);
  const titleY = useTransform(progress, [0, 0.16], [0, -60]);
  return (
    <motion.div className="scene" style={{ opacity }}>
      <div className="terminal-skyline" />
      <div className="terminal-window-row">
        {Array.from({ length: 9 }).map((_, i) => (
          <span key={i} />
        ))}
      </div>
      <motion.div style={{ y: titleY }}>
        <div className="scene__kicker">Madrid Barajas / LEMD</div>
        <h2 className="scene__title">You walk through the terminal</h2>
        <p className="scene__sub">Every day, around a thousand flights follow the same quiet pattern.</p>
      </motion.div>
    </motion.div>
  );
}

function JetBridge({ progress }: { progress: Progress }) {
  const opacity = useFade(progress, 0.13, 0.31);
  const scale = useTransform(progress, [0.13, 0.31], [0.4, 2.4]);
  return (
    <motion.div className="scene" style={{ opacity }}>
      <motion.div className="tunnel" style={{ scale }}>
        {[0.2, 0.36, 0.52, 0.68, 0.84, 1].map((ring, i) => (
          <div
            key={i}
            className="tunnel__ring"
            style={{ transform: `scale(${ring})`, opacity: 0.15 + ring * 0.6 }}
          />
        ))}
      </motion.div>
      <div style={{ position: "relative" }}>
        <div className="scene__kicker">Boarding</div>
        <h2 className="scene__title">Down the jet bridge</h2>
      </div>
    </motion.div>
  );
}

function CabinWindow({ progress }: { progress: Progress }) {
  const opacity = useTransform(progress, [0.28, 0.33, 0.52, 0.56], [0, 1, 1, 0]);
  const shadeY = useTransform(progress, [0.4, 0.48], ["-100%", "0%"]);
  const seatHint = useTransform(progress, [0.3, 0.34, 0.4, 0.44], [0, 1, 1, 0]);
  const beginHint = useTransform(progress, [0.46, 0.5], [0, 1]);
  return (
    <motion.div className="scene" style={{ opacity }}>
      <div className="cabin" />
      <div className="window-glow" />
      <div className="window">
        <div className="window__outside" />
        <motion.div className="window__shade" style={{ y: shadeY }} />
      </div>
      <motion.p className="scene__sub" style={{ opacity: seatHint, position: "absolute", bottom: "16%" }}>
        Take your seat. Lower the window shade.
      </motion.p>
      <motion.div style={{ opacity: beginHint, position: "absolute", bottom: "12%" }}>
        <div className="scene__kicker">Doors closed</div>
        <h2 className="scene__title">The flight begins</h2>
      </motion.div>
    </motion.div>
  );
}

const PANELS = [
  {
    range: [0.56, 0.63] as [number, number],
    title: "It learns what normal looks like",
    body: "A neural network sees thousands of ordinary approaches and departures and learns their shape.",
  },
  {
    range: [0.63, 0.69] as [number, number],
    title: "It flags what deviates",
    body: "Anything it cannot reconstruct, an odd route, an altitude bust, a frozen transponder, raises the score.",
  },
  {
    range: [0.69, 0.75] as [number, number],
    title: "You can provoke it live",
    body: "Inject a deviation or cut the transponder and watch the alert fire, with its detection latency.",
  },
];

function ExplainPanel({ progress, range, title, body }: {
  progress: Progress;
  range: [number, number];
  title: string;
  body: string;
}) {
  const opacity = useFade(progress, range[0], range[1]);
  const y = useTransform(progress, range, [30, -30]);
  return (
    <motion.div className="explain__panel" style={{ opacity, y, position: "absolute" }}>
      <h3>{title}</h3>
      <p>{body}</p>
    </motion.div>
  );
}

function Explain({ progress }: { progress: Progress }) {
  const opacity = useFade(progress, 0.54, 0.78);
  return (
    <motion.div className="scene" style={{ opacity }}>
      <div className="cabin" />
      <div className="explain">
        {PANELS.map((panel) => (
          <ExplainPanel key={panel.title} progress={progress} {...panel} />
        ))}
      </div>
    </motion.div>
  );
}

function Arrival({ progress }: { progress: Progress }) {
  const opacity = useFade(progress, 0.72, 0.9);
  const shadeY = useTransform(progress, [0.76, 0.84], ["0%", "-100%"]);
  const landed = useTransform(progress, [0.82, 0.88], [0, 1]);
  return (
    <motion.div className="scene" style={{ opacity }}>
      <div className="cabin" />
      <div className="window-glow" />
      <div className="window">
        <div className="window__outside window__outside--night" />
        <motion.div className="window__shade" style={{ y: shadeY }} />
      </div>
      <motion.div style={{ opacity: landed, position: "absolute", bottom: "12%" }}>
        <div className="scene__kicker">Destination</div>
        <h2 className="scene__title">Arrived safely</h2>
      </motion.div>
    </motion.div>
  );
}

function Acknowledgements({ progress }: { progress: Progress }) {
  const blur = useTransform(progress, [0.84, 0.92], [0, 1]);
  const opacity = useTransform(progress, [0.87, 0.94], [0, 1]);
  return (
    <>
      <motion.div
        className="scene"
        style={{ opacity: blur, backdropFilter: "blur(16px)", background: "rgba(5, 8, 13, 0.6)" }}
      />
      <motion.div className="ack" style={{ opacity }}>
        <div className="scene__kicker">Thank you</div>
        <div className="ack__title">SADAR</div>
        <p className="scene__sub">Smart Anomaly Detection for Aviation Routes</p>
        <Link to="/" style={{ marginTop: 18 }}>
          <button>Enter the console</button>
        </Link>
      </motion.div>
    </>
  );
}

export default function Presentation() {
  const trackRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: trackRef,
    offset: ["start start", "end end"],
  });

  return (
    <div className="presentation">
      <motion.div className="presentation__progress" style={{ scaleX: scrollYProgress, width: "100%" }} />
      <div className="presentation__skip">
        <Link to="/">
          <button>Skip</button>
        </Link>
      </div>
      <div className="presentation__track" ref={trackRef}>
        <div className="stage">
          <Terminal progress={scrollYProgress} />
          <JetBridge progress={scrollYProgress} />
          <CabinWindow progress={scrollYProgress} />
          <Explain progress={scrollYProgress} />
          <Arrival progress={scrollYProgress} />
          <Acknowledgements progress={scrollYProgress} />
        </div>
      </div>
    </div>
  );
}
