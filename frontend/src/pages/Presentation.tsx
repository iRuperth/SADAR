import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { useRef } from "react";
import { Link } from "react-router-dom";

import "./presentation.css";

type Progress = MotionValue<number>;
type Range = [number, number];

function useReveal(progress: Progress, range: Range, first = false): MotionValue<number> {
  const [start, end] = range;
  const pad = Math.min(0.04, (end - start) / 3);
  if (first) {
    return useTransform(progress, [start, end - pad, end], [1, 1, 0]);
  }
  return useTransform(progress, [start, start + pad, end - pad, end], [0, 1, 1, 0]);
}

function ImageScene({
  progress,
  range,
  image,
  kicker,
  title,
  sub,
  first,
}: {
  progress: Progress;
  range: Range;
  image: string;
  kicker: string;
  title: string;
  sub?: string;
  first?: boolean;
}) {
  const opacity = useReveal(progress, range, first);
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: `url(/presentation/${image})` }}
    >
      <div className="scene__veil" />
      <div className="scene__text">
        <div className="scene__kicker">{kicker}</div>
        <h2 className="scene__title">{title}</h2>
        {sub && <p className="scene__sub">{sub}</p>}
      </div>
    </motion.div>
  );
}

function WindowScene({
  progress,
  range,
  closing,
  kicker,
  title,
}: {
  progress: Progress;
  range: Range;
  closing: boolean;
  kicker: string;
  title: string;
}) {
  const opacity = useReveal(progress, range);
  const shadeFrom: Range = [range[0] + 0.02, range[0] + 0.07];
  const shadeY = useTransform(
    progress,
    shadeFrom,
    closing ? ["-100%", "0%"] : ["0%", "-100%"],
  );
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: "url(/presentation/window.jpg)" }}
    >
      <div className="scene__veil" />
      <motion.div className="scene__shade" style={{ y: shadeY }} />
      <div className="scene__text">
        <div className="scene__kicker">{kicker}</div>
        <h2 className="scene__title">{title}</h2>
      </div>
    </motion.div>
  );
}

const PANELS: { range: Range; title: string; body: string }[] = [
  {
    range: [0.51, 0.57],
    title: "It learns what normal looks like",
    body: "A neural network sees thousands of ordinary approaches and departures and learns their shape.",
  },
  {
    range: [0.57, 0.63],
    title: "It flags what deviates",
    body: "Anything it cannot reconstruct, an odd route, an altitude bust, a frozen transponder, raises the score.",
  },
  {
    range: [0.63, 0.68],
    title: "You can provoke it live",
    body: "Inject a deviation or cut the transponder and watch the alert fire, with its detection latency.",
  },
];

function ExplainPanel({
  progress,
  range,
  title,
  body,
}: {
  progress: Progress;
  range: Range;
  title: string;
  body: string;
}) {
  const opacity = useReveal(progress, range);
  const y = useTransform(progress, range, [28, -28]);
  return (
    <motion.div className="explain__panel" style={{ opacity, y }}>
      <h3>{title}</h3>
      <p>{body}</p>
    </motion.div>
  );
}

function Explain({ progress, range }: { progress: Progress; range: Range }) {
  const opacity = useReveal(progress, range);
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: "url(/presentation/console.jpg)" }}
    >
      <div className="scene__veil" />
      <div className="explain__header">Air traffic control / LEMD</div>
      <div className="explain">
        {PANELS.map((panel) => (
          <ExplainPanel key={panel.title} progress={progress} {...panel} />
        ))}
      </div>
    </motion.div>
  );
}

function Acknowledgements({ progress, range }: { progress: Progress; range: Range }) {
  const opacity = useReveal(progress, range);
  const textOpacity = useTransform(progress, [range[0] + 0.03, range[0] + 0.08], [0, 1]);
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: "url(/presentation/acknowledgements.jpg)" }}
    >
      <div className="scene__veil" />
      <motion.div className="ack" style={{ opacity: textOpacity }}>
        <div className="scene__kicker">Thank you</div>
        <div className="ack__title">SADAR</div>
        <p className="scene__sub">Smart Anomaly Detection for Aviation Routes</p>
        <Link to="/" style={{ marginTop: 18 }}>
          <button>Enter the console</button>
        </Link>
      </motion.div>
    </motion.div>
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
      <motion.div
        className="presentation__progress"
        style={{ scaleX: scrollYProgress, width: "100%" }}
      />
      <div className="presentation__skip">
        <Link to="/">
          <button>Skip</button>
        </Link>
      </div>
      <div className="presentation__track" ref={trackRef}>
        <div className="stage">
          <ImageScene
            progress={scrollYProgress}
            range={[0, 0.12]}
            first
            image="terminal.jpg"
            kicker="Madrid Barajas / LEMD"
            title="You walk through the terminal"
            sub="Every day, around a thousand flights follow the same quiet pattern."
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.11, 0.24]}
            image="jetbridge.jpg"
            kicker="Boarding"
            title="Down the jet bridge"
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.23, 0.36]}
            image="cabin.jpg"
            kicker="On board"
            title="Take your seat"
            sub="Lower the window shade."
          />
          <WindowScene
            progress={scrollYProgress}
            range={[0.35, 0.49]}
            closing
            kicker="Doors closed"
            title="The flight begins"
          />
          <Explain progress={scrollYProgress} range={[0.48, 0.69]} />
          <WindowScene
            progress={scrollYProgress}
            range={[0.68, 0.8]}
            closing={false}
            kicker="Destination"
            title="Arrived safely"
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.79, 0.9]}
            image="leaving.jpg"
            kicker="Conformance confirmed"
            title="You walk out"
          />
          <Acknowledgements progress={scrollYProgress} range={[0.89, 1.0]} />
        </div>
      </div>
    </div>
  );
}
