import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { useRef } from "react";
import { Link } from "react-router-dom";

import LangToggle from "../components/LangToggle";
import { useT } from "../i18n";
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
    <motion.div className="scene" style={{ opacity, backgroundImage: `url(/presentation/${image})` }}>
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
  const shadeY = useTransform(progress, shadeFrom, closing ? ["-100%", "0%"] : ["0%", "-100%"]);
  return (
    <motion.div className="scene" style={{ opacity, backgroundImage: "url(/presentation/window.jpg)" }}>
      <div className="scene__veil" />
      <motion.div className="scene__shade" style={{ y: shadeY }} />
      <div className="scene__text">
        <div className="scene__kicker">{kicker}</div>
        <h2 className="scene__title">{title}</h2>
      </div>
    </motion.div>
  );
}

const PANEL_RANGES: Range[] = [
  [0.51, 0.57],
  [0.57, 0.63],
  [0.63, 0.68],
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

function Explain({
  progress,
  range,
  header,
  panels,
}: {
  progress: Progress;
  range: Range;
  header: string;
  panels: { title: string; body: string }[];
}) {
  const opacity = useReveal(progress, range);
  return (
    <motion.div className="scene" style={{ opacity, backgroundImage: "url(/presentation/console.jpg)" }}>
      <div className="scene__veil" />
      <div className="explain__header">{header}</div>
      <div className="explain">
        {panels.map((panel, index) => (
          <ExplainPanel
            key={index}
            progress={progress}
            range={PANEL_RANGES[index]}
            title={panel.title}
            body={panel.body}
          />
        ))}
      </div>
    </motion.div>
  );
}

function Acknowledgements({
  progress,
  range,
  kicker,
  subtitle,
  enter,
}: {
  progress: Progress;
  range: Range;
  kicker: string;
  subtitle: string;
  enter: string;
}) {
  const opacity = useReveal(progress, range);
  const textOpacity = useTransform(progress, [range[0] + 0.03, range[0] + 0.08], [0, 1]);
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: "url(/presentation/acknowledgements.jpg)" }}
    >
      <div className="scene__veil" />
      <motion.div className="ack" style={{ opacity: textOpacity }}>
        <div className="scene__kicker">{kicker}</div>
        <div className="ack__title">SADAR</div>
        <p className="scene__sub">{subtitle}</p>
        <Link to="/" style={{ marginTop: 18 }}>
          <button>{enter}</button>
        </Link>
      </motion.div>
    </motion.div>
  );
}

export default function Presentation() {
  const t = useT();
  const s = t.scenes;
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
      <div className="presentation__skip" style={{ display: "flex", gap: 8 }}>
        <LangToggle />
        <Link to="/">
          <button>{s.skip}</button>
        </Link>
      </div>
      <div className="presentation__track" ref={trackRef}>
        <div className="stage">
          <ImageScene
            progress={scrollYProgress}
            range={[0, 0.12]}
            first
            image="terminal.jpg"
            kicker={s.terminal.kicker}
            title={s.terminal.title}
            sub={s.terminal.sub}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.11, 0.24]}
            image="jetbridge.jpg"
            kicker={s.jetbridge.kicker}
            title={s.jetbridge.title}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.23, 0.36]}
            image="cabin.jpg"
            kicker={s.cabin.kicker}
            title={s.cabin.title}
            sub={s.cabin.sub}
          />
          <WindowScene
            progress={scrollYProgress}
            range={[0.35, 0.49]}
            closing
            kicker={s.closing.kicker}
            title={s.closing.title}
          />
          <Explain
            progress={scrollYProgress}
            range={[0.48, 0.69]}
            header={s.explainHeader}
            panels={s.panels}
          />
          <WindowScene
            progress={scrollYProgress}
            range={[0.68, 0.8]}
            closing={false}
            kicker={s.opening.kicker}
            title={s.opening.title}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.79, 0.9]}
            image="leaving.jpg"
            kicker={s.leaving.kicker}
            title={s.leaving.title}
          />
          <Acknowledgements
            progress={scrollYProgress}
            range={[0.89, 1.0]}
            kicker={s.ack.kicker}
            subtitle={s.ack.subtitle}
            enter={s.enter}
          />
        </div>
      </div>
    </div>
  );
}
