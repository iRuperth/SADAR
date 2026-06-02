import { motion, useScroll, useTransform, type MotionValue } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import LangToggle from "../components/LangToggle";
import { useT } from "../i18n";
import "./presentation.css";

type Progress = MotionValue<number>;
type Range = [number, number];

const TEAM = [
  { roleKey: "tower1", name: "Nombre Apellido" },
  { roleKey: "pilot", name: "Nombre Apellido" },
  { roleKey: "tower2", name: "Nombre Apellido" },
] as const;

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
  panelRanges,
  header,
  panels,
}: {
  progress: Progress;
  range: Range;
  panelRanges: Range[];
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
            range={panelRanges[index]}
            title={panel.title}
            body={panel.body}
          />
        ))}
      </div>
    </motion.div>
  );
}

const INTRO_PANEL_RANGES: Range[] = [
  [0.51, 0.62],
  [0.62, 0.74],
  [0.74, 0.86],
];

export default function Presentation() {
  const t = useT();
  const s = t.scenes;
  const trackRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const handedOffRef = useRef(false);
  const { scrollYProgress } = useScroll({
    target: trackRef,
    offset: ["start start", "end end"],
  });

  useEffect(() => {
    const unsub = scrollYProgress.on("change", (value) => {
      if (value > 0.96 && !handedOffRef.current) {
        handedOffRef.current = true;
        navigate("/dashboard");
      }
    });
    return () => unsub();
  }, [scrollYProgress, navigate]);

  return (
    <div className="presentation">
      <motion.div
        className="presentation__progress"
        style={{ scaleX: scrollYProgress, width: "100%" }}
      />
      <div className="presentation__skip" style={{ display: "flex", gap: 8 }}>
        <LangToggle />
        <Link to="/dashboard">
          <button>{s.skip}</button>
        </Link>
      </div>
      <div className="presentation__track" ref={trackRef}>
        <div className="stage">
          <ImageScene
            progress={scrollYProgress}
            range={[0, 0.14]}
            first
            image="terminal.jpg"
            kicker={s.terminal.kicker}
            title={s.terminal.title}
            sub={s.terminal.sub}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.13, 0.28]}
            image="jetbridge.jpg"
            kicker={s.jetbridge.kicker}
            title={s.jetbridge.title}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.27, 0.42]}
            image="cabin.jpg"
            kicker={s.cabin.kicker}
            title={s.cabin.title}
            sub={s.cabin.sub}
          />
          <WindowScene
            progress={scrollYProgress}
            range={[0.41, 0.52]}
            closing
            kicker={s.closing.kicker}
            title={s.closing.title}
          />
          <Explain
            progress={scrollYProgress}
            range={[0.5, 0.92]}
            panelRanges={INTRO_PANEL_RANGES}
            header={s.explainHeader}
            panels={s.panels}
          />
        </div>
      </div>
    </div>
  );
}

export function PresentationFinal() {
  const t = useT();
  const s = t.scenes;
  const trackRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: trackRef,
    offset: ["start start", "end end"],
  });
  const [creditsVisible, setCreditsVisible] = useState(false);

  useEffect(() => {
    const unsub = scrollYProgress.on("change", (value) => {
      setCreditsVisible(value > 0.88);
    });
    return () => unsub();
  }, [scrollYProgress]);

  return (
    <div className="presentation">
      <motion.div
        className="presentation__progress"
        style={{ scaleX: scrollYProgress, width: "100%" }}
      />
      <div className="presentation__skip" style={{ display: "flex", gap: 8 }}>
        <LangToggle />
        <Link to="/dashboard">
          <button>{s.skip}</button>
        </Link>
      </div>
      <div className="presentation__track" ref={trackRef}>
        <div className="stage">
          <WindowScene
            progress={scrollYProgress}
            range={[0, 0.34]}
            closing={false}
            kicker={s.opening.kicker}
            title={s.opening.title}
          />
          <ImageScene
            progress={scrollYProgress}
            range={[0.33, 0.66]}
            image="leaving.jpg"
            kicker={s.leaving.kicker}
            title={s.leaving.title}
          />
          <Credits
            progress={scrollYProgress}
            range={[0.65, 1.0]}
            roles={s.ack.roles}
            teamHeader={s.ack.teamHeader}
            thanks={s.ack.thanks}
            subtitle={s.ack.subtitle}
            visible={creditsVisible}
          />
        </div>
      </div>
    </div>
  );
}

function Credits({
  progress,
  range,
  roles,
  teamHeader,
  thanks,
  subtitle,
  visible,
}: {
  progress: Progress;
  range: Range;
  roles: { tower1: string; pilot: string; tower2: string };
  teamHeader: string;
  thanks: string;
  subtitle: string;
  visible: boolean;
}) {
  const [start, end] = range;
  const pad = Math.min(0.04, (end - start) / 3);
  const opacity = useTransform(progress, [start, start + pad], [0, 1]);
  const textOpacity = useTransform(progress, [start + 0.03, start + 0.1], [0, 1]);
  return (
    <motion.div
      className="scene"
      style={{ opacity, backgroundImage: "url(/presentation/acknowledgements.jpg)" }}
    >
      <div className="scene__veil" />
      <motion.div
        className="ack"
        style={{
          opacity: textOpacity,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 18,
          textAlign: "center",
          padding: "24px",
        }}
      >
        <img
          src="/sadar-logo.png"
          alt="SADAR"
          className="radar-pulse"
          style={{ width: "min(440px, 70vw)", height: "auto", display: "block" }}
        />
        <p className="scene__sub" style={{ marginTop: 0 }}>{subtitle}</p>

        <div
          style={{
            marginTop: 24,
            fontFamily: "var(--mono)",
            fontSize: 13,
            letterSpacing: "0.32em",
            textTransform: "uppercase",
            color: "#ffffff",
            textShadow: "0 1px 6px rgba(0,0,0,0.85)",
          }}
        >
          {teamHeader}
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 22,
            marginTop: 10,
            opacity: visible ? 1 : 0,
            transition: "opacity 600ms ease",
          }}
        >
          {TEAM.map((member, idx) => (
            <div key={idx} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 12,
                  letterSpacing: "0.22em",
                  textTransform: "uppercase",
                  color: "#7fd1c6",
                  textShadow: "0 1px 5px rgba(0,0,0,0.85)",
                }}
              >
                {roles[member.roleKey]}
              </div>
              <div
                className="radar-name-pulse"
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 22,
                  letterSpacing: "0.1em",
                  color: "#ffffff",
                  animationDelay: `${idx * 0.4}s`,
                }}
              >
                {member.name}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            marginTop: 32,
            fontFamily: "var(--mono)",
            fontSize: 16,
            letterSpacing: "0.2em",
            color: "#7fd1c6",
            textTransform: "uppercase",
            textShadow: "0 1px 6px rgba(0,0,0,0.85)",
          }}
        >
          {thanks}
        </div>
      </motion.div>
    </motion.div>
  );
}
