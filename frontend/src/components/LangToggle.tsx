import { useLang } from "../i18n";

export default function LangToggle() {
  const { lang, toggle } = useLang();
  return (
    <button onClick={toggle} title="Language / Idioma">
      <span style={{ color: lang === "en" ? "var(--info)" : "var(--muted)" }}>EN</span>
      <span style={{ color: "var(--muted)" }}> / </span>
      <span style={{ color: lang === "es" ? "var(--info)" : "var(--muted)" }}>ES</span>
    </button>
  );
}
