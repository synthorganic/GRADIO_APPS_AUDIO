import { theme } from "@daw/theme";
import { cardSurfaceStyle } from "@daw/components/layout/styles";

export interface FxModuleConfig {
  id: string;
  label: string;
  amount: number;
  enabled: boolean;
  accent: string;
}

export interface FxRackPanelProps {
  title: string;
  modules: FxModuleConfig[];
  alignment: "left" | "right";
}

function ModuleRow({ module }: { module: FxModuleConfig }) {
  const percent = Math.round(module.amount * 100);
  const gradient = `linear-gradient(90deg, ${module.accent} 0%, rgba(9, 43, 60, 0.85) 90%)`;
  return (
    <div
      style={{
        display: "grid",
        gap: "10px",
        padding: "12px 14px",
        borderRadius: "12px",
        border: `1px solid ${module.enabled ? module.accent : "rgba(120, 203, 220, 0.25)"}`,
        background: module.enabled ? "rgba(9, 43, 60, 0.85)" : "rgba(6, 24, 34, 0.75)",
        boxShadow: module.enabled ? "0 18px 36px rgba(6, 20, 28, 0.55)" : "none",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span
          style={{
            fontSize: "0.72rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: module.enabled ? theme.button.primaryText : theme.textMuted,
          }}
        >
          {module.label}
        </span>
        <span
          style={{
            fontSize: "0.68rem",
            letterSpacing: "0.06em",
            color: module.enabled ? module.accent : "rgba(120, 203, 220, 0.6)",
          }}
        >
          {percent}%
        </span>
      </div>
      <div
        style={{
          position: "relative",
          height: "8px",
          borderRadius: "999px",
          overflow: "hidden",
          background: "rgba(9, 32, 45, 0.7)",
        }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            transformOrigin: "left",
            transform: `scaleX(${module.amount})`,
            background: gradient,
            opacity: module.enabled ? 1 : 0.35,
          }}
        />
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span
            style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              border: `1px solid ${module.accent}`,
              background: module.enabled ? module.accent : "transparent",
            }}
          />
          <span style={{ fontSize: "0.62rem", color: theme.textMuted, letterSpacing: "0.08em" }}>
            {module.enabled ? "Engaged" : "Bypassed"}
          </span>
        </div>
        <span style={{ fontSize: "0.62rem", color: theme.textMuted, letterSpacing: "0.08em" }}>
          Macro {module.label.substring(0, 1)}
        </span>
      </div>
    </div>
  );
}

export function FxRackPanel({ title, modules, alignment }: FxRackPanelProps) {
  const gradient =
    alignment === "left"
      ? "linear-gradient(180deg, rgba(16, 52, 72, 0.92) 0%, rgba(7, 22, 32, 0.95) 100%)"
      : "linear-gradient(180deg, rgba(12, 44, 63, 0.92) 0%, rgba(4, 18, 28, 0.95) 100%)";

  return (
    <aside
      style={{
        ...cardSurfaceStyle,
        background: gradient,
        padding: "22px 20px",
        display: "grid",
        gap: "16px",
        alignContent: "start",
        minWidth: "248px",
      }}
    >
      <header style={{ display: "grid", gap: "6px" }}>
        <span
          style={{
            fontSize: "0.66rem",
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            color: "rgba(120, 203, 220, 0.7)",
          }}
        >
          {alignment === "left" ? "Input" : "Output"} Chain
        </span>
        <h3
          style={{
            margin: 0,
            fontSize: "0.9rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          {title}
        </h3>
      </header>
      <div style={{ display: "grid", gap: "12px" }}>
        {modules.map((module) => (
          <ModuleRow key={module.id} module={module} />
        ))}
      </div>
    </aside>
  );
}
