import type { CSSProperties } from "react";
import { theme } from "../../theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "../layout/styles";

const HARMONIC_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"];

export type HarmonicWheelProps = {
  selectedKey?: string | null;
  onSelectKey?: (key: string) => void;
};

const containerStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  gap: "12px",
  padding: "12px 16px",
  minHeight: "200px"
};

export function HarmonicWheel({ selectedKey, onSelectKey }: HarmonicWheelProps) {
  return (
    <section style={containerStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: "0.8rem", letterSpacing: "0.05em" }}>Harmonic Wheel</strong>
        {selectedKey && <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>Key: {selectedKey}</span>}
      </header>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
          gap: "8px",
          fontSize: "0.75rem"
        }}
      >
        {HARMONIC_KEYS.map((key) => {
          const isActive = selectedKey === key;
          return (
            <button
              key={key}
              type="button"
              style={{
                ...toolbarButtonStyle,
                padding: "10px 12px",
                background: isActive ? theme.button.primary : theme.button.base,
                color: isActive ? theme.button.primaryText : theme.text,
                borderColor: isActive ? theme.button.primary : theme.button.outline
              }}
              onClick={() => onSelectKey?.(key)}
            >
              {key}
            </button>
          );
        })}
      </div>
      <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
        Choose a root key to highlight harmonically compatible material.
      </p>
    </section>
  );
}
