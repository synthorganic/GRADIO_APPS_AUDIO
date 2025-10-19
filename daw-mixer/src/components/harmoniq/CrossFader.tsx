import { useMemo } from "react";
import type { CSSProperties, ChangeEvent } from "react";
import { theme } from "../../theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "../layout/styles";

export type CrossFaderProps = {
  value: number;
  onChange: (value: number) => void;
  onCueDeck?: (deck: "A" | "B") => void;
};

const containerStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  gap: "12px",
  padding: "12px 16px"
};

export function CrossFader({ value, onChange, onCueDeck }: CrossFaderProps) {
  const percentage = useMemo(() => Math.round(value * 100), [value]);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = Number(event.target.value) / 100;
    onChange(next);
  };

  return (
    <section style={containerStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: "0.8rem", letterSpacing: "0.05em" }}>Crossfader</strong>
        <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{percentage}%</span>
      </header>
      <div style={{ display: "grid", gap: "8px" }}>
        <input
          type="range"
          min="0"
          max="100"
          value={percentage}
          onChange={handleChange}
          style={{
            width: "100%",
            accentColor: theme.button.primary,
            cursor: "pointer"
          }}
        />
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: theme.textMuted }}>
          <span>Deck A</span>
          <span>Deck B</span>
        </div>
      </div>
      {onCueDeck && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: "8px" }}>
          <button type="button" style={toolbarButtonStyle} onClick={() => onCueDeck("A")}>
            Cue A
          </button>
          <button type="button" style={toolbarButtonStyle} onClick={() => onCueDeck("B")}>
            Cue B
          </button>
        </div>
      )}
    </section>
  );
}
