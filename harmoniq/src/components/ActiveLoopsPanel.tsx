import type { CSSProperties } from "react";
import { memo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import { WaveformPreview } from "@daw/shared/WaveformPreview";

export interface ActiveLoop {
  id: string;
  name: string;
  bpm: number;
  key: string;
  deck: "A" | "B" | "C" | "D" | null;
  waveform: Float32Array;
  energy: number;
  filter: number;
  level: number;
  isMuted: boolean;
}

export interface ActiveLoopsPanelProps {
  loops: ActiveLoop[];
  onToggleMute: (loopId: string) => void;
  onFocusDeck: (deck: "A" | "B" | "C" | "D") => void;
  highlightKey?: string;
}

const panelStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  padding: "18px 20px",
  gap: "16px",
  minHeight: "320px",
};

const listStyle: CSSProperties = {
  display: "grid",
  gap: "12px",
};

const clippingPulse: CSSProperties = {
  boxShadow: `0 0 0 0 rgba(255, 71, 112, 0.65)`,
  animation: "clippingPulse 1s ease-out",
};

export const ActiveLoopsPanel = memo(function ActiveLoopsPanel({
  loops,
  onToggleMute,
  onFocusDeck,
  highlightKey,
}: ActiveLoopsPanelProps) {
  return (
    <section style={panelStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.92rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Active Loops
          </h2>
          <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
            Pinch to zoom waveform. Drag vertically to sweep filters.
          </p>
        </div>
        <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>{loops.length} running</span>
      </header>
      <div style={listStyle}>
        {loops.map((loop) => {
          const clipping = loop.level >= 0.92;
          const inKey = highlightKey ? loop.key === highlightKey : false;
          const borderColor = inKey ? theme.button.primary : theme.border;
          const meterGradient = `linear-gradient(90deg, rgba(99, 238, 205, 0.85) ${
            loop.level * 80
          }%, rgba(255, 97, 146, 0.65) ${loop.level * 100}%)`;
          return (
            <article
              key={loop.id}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1fr) auto",
                gap: "16px",
                padding: "14px",
                borderRadius: "12px",
                border: `1px solid ${borderColor}`,
                background: inKey ? "rgba(21, 61, 78, 0.95)" : "rgba(11, 39, 51, 0.85)",
                position: "relative",
                overflow: "hidden",
                ...(clipping ? clippingPulse : {}),
              }}
            >
              <div style={{ display: "grid", gap: "10px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "grid", gap: "2px" }}>
                    <strong style={{ fontSize: "0.86rem" }}>{loop.name}</strong>
                    <span
                      style={{ fontSize: "0.68rem", color: inKey ? theme.button.primary : theme.textMuted }}
                    >
                      {loop.bpm} BPM · Key {loop.key}
                    </span>
                  </div>
                  {loop.deck && (
                    <button
                      type="button"
                      style={{ ...toolbarButtonStyle, padding: "4px 10px" }}
                      onClick={() => onFocusDeck(loop.deck!)}
                    >
                      Focus Deck {loop.deck}
                    </button>
                  )}
                </div>
                <div
                  style={{
                    position: "relative",
                    borderRadius: "10px",
                    border: `1px solid rgba(120, 203, 220, 0.24)`,
                    background: "rgba(8, 29, 40, 0.6)",
                    padding: "6px 10px",
                    display: "grid",
                    gridTemplateColumns: "1fr 120px",
                    gap: "12px",
                    alignItems: "center",
                  }}
                >
                  <div style={{ height: "60px", width: "100%" }}>
                    <WaveformPreview
                      waveform={loop.waveform}
                      fillColor="rgba(108, 198, 216, 0.35)"
                      strokeColor="rgba(240, 170, 255, 0.8)"
                    />
                  </div>
                  <div style={{ display: "grid", gap: "6px" }}>
                    <div
                      style={{
                        height: "8px",
                        borderRadius: "999px",
                        background: meterGradient,
                        transition: "background 0.2s ease",
                      }}
                    />
                    <div style={{ fontSize: "0.65rem", color: theme.textMuted }}>
                      Energy {(loop.energy * 100).toFixed(0)}% · Filter {Math.round(loop.filter * 100)}%
                    </div>
                  </div>
                </div>
              </div>
              <div style={{ display: "grid", gap: "8px", alignContent: "start" }}>
                <button
                  type="button"
                  onClick={() => onToggleMute(loop.id)}
                  style={{
                    ...toolbarButtonStyle,
                    padding: "8px 14px",
                    background: loop.isMuted ? "rgba(50, 77, 90, 0.6)" : theme.button.primary,
                    color: loop.isMuted ? theme.textMuted : theme.button.primaryText,
                    borderColor: loop.isMuted ? theme.button.outline : theme.button.primary,
                  }}
                >
                  {loop.isMuted ? "Unmute" : "Mute"}
                </button>
                <div
                  style={{
                    fontSize: "0.65rem",
                    color: theme.textMuted,
                    display: "grid",
                    gap: "4px",
                  }}
                >
                  <span>Deck Slot: {loop.deck ?? "Free"}</span>
                  <span>Level {(loop.level * 100).toFixed(0)}%</span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
});
