import type { CSSProperties } from "react";
import { theme } from "@daw-theme";
import { WaveformPreview } from "@daw-shared/WaveformPreview";

export interface ActiveLoopDescriptor {
  id: string;
  title: string;
  key: string;
  bpm: number;
  length: string;
  energy: number;
  waveform: Float32Array;
  assignedDeck: "A" | "B" | null;
}

export interface ActiveLoopsPanelProps {
  loops: ActiveLoopDescriptor[];
  onAssign: (deck: "A" | "B", loopId: string) => void;
}

export function ActiveLoopsPanel({ loops, onAssign }: ActiveLoopsPanelProps) {
  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "14px",
        padding: "18px",
        borderRadius: "16px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: theme.cardGlow,
        color: theme.text,
        minWidth: 0,
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: "12px",
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.95rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Active Loops
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
            Drag to resequence · Tap FX to stack processors
          </p>
        </div>
        <span
          style={{
            fontSize: "0.7rem",
            padding: "4px 10px",
            borderRadius: "999px",
            border: `1px solid ${theme.button.outline}`,
            background: theme.surface,
            color: theme.textMuted,
            letterSpacing: "0.06em",
          }}
        >
          LIVE
        </span>
      </header>
      <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
        {loops.map((loop) => {
          const accent =
            loop.assignedDeck === "A"
              ? theme.accentBeam[0]
              : loop.assignedDeck === "B"
              ? theme.accentBeam[4]
              : theme.button.outline;
          return (
            <article
              key={loop.id}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1fr)",
                gap: "10px",
                padding: "14px",
                borderRadius: "12px",
                background:
                  loop.assignedDeck != null
                    ? `linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0))`
                    : theme.surface,
                border: `1px solid ${accent}`,
                transition: "border-color 150ms ease, transform 150ms ease",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: "12px",
                  flexWrap: "wrap",
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <strong style={{ fontSize: "0.85rem", letterSpacing: "0.04em" }}>
                    {loop.title}
                  </strong>
                  <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                    {loop.key} · {loop.bpm} BPM · {loop.length}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => onAssign("A", loop.id)}
                  style={getAssignButtonStyle("A", loop.assignedDeck === "A")}
                >
                  Load A
                </button>
                <button
                  type="button"
                  onClick={() => onAssign("B", loop.id)}
                  style={getAssignButtonStyle("B", loop.assignedDeck === "B")}
                >
                  Load B
                </button>
              </div>
              <div
                style={{
                  position: "relative",
                  height: "84px",
                  borderRadius: "10px",
                  overflow: "hidden",
                  border: `1px solid ${theme.border}`,
                  background: "rgba(255, 255, 255, 0.05)",
                }}
              >
                <WaveformPreview
                  waveform={loop.waveform}
                  fillColor="rgba(120, 203, 220, 0.28)"
                  strokeColor="rgba(120, 203, 220, 0.48)"
                />
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    bottom: 0,
                    height: "6px",
                    width: `${Math.round(loop.energy * 100)}%`,
                    background: accent,
                    transition: "width 200ms ease",
                  }}
                />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: "0.7rem",
                  color: theme.textMuted,
                }}
              >
                <span>Energy</span>
                <button
                  type="button"
                  style={{
                    borderRadius: "999px",
                    padding: "4px 12px",
                    border: `1px solid ${theme.button.outline}`,
                    background: "transparent",
                    color: theme.text,
                    fontWeight: 600,
                    letterSpacing: "0.06em",
                    cursor: "pointer",
                  }}
                >
                  FX
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function getAssignButtonStyle(deck: "A" | "B", isActive: boolean): CSSProperties {
  const accent = deck === "A" ? theme.accentBeam[0] : theme.accentBeam[4];
  return {
    borderRadius: "999px",
    padding: "6px 12px",
    border: `1px solid ${accent}`,
    background: isActive ? accent : "transparent",
    color: isActive ? theme.button.primaryText : accent,
    fontSize: "0.68rem",
    fontWeight: 600,
    letterSpacing: "0.08em",
    cursor: "pointer",
    transition: "background 150ms ease, color 150ms ease",
  };
}
