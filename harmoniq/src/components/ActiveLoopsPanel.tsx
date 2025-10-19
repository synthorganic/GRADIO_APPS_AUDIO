import { memo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import type { DeckId } from "../types";
import { WaveformPreview } from "./WaveformPreview";

export interface ActiveLoopSummary {
  id: string;
  name: string;
  bpm: number;
  key: string;
  deck: DeckId;
  waveform: Float32Array;
  energy: number;
  filter: number;
  level: number;
  isMuted: boolean;
}

interface ActiveLoopsPanelProps {
  loops: ActiveLoopSummary[];
  highlightKey?: string | null;
  onToggleMute?: (loopId: string) => void;
  onFocusDeck?: (deckId: DeckId) => void;
}

function formatPercentage(value: number) {
  return `${Math.round(value * 100)}%`;
}

export const ActiveLoopsPanel = memo(function ActiveLoopsPanel({
  loops,
  highlightKey,
  onToggleMute,
  onFocusDeck,
}: ActiveLoopsPanelProps) {
  return (
    <section
      style={{
        ...cardSurfaceStyle,
        padding: "18px 20px",
        display: "grid",
        gap: "16px",
        alignContent: "start",
      }}
    >
      <header style={{ display: "grid", gap: "6px" }}>
        <h2
          style={{
            margin: 0,
            fontSize: "0.92rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          Active Decks
        </h2>
        <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
          Monitor playing loops, focus decks for performance tweaks, and toggle mute for quick breakdowns.
        </p>
      </header>
      <div style={{ display: "grid", gap: "12px" }}>
        {loops.map((loop) => {
          const isHighlighted = highlightKey && loop.key === highlightKey;
          const muteLabel = loop.isMuted ? "Unmute" : "Mute";
          return (
            <article
              key={loop.id}
              style={{
                borderRadius: "16px",
                border: `1px solid ${
                  isHighlighted ? "rgba(103, 255, 230, 0.65)" : "rgba(120, 203, 220, 0.3)"
                }`,
                background: isHighlighted ? "rgba(10, 46, 62, 0.88)" : "rgba(8, 28, 38, 0.82)",
                padding: "14px",
                display: "grid",
                gap: "10px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "12px",
                  flexWrap: "wrap",
                }}
              >
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                    <span
                      style={{
                        padding: "4px 10px",
                        borderRadius: "999px",
                        background: loop.isMuted ? "rgba(120, 203, 220, 0.18)" : "rgba(103, 255, 230, 0.22)",
                        color: loop.isMuted ? theme.textMuted : theme.button.primaryText,
                        fontSize: "0.65rem",
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                      }}
                    >
                      Deck {loop.deck}
                    </span>
                    <strong style={{ fontSize: "0.86rem" }}>{loop.name}</strong>
                  </div>
                  <div style={{ fontSize: "0.66rem", color: theme.textMuted }}>
                    {loop.bpm} BPM · Key {loop.key}
                  </div>
                </div>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  <button
                    type="button"
                    onClick={() => onFocusDeck?.(loop.deck)}
                    style={{
                      ...toolbarButtonStyle,
                      padding: "6px 12px",
                      borderRadius: "10px",
                      background: "rgba(9, 30, 42, 0.75)",
                      color: theme.text,
                    }}
                  >
                    Focus Deck
                  </button>
                  <button
                    type="button"
                    onClick={() => onToggleMute?.(loop.id)}
                    style={{
                      ...toolbarButtonStyle,
                      padding: "6px 12px",
                      borderRadius: "10px",
                      background: loop.isMuted ? "rgba(120, 203, 220, 0.24)" : theme.button.base,
                      color: loop.isMuted ? theme.text : theme.button.primaryText,
                      borderColor: loop.isMuted ? "rgba(120, 203, 220, 0.45)" : theme.button.base,
                    }}
                  >
                    {muteLabel}
                  </button>
                </div>
              </div>
              <WaveformPreview
                waveform={loop.waveform}
                height={48}
                fillColor={loop.isMuted ? "rgba(120, 203, 220, 0.35)" : "rgba(103, 255, 230, 0.8)"}
              />
              <div style={{ display: "grid", gap: "8px" }}>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                  <Meter label="Energy" value={loop.energy} />
                  <Meter label="Filter" value={loop.filter} />
                  <Meter label="Level" value={loop.level} />
                </div>
              </div>
            </article>
          );
        })}
        {loops.length === 0 && (
          <div
            style={{
              border: "1px dashed rgba(120, 203, 220, 0.3)",
              borderRadius: "14px",
              padding: "20px",
              textAlign: "center",
              fontSize: "0.7rem",
              color: theme.textMuted,
            }}
          >
            Load a loop to any deck to see real-time performance stats here.
          </div>
        )}
      </div>
    </section>
  );
});

interface MeterProps {
  label: string;
  value: number;
}

function Meter({ label, value }: MeterProps) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div style={{ display: "grid", gap: "4px", minWidth: "120px" }}>
      <span style={{ fontSize: "0.64rem", color: theme.textMuted, letterSpacing: "0.06em" }}>{label}</span>
      <div
        style={{
          position: "relative",
          height: "8px",
          borderRadius: "999px",
          background: "rgba(9, 32, 45, 0.7)",
          overflow: "hidden",
        }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            transform: `scaleX(${clamped})`,
            transformOrigin: "left",
            background: "linear-gradient(90deg, rgba(103, 255, 230, 0.8) 0%, rgba(120, 203, 220, 0.4) 100%)",
          }}
        />
      </div>
      <span style={{ fontSize: "0.64rem", color: theme.button.primaryText }}>{formatPercentage(clamped)}</span>
    </div>
  );
}
