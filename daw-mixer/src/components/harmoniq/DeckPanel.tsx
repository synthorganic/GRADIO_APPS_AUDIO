import type { CSSProperties } from "react";
import type { SampleClip } from "../../types";
import { theme } from "../../theme";
import { cardSurfaceStyle, toolbarButtonDisabledStyle, toolbarButtonStyle } from "../layout/styles";

export type DeckId = "A" | "B";

export type DeckPanelProps = {
  deckId: DeckId;
  assignedSample?: SampleClip | null;
  onRequestAssign: (deckId: DeckId) => void;
  onFocusDeck: (deckId: DeckId) => void;
  canAssign?: boolean;
};

const panelStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  gap: "10px",
  padding: "12px 14px",
  minHeight: "180px"
};

export function DeckPanel({ deckId, assignedSample, onRequestAssign, onFocusDeck, canAssign = true }: DeckPanelProps) {
  const durationLabel = assignedSample?.duration
    ? `${assignedSample.duration.toFixed(2)}s`
    : "Unknown length";
  const bpmLabel = assignedSample?.bpm ? `${assignedSample.bpm} BPM` : "No BPM";

  return (
    <section style={panelStyle}>
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "0.8rem",
          letterSpacing: "0.05em"
        }}
      >
        <strong>Deck {deckId}</strong>
        <button
          type="button"
          disabled={!canAssign}
          style={{
            ...toolbarButtonStyle,
            ...(canAssign ? {} : toolbarButtonDisabledStyle)
          }}
          onClick={() => onRequestAssign(deckId)}
        >
          Assign
        </button>
      </header>
      <div
        style={{
          flex: 1,
          display: "grid",
          gap: "6px",
          alignContent: "start",
          fontSize: "0.75rem",
          color: theme.textMuted
        }}
      >
        <span style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          Assigned Sample
        </span>
        {assignedSample ? (
          <div
            style={{
              padding: "10px",
              borderRadius: "10px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "6px"
            }}
          >
            <strong style={{ fontSize: "0.8rem" }}>{assignedSample.name}</strong>
            <span style={{ fontSize: "0.7rem" }}>
              {durationLabel} · {bpmLabel}
            </span>
          </div>
        ) : (
          <div
            style={{
              padding: "18px",
              borderRadius: "10px",
              border: `1px dashed ${theme.border}`,
              background: "rgba(18, 53, 67, 0.32)",
              textAlign: "center"
            }}
          >
            Select a sample and assign it to this deck to prepare a mix.
          </div>
        )}
      </div>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button type="button" style={toolbarButtonStyle} onClick={() => onFocusDeck(deckId)}>
          Focus Deck
        </button>
      </div>
    </section>
  );
}
