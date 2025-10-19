import type { MouseEvent } from "react";
import { memo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import { WaveformPreview } from "./WaveformPreview";
import type { LoopLibraryItem } from "../state/LoopLibraryStore";

export interface TrackSelectionModalProps {
  keyLabel: string;
  loops: LoopLibraryItem[];
  onClose: () => void;
  onLoadLoop: (loopId: string) => void;
}

export const TrackSelectionModal = memo(function TrackSelectionModal({
  keyLabel,
  loops,
  onClose,
  onLoadLoop,
}: TrackSelectionModalProps) {
  const handleOverlayClick = () => {
    onClose();
  };

  const handleContentClick = (event: MouseEvent<HTMLDivElement>) => {
    event.stopPropagation();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Tracks in key ${keyLabel}`}
      onClick={handleOverlayClick}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(3, 14, 22, 0.78)",
        display: "grid",
        placeItems: "center",
        zIndex: 40,
        padding: "24px",
      }}
    >
      <div
        onClick={handleContentClick}
        style={{
          ...cardSurfaceStyle,
          width: "min(760px, 92vw)",
          maxHeight: "80vh",
          overflow: "hidden",
          display: "grid",
          gridTemplateRows: "auto 1fr",
          gap: "18px",
          padding: "24px",
          position: "relative",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "16px",
          }}
        >
          <div>
            <h2
              style={{
                margin: 0,
                fontSize: "1rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
              }}
            >
              Key {keyLabel} selections
            </h2>
            <p style={{ margin: "6px 0 0", fontSize: "0.72rem", color: theme.textMuted }}>
              Choose a track to drop on the focused deck. Stems will adopt the deck tempo at the next bar.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              ...toolbarButtonStyle,
              padding: "6px 12px",
              fontSize: "0.7rem",
            }}
          >
            Close
          </button>
        </div>
        <div
          style={{
            overflowY: "auto",
            display: "grid",
            gap: "12px",
            paddingRight: "6px",
          }}
        >
          {loops.length === 0 ? (
            <div
              style={{
                borderRadius: "12px",
                border: "1px solid rgba(120, 203, 220, 0.32)",
                padding: "20px",
                background: "rgba(9, 33, 45, 0.82)",
                fontSize: "0.75rem",
                color: theme.textMuted,
                textAlign: "center",
              }}
            >
              No saved loops are tagged with Camelot key {keyLabel}. Import or resample a loop to populate this slot.
            </div>
          ) : (
            loops.map((loop) => (
              <article
                key={loop.id}
                style={{
                  borderRadius: "14px",
                  border: "1px solid rgba(120, 203, 220, 0.32)",
                  padding: "16px",
                  background: "rgba(9, 32, 44, 0.85)",
                  display: "grid",
                  gap: "12px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "grid", gap: "4px" }}>
                    <strong style={{ fontSize: "0.88rem" }}>{loop.name}</strong>
                    <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                      {loop.bpm} BPM · {loop.mood}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onLoadLoop(loop.id)}
                    style={{
                      ...toolbarButtonStyle,
                      padding: "6px 14px",
                      background: theme.button.primary,
                      color: theme.button.primaryText,
                      borderColor: theme.button.primary,
                    }}
                  >
                    Load to deck
                  </button>
                </div>
                <div
                  style={{
                    height: "80px",
                    borderRadius: "12px",
                    border: "1px solid rgba(108, 198, 216, 0.32)",
                    background: "rgba(5, 22, 30, 0.6)",
                    padding: "8px",
                  }}
                >
                  <WaveformPreview
                    waveform={loop.waveform}
                    fillColor="rgba(132, 94, 255, 0.35)"
                    strokeColor="rgba(255, 148, 241, 0.75)"
                  />
                </div>
              </article>
            ))
          )}
        </div>
      </div>
    </div>
  );
});
