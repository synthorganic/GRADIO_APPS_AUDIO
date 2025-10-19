import { memo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import { WaveformPreview } from "./WaveformPreview";

export interface SavedLoopPreview {
  id: string;
  name: string;
  bpm: number;
  key: string;
  waveform: Float32Array;
  mood: string;
}

interface SavedLoopsLibraryProps {
  loops: SavedLoopPreview[];
  onLoadLoop: (loopId: string) => void;
}

export const SavedLoopsLibrary = memo(function SavedLoopsLibrary({
  loops,
  onLoadLoop,
}: SavedLoopsLibraryProps) {
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
          Saved Loop Library
        </h2>
        <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
          Recall curated loops with key and tempo metadata. Loading a loop will focus the active deck.
        </p>
      </header>
      <div style={{ display: "grid", gap: "12px" }}>
        {loops.map((loop) => (
          <article
            key={loop.id}
            style={{
              borderRadius: "16px",
              border: "1px solid rgba(120, 203, 220, 0.28)",
              background: "rgba(8, 28, 38, 0.82)",
              padding: "14px",
              display: "grid",
              gap: "10px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", flexWrap: "wrap" }}>
              <div>
                <strong style={{ fontSize: "0.86rem" }}>{loop.name}</strong>
                <div style={{ fontSize: "0.66rem", color: theme.textMuted }}>Mood · {loop.mood}</div>
              </div>
              <div style={{ textAlign: "right", display: "grid", gap: "4px" }}>
                <span style={{ fontSize: "0.7rem", color: theme.button.primaryText }}>{loop.bpm} BPM</span>
                <span
                  style={{
                    fontSize: "0.7rem",
                    color: "rgba(120, 203, 220, 0.8)",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                  }}
                >
                  Key {loop.key}
                </span>
              </div>
            </div>
            <WaveformPreview waveform={loop.waveform} height={56} />
            <button
              type="button"
              onClick={() => onLoadLoop(loop.id)}
              style={{
                ...toolbarButtonStyle,
                justifySelf: "start",
                padding: "8px 16px",
                borderRadius: "12px",
                background: theme.button.primary,
                color: theme.button.primaryText,
                borderColor: theme.button.primary,
              }}
            >
              Load into focused deck
            </button>
          </article>
        ))}
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
            Saved loops will appear here after importing or recording new material. Use the upload panel to seed your library.
          </div>
        )}
      </div>
    </section>
  );
});
