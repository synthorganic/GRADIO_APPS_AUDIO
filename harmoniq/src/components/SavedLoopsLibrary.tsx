import type { CSSProperties } from "react";
import { memo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import { WaveformPreview } from "@daw/shared/WaveformPreview";

export interface SavedLoopItem {
  id: string;
  name: string;
  bpm: number;
  key: string;
  waveform: Float32Array;
  mood: string;
}

export interface SavedLoopsLibraryProps {
  loops: SavedLoopItem[];
  onLoadLoop: (loopId: string) => void;
}

const containerStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  gap: "16px",
  padding: "18px 20px",
};

export const SavedLoopsLibrary = memo(function SavedLoopsLibrary({
  loops,
  onLoadLoop,
}: SavedLoopsLibraryProps) {
  return (
    <section style={containerStyle}>
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
            Saved Loops
          </h2>
          <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
            Drag to reorder, tap Load to push into the deck buffer.
          </p>
        </div>
        <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{loops.length} stored</span>
      </header>
      <div style={{ display: "grid", gap: "12px", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        {loops.map((loop) => (
          <article
            key={loop.id}
            style={{
              border: `1px solid rgba(120, 203, 220, 0.3)`,
              borderRadius: "14px",
              padding: "14px",
              background: "rgba(9, 33, 45, 0.82)",
              display: "grid",
              gap: "10px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <strong style={{ fontSize: "0.82rem" }}>{loop.name}</strong>
                <div style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                  {loop.bpm} BPM · {loop.key} · {loop.mood}
                </div>
              </div>
              <button
                type="button"
                onClick={() => onLoadLoop(loop.id)}
                style={{
                  ...toolbarButtonStyle,
                  padding: "6px 12px",
                  background: theme.button.primary,
                  color: theme.button.primaryText,
                  borderColor: theme.button.primary,
                }}
              >
                Load
              </button>
            </div>
            <div
              style={{
                height: "68px",
                borderRadius: "12px",
                border: `1px solid rgba(108, 198, 216, 0.32)`,
                background: "rgba(5, 22, 30, 0.6)",
                padding: "6px",
              }}
            >
              <WaveformPreview
                waveform={loop.waveform}
                fillColor="rgba(79, 197, 255, 0.32)"
                strokeColor="rgba(255, 148, 241, 0.75)"
              />
            </div>
          </article>
        ))}
      </div>
    </section>
  );
});
