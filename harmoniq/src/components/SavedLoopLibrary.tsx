import { theme } from "@daw-theme";
import { WaveformPreview } from "@daw-shared/WaveformPreview";

export interface SavedLoopDescriptor {
  id: string;
  title: string;
  key: string;
  mood: string;
  tags: string[];
  waveform: Float32Array;
}

export interface SavedLoopLibraryProps {
  loops: SavedLoopDescriptor[];
  onRecall: (loopId: string) => void;
}

export function SavedLoopLibrary({ loops, onRecall }: SavedLoopLibraryProps) {
  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
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
          alignItems: "center",
          gap: "10px",
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
            Saved Loops
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
            Double-tap to audition · Hold to favorite
          </p>
        </div>
        <button
          type="button"
          style={{
            borderRadius: "8px",
            padding: "6px 12px",
            border: `1px solid ${theme.button.outline}`,
            background: theme.surface,
            color: theme.text,
            fontSize: "0.7rem",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Load
        </button>
      </header>
      <div
        style={{
          display: "grid",
          gap: "12px",
        }}
      >
        {loops.map((loop) => (
          <article
            key={loop.id}
            style={{
              display: "grid",
              gap: "10px",
              padding: "14px",
              borderRadius: "12px",
              border: `1px solid ${theme.border}`,
              background: theme.surface,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: "10px",
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <strong style={{ fontSize: "0.82rem", letterSpacing: "0.06em" }}>
                  {loop.title}
                </strong>
                <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                  {loop.key} · {loop.mood}
                </span>
              </div>
              <button
                type="button"
                onClick={() => onRecall(loop.id)}
                style={{
                  borderRadius: "999px",
                  padding: "6px 14px",
                  border: `1px solid ${theme.button.outline}`,
                  background: theme.button.base,
                  color: theme.text,
                  fontSize: "0.68rem",
                  fontWeight: 600,
                  letterSpacing: "0.06em",
                  cursor: "pointer",
                }}
              >
                Recall
              </button>
            </div>
            <div
              style={{
                position: "relative",
                height: "64px",
                borderRadius: "10px",
                overflow: "hidden",
                border: `1px solid ${theme.border}`,
                background: "rgba(255, 255, 255, 0.04)",
              }}
            >
              <WaveformPreview
                waveform={loop.waveform}
                fillColor="rgba(244, 227, 165, 0.24)"
                strokeColor="rgba(244, 227, 165, 0.46)"
              />
            </div>
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
              {loop.tags.map((tag) => (
                <span
                  key={tag}
                  style={{
                    fontSize: "0.62rem",
                    padding: "4px 8px",
                    borderRadius: "999px",
                    border: `1px solid ${theme.button.outline}`,
                    color: theme.textMuted,
                    letterSpacing: "0.04em",
                  }}
                >
                  #{tag}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
