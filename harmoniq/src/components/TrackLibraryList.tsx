import { theme } from "@daw/theme";
import type { DeckId } from "./DeckMatrix";
import type { AnalyzedTrackSummary } from "./TrackUploadPanel";

export interface TrackLibraryListProps {
  tracks: AnalyzedTrackSummary[];
  onLoad: (deckId: DeckId, track: AnalyzedTrackSummary) => void;
}

const deckTargets: DeckId[] = ["A", "B", "C", "D"];

export function TrackLibraryList({ tracks, onLoad }: TrackLibraryListProps) {
  if (!tracks.length) {
    return (
      <div
        style={{
          borderRadius: "14px",
          border: "1px solid rgba(120, 203, 220, 0.22)",
          padding: "18px",
          background: "rgba(6, 24, 34, 0.68)",
          color: theme.textMuted,
          fontSize: "0.72rem",
        }}
      >
        Upload tracks to build your library. Beat grids and scales will appear here for quick loading.
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gap: "12px" }}>
      {tracks.map((track) => (
        <article
          key={track.id}
          style={{
            borderRadius: "14px",
            border: "1px solid rgba(120, 203, 220, 0.25)",
            padding: "16px",
            background: "rgba(8, 30, 44, 0.82)",
            display: "grid",
            gap: "10px",
          }}
        >
          <header style={{ display: "flex", justifyContent: "space-between", gap: "12px" }}>
            <div>
              <strong style={{ fontSize: "0.88rem" }}>{track.name}</strong>
              <div style={{ fontSize: "0.68rem", color: theme.textMuted }}>{track.origin}</div>
            </div>
            <div style={{ display: "grid", gap: "4px", textAlign: "right" }}>
              <span style={{ fontSize: "0.7rem", color: theme.button.primaryText }}>
                {track.bpm} BPM
              </span>
              <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>Scale {track.scale}</span>
            </div>
          </header>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {track.stems.map((stem) => (
              <span
                key={stem.id}
                style={{
                  padding: "6px 10px",
                  borderRadius: "999px",
                  border: "1px solid rgba(103, 255, 230, 0.3)",
                  background: "rgba(13, 46, 64, 0.75)",
                  fontSize: "0.65rem",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  color: theme.button.primaryText,
                }}
              >
                {stem.label}
              </span>
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {deckTargets.map((deck) => (
              <button
                key={deck}
                type="button"
                onClick={() => onLoad(deck, track)}
                style={{
                  padding: "8px 12px",
                  borderRadius: "10px",
                  border: "1px solid rgba(120, 203, 220, 0.3)",
                  background: "rgba(10, 36, 48, 0.75)",
                  color: theme.button.primaryText,
                  fontSize: "0.7rem",
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                Load Deck {deck}
              </button>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
