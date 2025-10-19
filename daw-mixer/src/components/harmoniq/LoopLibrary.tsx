import type { CSSProperties } from "react";
import type { SampleClip } from "../../types";
import { theme } from "../../theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "../layout/styles";
import type { DeckId } from "./DeckPanel";

export type LoopLibraryProps = {
  loops: SampleClip[];
  selectedKey?: string | null;
  onPreviewLoop?: (loopId: string) => void;
  onAssignToDeck?: (deckId: DeckId, loopId: string) => void;
};

const containerStyle: CSSProperties = {
  ...cardSurfaceStyle,
  display: "grid",
  gap: "12px",
  padding: "12px 16px"
};

export function LoopLibrary({ loops, selectedKey, onPreviewLoop, onAssignToDeck }: LoopLibraryProps) {
  return (
    <section style={containerStyle}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong style={{ fontSize: "0.8rem", letterSpacing: "0.05em" }}>Loop Library</strong>
        <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{loops.length} items</span>
      </header>
      {selectedKey && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.68rem",
            color: theme.textMuted
          }}
        >
          <span>Filtering by key: {selectedKey}</span>
          <span>
            {loops.length > 0
              ? "Ready for harmonic mixing"
              : "Drop clips tagged with this key"}
          </span>
        </div>
      )}
      <div style={{ display: "grid", gap: "10px", maxHeight: "260px", overflowY: "auto" }}>
        {loops.map((loop) => (
          <div
            key={loop.id}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              alignItems: "center",
              gap: "10px",
              padding: "10px 0",
              borderBottom: `1px solid rgba(120, 203, 220, 0.2)`
            }}
          >
            <div style={{ display: "grid", gap: "4px" }}>
              <strong style={{ fontSize: "0.78rem" }}>{loop.name}</strong>
              <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                {loop.bpm ? `${loop.bpm} BPM` : "Unquantized"} · {loop.key ?? "No key"}
              </span>
            </div>
            <div style={{ display: "flex", gap: "6px" }}>
              {onPreviewLoop && (
                <button type="button" style={toolbarButtonStyle} onClick={() => onPreviewLoop(loop.id)}>
                  Preview
                </button>
              )}
              {onAssignToDeck && (
                <div style={{ display: "flex", gap: "6px" }}>
                  <button type="button" style={toolbarButtonStyle} onClick={() => onAssignToDeck("A", loop.id)}>
                    To Deck A
                  </button>
                  <button type="button" style={toolbarButtonStyle} onClick={() => onAssignToDeck("B", loop.id)}>
                    To Deck B
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
        {loops.length === 0 && (
          <div
            style={{
              padding: "20px",
              borderRadius: "10px",
              border: `1px dashed ${theme.border}`,
              background: "rgba(18, 53, 67, 0.32)",
              textAlign: "center",
              color: theme.textMuted
            }}
          >
            {selectedKey ? (
              `No loops detected in the ${selectedKey} bin yet. Upload audio in that key or clear the filter to browse everything.`
            ) : (
              "Drop loops into your project to populate the performance library."
            )}
          </div>
        )}
      </div>
    </section>
  );
}
