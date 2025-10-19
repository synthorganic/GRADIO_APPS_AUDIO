import { memo, useRef } from "react";
import type { ChangeEvent } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import type { AutomationEnvelope, LoopLibraryItem } from "../state/LoopLibraryStore";

export interface AutomationEnvelopeSummary extends AutomationEnvelope {
  linkedLoops: LoopLibraryItem[];
}

interface AutomationEnvelopeListProps {
  envelopes: AutomationEnvelopeSummary[];
  onExport?: () => void;
  onImport?: (file: File) => void;
  lastPersistedAt: number | null;
  persistError?: string | null;
}

function formatTimestamp(timestamp: number | null) {
  if (!timestamp) return "Never";
  const date = new Date(timestamp);
  return date.toLocaleString();
}

export const AutomationEnvelopeList = memo(function AutomationEnvelopeList({
  envelopes,
  onExport,
  onImport,
  lastPersistedAt,
  persistError,
}: AutomationEnvelopeListProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleImportClick = () => {
    if (onImport) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = "";
    if (file && onImport) {
      await onImport(file);
    }
  };

  return (
    <section
      style={{
        ...cardSurfaceStyle,
        display: "grid",
        gap: "16px",
        padding: "18px 20px",
        alignContent: "start",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "12px",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.92rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Automation Envelopes
          </h2>
          <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
            Export JSON snapshots or import from disk to sync sessions across devices.
          </p>
          <div style={{ fontSize: "0.64rem", color: theme.textMuted, marginTop: "6px" }}>
            Last saved: <strong style={{ color: theme.button.primaryText }}>{formatTimestamp(lastPersistedAt)}</strong>
            {persistError && (
              <span style={{ color: "#ff6384", marginLeft: "6px" }}>
                · {persistError}
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <button
            type="button"
            onClick={handleImportClick}
            style={{ ...toolbarButtonStyle, padding: "6px 12px" }}
          >
            Import JSON
          </button>
          <button
            type="button"
            onClick={onExport}
            style={{
              ...toolbarButtonStyle,
              padding: "6px 12px",
              background: theme.button.primary,
              color: theme.button.primaryText,
              borderColor: theme.button.primary,
            }}
          >
            Export JSON
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="application/json"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
      </header>
      <div style={{ display: "grid", gap: "12px" }}>
        {envelopes.map((envelope) => {
          const assignments = envelope.linkedLoops.map((loop) => loop.name).join(", ") || "Unassigned";
          const totalPoints = envelope.points.length;
          const duration = `${envelope.lengthBeats} beat${envelope.lengthBeats === 1 ? "" : "s"}`;
          return (
            <article
              key={envelope.id}
              style={{
                border: "1px solid rgba(120, 203, 220, 0.3)",
                borderRadius: "14px",
                padding: "14px",
                background: "rgba(8, 28, 38, 0.78)",
                display: "grid",
                gap: "10px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <strong style={{ fontSize: "0.82rem" }}>{envelope.name}</strong>
                  <div style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                    Target: {envelope.target} · Deck: {envelope.deckId ?? "Global"}
                  </div>
                </div>
                <span
                  style={{
                    padding: "4px 10px",
                    borderRadius: "999px",
                    background: envelope.color,
                    color: "rgba(6, 18, 26, 0.95)",
                    fontSize: "0.65rem",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                  }}
                >
                  {totalPoints} pts
                </span>
              </div>
              <div style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                Resolution {envelope.resolution} · Span {duration}
              </div>
              <div style={{ display: "grid", gap: "4px" }}>
                <span style={{ fontSize: "0.68rem", color: theme.button.primaryText }}>Linked loops</span>
                <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>{assignments}</span>
              </div>
            </article>
          );
        })}
        {envelopes.length === 0 && (
          <div
            style={{
              border: "1px dashed rgba(120, 203, 220, 0.3)",
              borderRadius: "12px",
              padding: "18px",
              textAlign: "center",
              fontSize: "0.72rem",
              color: theme.textMuted,
            }}
          >
            No envelopes have been authored yet. Exporting will capture newly created automation lanes once they are added.
          </div>
        )}
      </div>
    </section>
  );
});
