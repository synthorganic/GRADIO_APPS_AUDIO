import { theme } from "@daw-theme";
import { WaveformPreview } from "@daw-shared/WaveformPreview";
import { FxRack, type FxPluginDescriptor } from "@daw-shared/FxRack";

export interface DeckDescriptor {
  id: "A" | "B";
  loopTitle: string;
  key: string;
  waveform: Float32Array;
  color: string;
  loudness: number;
  plugins: FxPluginDescriptor[];
  onLoadPlugin: (file: File | null) => void;
}

export interface DualDeckPanelProps {
  decks: DeckDescriptor[];
}

export function DualDeckPanel({ decks }: DualDeckPanelProps) {
  return (
    <section
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "18px",
      }}
    >
      {decks.map((deck) => (
        <article
          key={deck.id}
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "14px",
            padding: "18px",
            borderRadius: "16px",
            background: theme.surfaceOverlay,
            border: `1px solid ${deck.color}`,
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
              gap: "12px",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <span
                style={{
                  fontSize: "0.68rem",
                  letterSpacing: "0.1em",
                  color: deck.color,
                  textTransform: "uppercase",
                }}
              >
                Deck {deck.id}
              </span>
              <strong style={{ fontSize: "0.95rem", letterSpacing: "0.06em" }}>
                {deck.loopTitle}
              </strong>
              <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{deck.key}</span>
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-end",
                gap: "4px",
                fontSize: "0.68rem",
                color: theme.textMuted,
              }}
            >
              <span>RMS</span>
              <div
                style={{
                  position: "relative",
                  width: "36px",
                  height: "36px",
                  borderRadius: "50%",
                  border: `2px solid ${deck.color}`,
                  display: "grid",
                  placeItems: "center",
                  boxShadow: `0 0 24px -12px ${deck.color}`,
                }}
              >
                <span style={{ fontSize: "0.72rem", color: deck.color }}>
                  {Math.round(deck.loudness * 12)}dB
                </span>
              </div>
            </div>
          </header>
          <div
            style={{
              position: "relative",
              height: "120px",
              borderRadius: "12px",
              overflow: "hidden",
              border: `1px solid ${deck.color}`,
              background: "rgba(255, 255, 255, 0.04)",
            }}
          >
            <WaveformPreview
              waveform={deck.waveform}
              fillColor={`${deck.color}33`}
              strokeColor={`${deck.color}66`}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                background:
                  "radial-gradient(circle at 15% 30%, rgba(255, 255, 255, 0.12), transparent 70%)",
                pointerEvents: "none",
              }}
            />
          </div>
          <FxRack
            title={`Deck ${deck.id} FX`}
            description="Drop AU/VST or drag from library"
            targetLabel="Routing"
            targetName={`Bus ${deck.id}`}
            actionLabel="Add FX"
            plugins={deck.plugins}
            emptyStateMessage="No processors assigned"
            onLoadPlugin={deck.onLoadPlugin}
          />
        </article>
      ))}
    </section>
  );
}
