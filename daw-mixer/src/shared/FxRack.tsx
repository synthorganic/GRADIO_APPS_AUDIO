import { useRef } from "react";
import { theme } from "../theme";

export type FxPluginType = "synth" | "effect" | "utility";

export interface FxPluginDescriptor {
  id: string;
  name: string;
  type: FxPluginType;
  loadedAt: Date;
  fileName?: string;
}

export interface FxRackProps {
  title: string;
  description: string;
  targetLabel?: string;
  targetName: string;
  actionLabel?: string;
  accept?: string;
  plugins: FxPluginDescriptor[];
  emptyStateMessage: string;
  onLoadPlugin: (file: File | null) => void;
}

export function FxRack({
  title,
  description,
  targetLabel = "Target",
  targetName,
  actionLabel = "Load FX",
  accept = ".wasm,.json,.vstpreset",
  plugins,
  emptyStateMessage,
  onLoadPlugin,
}: FxRackProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div
      style={{
        padding: "14px",
        borderRadius: "12px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        color: theme.text,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            {title}
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>{description}</p>
        </div>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `1px solid ${theme.button.outline}`,
            padding: "6px 10px",
            borderRadius: "8px",
            background: theme.button.primary,
            color: theme.button.primaryText,
            fontWeight: 600,
            fontSize: "0.7rem",
            cursor: "pointer",
            boxShadow: theme.cardGlow,
          }}
        >
          {actionLabel}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            onLoadPlugin(file);
            if (fileInputRef.current) {
              fileInputRef.current.value = "";
            }
          }}
        />
      </div>
      <div style={{ fontSize: "0.7rem", color: theme.textMuted }}>
        {targetLabel}: {targetName}
      </div>
      {plugins.length === 0 ? (
        <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>{emptyStateMessage}</p>
      ) : (
        <ul style={{ margin: 0, paddingLeft: "18px", display: "grid", gap: "4px" }}>
          {plugins.map((plugin) => (
            <li key={plugin.id} style={{ fontSize: "0.72rem" }}>
              <strong>{plugin.name}</strong> — {plugin.type}
              {plugin.fileName ? ` (${plugin.fileName})` : null}
              <div style={{ fontSize: "0.65rem", color: theme.textMuted }}>
                Loaded {plugin.loadedAt.toLocaleTimeString()} · Routed to {targetName}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function deriveFxPluginType(file: File): FxPluginType {
  if (file.name.endsWith(".wasm")) {
    return "synth";
  }
  if (file.name.endsWith(".json") || file.name.endsWith(".vstpreset")) {
    return "effect";
  }
  return "utility";
}
