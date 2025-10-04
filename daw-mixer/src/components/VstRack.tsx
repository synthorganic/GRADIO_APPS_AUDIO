import { useRef, useState } from "react";
import { nanoid } from "nanoid";
import type { Project, SampleClip } from "../types";
import { theme } from "../theme";

interface VstRackProps {
  project: Project;
  targetSample?: SampleClip | null;
}

interface VstPlugin {
  id: string;
  name: string;
  type: "synth" | "effect";
  loadedAt: Date;
  fileName?: string;
}

export function VstRack({ project, targetSample }: VstRackProps) {
  const [plugins, setPlugins] = useState<VstPlugin[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const addPlugin = (file: File | null) => {
    if (!file) return;
    const plugin: VstPlugin = {
      id: nanoid(),
      name: file.name.replace(/\.[^.]+$/, ""),
      type: file.name.endsWith(".wasm") ? "synth" : "effect",
      loadedAt: new Date(),
      fileName: file.name
    };
    setPlugins((prev) => [...prev, plugin]);
  };

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
        color: theme.text
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            VST Rack
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
            Drop WebAssembly VSTs or effect presets to extend your collage toolkit.
          </p>
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
            boxShadow: theme.cardGlow
          }}
        >
          Load VST
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".wasm,.json,.vstpreset"
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.target.files?.[0] ?? null;
            addPlugin(file);
            if (fileInputRef.current) fileInputRef.current.value = "";
          }}
        />
      </div>
      <div style={{ fontSize: "0.7rem", color: theme.textMuted }}>
        Target: {targetSample ? targetSample.name : project.name}
      </div>
      {plugins.length === 0 ? (
        <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
          No VSTs loaded yet. Import compatible WASM or preset files to chain them after the mastering rack.
        </p>
      ) : (
        <ul style={{ margin: 0, paddingLeft: "18px", display: "grid", gap: "4px" }}>
          {plugins.map((plugin) => (
            <li key={plugin.id} style={{ fontSize: "0.72rem" }}>
              <strong>{plugin.name}</strong> — {plugin.type} ({plugin.fileName})
              <div style={{ fontSize: "0.65rem", color: theme.textMuted }}>
                Loaded {plugin.loadedAt.toLocaleTimeString()} · Routed to {targetSample ? targetSample.name : "master"}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
