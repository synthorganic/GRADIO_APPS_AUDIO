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
        padding: "16px",
        borderRadius: "16px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: theme.cardGlow,
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        color: theme.text
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "1.05rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            VST Rack
          </h2>
          <p style={{ margin: 0, fontSize: "0.75rem", color: theme.textMuted }}>
            Drop WebAssembly VSTs or effect presets to extend your collage toolkit.
          </p>
        </div>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          style={{
            border: `1px solid ${theme.button.outline}`,
            padding: "8px 12px",
            borderRadius: "10px",
            background: theme.button.primary,
            color: theme.button.primaryText,
            fontWeight: 600,
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
      <div style={{ fontSize: "0.8rem", color: theme.textMuted }}>
        Target: {targetSample ? targetSample.name : project.name}
      </div>
      {plugins.length === 0 ? (
        <p style={{ margin: 0, fontSize: "0.85rem", color: theme.textMuted }}>
          No VSTs loaded yet. Import compatible WASM or preset files to chain them after the mastering rack.
        </p>
      ) : (
        <ul style={{ margin: 0, paddingLeft: "18px", display: "grid", gap: "6px" }}>
          {plugins.map((plugin) => (
            <li key={plugin.id} style={{ fontSize: "0.85rem" }}>
              <strong>{plugin.name}</strong> — {plugin.type} ({plugin.fileName})
              <div style={{ fontSize: "0.75rem", color: theme.textMuted }}>
                Loaded {plugin.loadedAt.toLocaleTimeString()} · Routed to {targetSample ? targetSample.name : "master"}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
