import type { Project } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";

const masteringDescriptors: Array<{
  key: keyof Project["mastering"];
  label: string;
  description: string;
  accent: string;
}> = [
  {
    key: "widenStereo",
    label: "Stereo Bloom",
    description: "Spread the mix subtly to emulate mid/side widening.",
    accent: theme.accentBeam[0]
  },
  {
    key: "glueCompression",
    label: "Glue Compressor",
    description: "Bus compression with responsive attack for collage cohesion.",
    accent: theme.accentBeam[1]
  },
  {
    key: "spectralTilt",
    label: "Spectral Tilt",
    description: "Balance brightness vs warmth. 0 = neutral, 1 = bright.",
    accent: theme.accentBeam[2]
  },
  {
    key: "limiterCeiling",
    label: "Limiter Ceiling",
    description: "Push loudness while preserving headroom.",
    accent: theme.accentBeam[3]
  },
  {
    key: "tapeSaturation",
    label: "Tape Satin",
    description: "Analog-inspired sheen to glue the palette together.",
    accent: theme.accentBeam[4]
  }
];

interface MasteringPanelProps {
  project: Project;
  variant?: "card" | "inline";
}

export function MasteringPanel({ project, variant = "card" }: MasteringPanelProps) {
  const { dispatch, currentProjectId } = useProjectStore();
  const isInline = variant === "inline";

  return (
    <div
      style={{
        padding: isInline ? "12px 14px" : "16px",
        borderRadius: isInline ? "12px" : "16px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: isInline ? "none" : theme.cardGlow,
        display: "flex",
        flexDirection: "column",
        gap: isInline ? "12px" : "16px",
        color: theme.text
      }}
    >
      <div>
        <h2
          style={{
            margin: 0,
            fontSize: isInline ? "0.9rem" : "1.1rem",
            textTransform: "uppercase",
            letterSpacing: "0.06em"
          }}
        >
          Mastering Rack
        </h2>
        <p style={{ margin: 0, fontSize: isInline ? "0.7rem" : "0.8rem", color: theme.textMuted }}>
          Borrowed from the mastering tab: automate width, glue, tonal tilt, limiting, and saturation.
        </p>
      </div>
      {masteringDescriptors.map((item) => (
        <label
          key={item.key}
          style={{ display: "flex", flexDirection: "column", gap: isInline ? "4px" : "6px" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontWeight: 600, fontSize: "0.75rem" }}>{item.label}</span>
            <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
              {project.mastering[item.key].toFixed(2)}
            </span>
          </div>
          <div style={{ fontSize: "0.68rem", color: theme.textMuted }}>{item.description}</div>
          <input
            type="range"
            min={item.key === "limiterCeiling" ? -2 : 0}
            max={item.key === "limiterCeiling" ? 0 : 1}
            step={0.01}
            value={project.mastering[item.key]}
            onChange={(event) =>
              dispatch({
                type: "update-mastering",
                projectId: currentProjectId,
                payload: { [item.key]: Number(event.target.value) }
              })
            }
            style={{ accentColor: undefined, background: item.accent, height: "4px", borderRadius: "999px" }}
          />
        </label>
      ))}
      <button
        type="button"
        style={{
          border: `1px solid ${theme.button.outline}`,
          padding: isInline ? "8px 12px" : "12px 16px",
          borderRadius: "999px",
          background: theme.button.primary,
          color: theme.button.primaryText,
          fontWeight: 700,
          fontSize: "0.75rem",
          cursor: "pointer",
          boxShadow: isInline ? "none" : theme.cardGlow
        }}
        onClick={() => {
          const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const anchor = document.createElement("a");
          anchor.href = url;
          anchor.download = `${project.name.replace(/\s+/g, "-")}.json`;
          anchor.click();
          URL.revokeObjectURL(url);
        }}
      >
        Save project snapshot
      </button>
    </div>
  );
}
