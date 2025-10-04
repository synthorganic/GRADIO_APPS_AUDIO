import { useState } from "react";
import type { Project } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";

interface MasterControlsProps {
  project: Project;
}

export function MasterControls({ project }: MasterControlsProps) {
  const { dispatch } = useProjectStore();
  const [bpm, setBpm] = useState(project.masterBpm);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
      <label
        style={{
          display: "flex",
          flexDirection: "column",
          fontSize: "0.75rem",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: theme.text
        }}
      >
        Master BPM
        <input
          type="number"
          value={bpm}
          onChange={(event) => setBpm(Number(event.target.value))}
          onBlur={() =>
            dispatch({
              type: "set-project",
              project: { ...project, masterBpm: bpm }
            })
          }
          style={{
            marginTop: "4px",
            padding: "8px 12px",
            borderRadius: "10px",
            border: `1px solid ${theme.border}`,
            fontSize: "1rem",
            width: "100px",
            background: theme.surfaceOverlay,
            color: theme.text
          }}
        />
      </label>
      <button
        type="button"
        style={{
          border: `1px solid ${theme.button.outline}`,
          padding: "10px 16px",
          borderRadius: "999px",
          background: theme.button.primary,
          color: theme.button.primaryText,
          fontWeight: 700,
          cursor: "pointer",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          boxShadow: theme.cardGlow
        }}
        onClick={() => {
          dispatch({
            type: "set-project",
            project: {
              ...project,
              masterBpm: bpm,
              samples: project.samples.map((sample) => ({
                ...sample,
                position: Math.round(sample.position / (60 / bpm * 4)) * (60 / bpm * 4)
              }))
            }
          });
        }}
      >
        Quantize grid
      </button>
    </div>
  );
}
