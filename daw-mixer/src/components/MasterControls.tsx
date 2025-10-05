import { useEffect, useState } from "react";
import type { Project } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";

interface MasterControlsProps {
  project: Project;
}

const PROJECT_SCALES = [
  "C Major",
  "G Major",
  "D Major",
  "A Major",
  "E Major",
  "B Major",
  "F# Major",
  "C# Major",
  "F Major",
  "Bb Major",
  "Eb Major",
  "Ab Major",
  "Db Major",
  "Gb Major",
  "C Minor",
  "G Minor",
  "D Minor",
  "A Minor",
  "E Minor",
  "B Minor",
  "F# Minor",
  "C# Minor",
];

export function MasterControls({ project }: MasterControlsProps) {
  const { dispatch } = useProjectStore();
  const [bpm, setBpm] = useState(project.masterBpm);
  const [scale, setScale] = useState(project.scale);

  useEffect(() => {
    setBpm(project.masterBpm);
  }, [project.masterBpm]);

  useEffect(() => {
    setScale(project.scale);
  }, [project.scale]);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: theme.text
          }}
        >
          Master BPM
          <input
            type="number"
            value={bpm}
            onChange={(event) => {
              const value = Number(event.target.value);
              setBpm(value);
              dispatch({
                type: "register-control-change",
                target: { id: "master-bpm", label: "Master BPM", value, unit: "BPM" },
              });
            }}
            onBlur={() =>
              dispatch({
                type: "set-project",
                project: { ...project, masterBpm: bpm }
              })
            }
            style={{
              marginTop: "4px",
              padding: "6px 8px",
              borderRadius: "8px",
              border: `1px solid ${theme.border}`,
              fontSize: "0.85rem",
              width: "78px",
              background: theme.surfaceOverlay,
              color: theme.text
            }}
          />
        </label>
        <label
          style={{
            display: "flex",
            flexDirection: "column",
            fontSize: "0.65rem",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: theme.text
          }}
        >
          Project Scale
          <select
            value={scale}
            onChange={(event) => {
              const nextScale = event.target.value;
              setScale(nextScale);
              dispatch({
                type: "set-project",
                project: { ...project, scale: nextScale }
              });
            }}
            style={{
              marginTop: "4px",
              padding: "6px 8px",
              borderRadius: "8px",
              border: `1px solid ${theme.border}`,
              fontSize: "0.85rem",
              width: "110px",
              background: theme.surfaceOverlay,
              color: theme.text
            }}
          >
            {PROJECT_SCALES.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
      <button
        type="button"
        style={{
          border: `1px solid ${theme.button.outline}`,
          padding: "6px 12px",
          borderRadius: "999px",
          background: theme.button.primary,
          color: theme.button.primaryText,
          fontWeight: 600,
          fontSize: "0.7rem",
          cursor: "pointer",
          textTransform: "uppercase",
          letterSpacing: "0.06em"
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
