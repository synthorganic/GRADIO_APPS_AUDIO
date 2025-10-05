import { useEffect, useState } from "react";
import type { Project } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";

interface MasterControlsProps {
  project: Project;
}

const CAMELOT_ENTRIES = [
  { code: "1A", label: "Ab Minor" },
  { code: "2A", label: "Eb Minor" },
  { code: "3A", label: "Bb Minor" },
  { code: "4A", label: "F Minor" },
  { code: "5A", label: "C Minor" },
  { code: "6A", label: "G Minor" },
  { code: "7A", label: "D Minor" },
  { code: "8A", label: "A Minor" },
  { code: "9A", label: "E Minor" },
  { code: "10A", label: "B Minor" },
  { code: "11A", label: "F# Minor" },
  { code: "12A", label: "Db Minor" },
  { code: "1B", label: "B Major" },
  { code: "2B", label: "F# Major" },
  { code: "3B", label: "Db Major" },
  { code: "4B", label: "Ab Major" },
  { code: "5B", label: "Eb Major" },
  { code: "6B", label: "Bb Major" },
  { code: "7B", label: "F Major" },
  { code: "8B", label: "C Major" },
  { code: "9B", label: "G Major" },
  { code: "10B", label: "D Major" },
  { code: "11B", label: "A Major" },
  { code: "12B", label: "E Major" }
];

const CAMELOT_SCALES = CAMELOT_ENTRIES.map(({ code, label }) => `${code} (${label})`);

const LEGACY_SCALE_TO_CAMELOT = CAMELOT_ENTRIES.reduce<Record<string, string>>(
  (mapping, entry) => {
    mapping[entry.label] = `${entry.code} (${entry.label})`;
    return mapping;
  },
  {
    "C# Major": "3B (Db Major)",
    "Gb Major": "2B (F# Major)",
    "C# Minor": "12A (Db Minor)",
    "Gb Minor": "11A (F# Minor)",
    "F# Major": "2B (F# Major)",
    "F# Minor": "11A (F# Minor)",
    "Db Major": "3B (Db Major)",
    "Db Minor": "12A (Db Minor)",
  }
);

const PROJECT_SCALES = CAMELOT_SCALES;

function normalizeScale(value: string) {
  if (PROJECT_SCALES.includes(value)) {
    return value;
  }
  return LEGACY_SCALE_TO_CAMELOT[value] ?? value;
}

export function MasterControls({ project }: MasterControlsProps) {
  const { dispatch } = useProjectStore();
  const [bpm, setBpm] = useState(project.masterBpm);
  const [scale, setScale] = useState(() => normalizeScale(project.scale));

  useEffect(() => {
    setBpm(project.masterBpm);
  }, [project.masterBpm]);

  useEffect(() => {
    const normalized = normalizeScale(project.scale);
    setScale(normalized);
    if (normalized !== project.scale) {
      dispatch({
        type: "set-project",
        project: { ...project, scale: normalized }
      });
    }
  }, [dispatch, project, project.scale]);

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
              const normalized = normalizeScale(nextScale);
              setScale(normalized);
              dispatch({
                type: "set-project",
                project: { ...project, scale: normalized }
              });
            }}
            style={{
              marginTop: "4px",
              padding: "6px 8px",
              borderRadius: "8px",
              border: `1px solid ${theme.border}`,
              fontSize: "0.85rem",
              width: "160px",
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
