import { useState } from "react";
import type { Project, SampleClip } from "../types";
import { nanoid } from "nanoid";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";
import { createDefaultTrackEffects } from "../lib/effectPresets";

interface LiveLooperProps {
  project: Project;
}

export function LiveLooper({ project }: LiveLooperProps) {
  const { dispatch, currentProjectId } = useProjectStore();
  const [isArmed, setIsArmed] = useState(false);
  const [loopLength, setLoopLength] = useState(2);

  const scheduleRecording = () => {
    setIsArmed(true);
    setTimeout(() => {
      const now = Date.now();
      const fakeFile = new File([""], `Live-${now}.wav`, { type: "audio/wav" });
      const sample: SampleClip = {
        id: nanoid(),
        name: `Live take ${project.samples.length + 1}`,
        file: fakeFile,
        url: undefined,
        bpm: project.masterBpm,
        key: "1A",
        measures: Array.from({ length: loopLength }, (_, index) => ({
          id: nanoid(),
          start: index * (60 / project.masterBpm * 4),
          end: (index + 1) * (60 / project.masterBpm * 4),
          beatCount: 4,
          isDownbeat: true
        })),
        stems: [],
        position: project.samples.length * (60 / project.masterBpm * 4),
        length: loopLength * (60 / project.masterBpm * 4),
        isLooping: true,
        effects: createDefaultTrackEffects()
      };
      dispatch({ type: "add-sample", projectId: currentProjectId, sample });
      setIsArmed(false);
    }, 600);
  };

  return (
    <div
      style={{
        padding: "16px 20px",
        borderTop: `1px solid ${theme.divider}`,
        background: theme.surfaceOverlay,
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        color: theme.text
      }}
    >
      <h3 style={{ margin: 0, fontSize: "1rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        Live Looper
      </h3>
      <p style={{ margin: 0, fontSize: "0.8rem", color: theme.textMuted }}>
        Arm recording to capture the next downbeat. Perfect for layering ideas with Ableton Push style flow.
      </p>
      <label style={{ fontSize: "0.85rem", color: theme.text }}>
        Loop length (measures)
        <input
          type="number"
          min={1}
          max={8}
          value={loopLength}
          onChange={(event) => setLoopLength(Number(event.target.value))}
          style={{
            marginLeft: "8px",
            padding: "6px 10px",
            borderRadius: "10px",
            border: `1px solid ${theme.border}`,
            width: "80px",
            background: theme.surface,
            color: theme.text
          }}
        />
      </label>
      <button
        type="button"
        onClick={scheduleRecording}
        disabled={isArmed}
        style={{
          border: `1px solid ${theme.button.outline}`,
          padding: "12px 16px",
          borderRadius: "999px",
          background: isArmed ? theme.button.base : theme.button.primary,
          color: isArmed ? theme.text : theme.button.primaryText,
          fontWeight: 700,
          cursor: isArmed ? "not-allowed" : "pointer",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          opacity: isArmed ? 0.65 : 1,
          boxShadow: isArmed ? "none" : theme.cardGlow
        }}
      >
        {isArmed ? "Waiting for downbeat…" : "Record on 1"}
      </button>
    </div>
  );
}
