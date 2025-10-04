import { useMemo, useState } from "react";
import type { SampleClip } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { theme } from "../theme";

interface SampleDetailPanelProps {
  sample: SampleClip | null;
}

export function SampleDetailPanel({ sample }: SampleDetailPanelProps) {
  const { currentProjectId, dispatch } = useProjectStore();
  const [nudge, setNudge] = useState(0);

  const rekeyTimestamp = useMemo(() => {
    if (!sample?.rekeyedAt) return null;
    const date = new Date(sample.rekeyedAt);
    return `${date.toLocaleDateString()} · ${date.toLocaleTimeString()}`;
  }, [sample?.rekeyedAt]);

  if (!sample) {
    return (
      <div
        style={{
          padding: "18px",
          borderRadius: "18px",
          background: theme.surfaceOverlay,
          border: `1px dashed ${theme.button.outline}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "200px",
          textAlign: "center",
          color: theme.textMuted
        }}
      >
        <p style={{ margin: 0, fontSize: "0.9rem" }}>
          Select a sample to explore beat detection, technicolor stems, and retuning notes.
        </p>
      </div>
    );
  }

  const applyNudge = () => {
    const nudgedMeasures = sample.measures.map((measure) => ({
      ...measure,
      start: Math.max(0, measure.start + nudge),
      end: Math.max(0, measure.end + nudge)
    }));
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: {
        measures: nudgedMeasures,
        position: Math.max(0, sample.position + nudge)
      }
    });
    setNudge(0);
  };

  return (
    <div
      style={{
        padding: "18px",
        borderRadius: "18px",
        background: theme.surfaceOverlay,
        display: "flex",
        flexDirection: "column",
        gap: "14px",
        border: `1px solid ${theme.border}`,
        color: theme.text,
        boxShadow: theme.cardGlow
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h2 style={{ margin: 0, letterSpacing: "0.05em" }}>{sample.name}</h2>
          {sample.variantLabel && (
            <span style={{ fontSize: "0.78rem", color: theme.textMuted }}>{sample.variantLabel}</span>
          )}
        </div>
        {sample.originSampleId && (
          <span style={{ fontSize: "0.75rem", color: theme.textMuted }}>
            Fragment of {sample.originSampleId.slice(0, 6)}…
          </span>
        )}
      </div>
      <div style={{ display: "flex", gap: "12px", fontSize: "0.82rem", color: theme.textMuted }}>
        <span>BPM: {sample.bpm ?? "Detecting"}</span>
        <span>Key: {sample.key ?? "Detecting"}</span>
        <span>Measures: {sample.measures.length || "Detecting"}</span>
        {rekeyTimestamp && <span>Re-keyed {rekeyTimestamp}</span>}
      </div>
      <label style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.82rem" }}>
        First beat nudging (seconds)
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <input
            type="range"
            min={-0.5}
            max={0.5}
            step={0.01}
            value={nudge}
            onChange={(event) => setNudge(Number(event.target.value))}
          />
          <span style={{ fontSize: "0.8rem", width: "56px", color: theme.textMuted }}>{nudge.toFixed(2)}</span>
          <button
            type="button"
            onClick={applyNudge}
            style={{
              border: `1px solid ${theme.button.outline}`,
              padding: "8px 14px",
              borderRadius: "999px",
              background: theme.button.primary,
              color: theme.button.primaryText,
              fontWeight: 600,
              cursor: "pointer",
              boxShadow: theme.cardGlow
            }}
          >
            Apply
          </button>
        </div>
      </label>
      <div>
        <h3 style={{ margin: "0 0 8px", fontSize: "0.9rem", color: theme.button.primary }}>
          Measure detection
        </h3>
        <ul
          style={{
            margin: 0,
            paddingLeft: "18px",
            maxHeight: "160px",
            overflow: "auto",
            color: theme.textMuted,
            display: "flex",
            flexDirection: "column",
            gap: "6px"
          }}
        >
          {sample.measures.map((measure, index) => (
            <li key={measure.id} style={{ fontSize: "0.85rem" }}>
              Measure {index + 1}: {measure.start.toFixed(2)}s → {measure.end.toFixed(2)}s · Pitch
              {" "}
              {measure.detectedPitch ?? "?"}
              {measure.tunedPitch && measure.tunedPitch !== measure.detectedPitch
                ? ` → ${measure.tunedPitch}`
                : ""}
            </li>
          ))}
        </ul>
      </div>
      {sample.retuneMap && (
        <div>
          <h3 style={{ margin: "0 0 8px", fontSize: "0.9rem", color: theme.button.primary }}>
            Re-key map
          </h3>
          <ul
            style={{
              margin: 0,
              paddingLeft: "18px",
              display: "grid",
              gap: "4px",
              color: theme.textMuted
            }}
          >
            {sample.retuneMap.map((entry, index) => (
              <li key={`${sample.id}-retune-${index}`} style={{ fontSize: "0.82rem" }}>
                {entry}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <h3 style={{ margin: "0 0 8px", fontSize: "0.9rem", color: theme.button.primary }}>
          Stem routing
        </h3>
        <ul
          style={{
            margin: 0,
            paddingLeft: "18px",
            display: "grid",
            gap: "4px",
            color: theme.textMuted
          }}
        >
          {sample.stems.map((stem) => (
            <li key={stem.id} style={{ fontSize: "0.85rem" }}>
              {stem.name} — {stem.type}
            </li>
          ))}
        </ul>
      </div>
      <button
        type="button"
        style={{
          border: `1px solid ${theme.button.outline}`,
          padding: "10px 18px",
          borderRadius: "999px",
          background: theme.button.primary,
          color: theme.button.primaryText,
          fontWeight: 600,
          cursor: "pointer",
          boxShadow: theme.cardGlow
        }}
        onClick={() => {
          const blob = new Blob([JSON.stringify(sample, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const anchor = document.createElement("a");
          anchor.href = url;
          anchor.download = `${sample.name.replace(/\s+/g, "-")}-stem.json`;
          anchor.click();
          URL.revokeObjectURL(url);
        }}
      >
        Export stems metadata
      </button>
    </div>
  );
}
