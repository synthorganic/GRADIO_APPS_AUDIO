import { useState, type CSSProperties } from "react";
import { audioEngine } from "../lib/audioEngine";
import type { Project } from "../types";
import { theme } from "../theme";

interface TransportControlsProps {
  project: Project;
}

export function TransportControls({ project }: TransportControlsProps) {
  const [volume, setVolume] = useState(0.9);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRendering, setIsRendering] = useState(false);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 16px",
        borderRadius: "16px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: theme.cardGlow,
        color: theme.text
      }}
    >
      <div style={{ display: "flex", gap: "12px" }}>
        <button
          type="button"
          onClick={() => {
            audioEngine.stop();
            setIsPlaying(false);
          }}
          style={buttonStyle}
        >
          Stop
        </button>
        <button
          type="button"
          onClick={() => {
            if (isPlaying) {
              audioEngine.stop();
              setIsPlaying(false);
            } else {
              const firstSample = project.samples[0];
              if (firstSample) {
                void audioEngine.play(firstSample, firstSample.measures);
                setIsPlaying(true);
              }
            }
          }}
          style={{
            ...buttonStyle,
            background: theme.button.primary,
            color: theme.button.primaryText
          }}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          disabled={isRendering}
          onClick={() => {
            setIsRendering(true);
            setTimeout(() => {
              const audioData = new Blob(["mixdown"], { type: "audio/wav" });
              const url = URL.createObjectURL(audioData);
              const anchor = document.createElement("a");
              anchor.href = url;
              anchor.download = `${project.name.replace(/\s+/g, "-")}-mixdown.wav`;
              anchor.click();
              URL.revokeObjectURL(url);
              setIsRendering(false);
            }, 1200);
          }}
          style={{
            ...buttonStyle,
            background: theme.surface,
            color: theme.text,
            border: `1px solid ${theme.button.outline}`,
            cursor: isRendering ? "wait" : "pointer",
            opacity: isRendering ? 0.6 : 1
          }}
        >
          {isRendering ? "Rendering…" : "Render mixdown"}
        </button>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "0.85rem", color: theme.textMuted }}>Volume</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(event) => {
            const value = Number(event.target.value);
            setVolume(value);
            audioEngine.setVolume(value);
          }}
        />
      </div>
    </div>
  );
}

const buttonStyle: CSSProperties = {
  border: `1px solid ${theme.button.outline}`,
  background: theme.button.base,
  color: theme.text,
  padding: "10px 18px",
  borderRadius: "999px",
  fontWeight: 600,
  cursor: "pointer",
  transition: "background 0.2s ease, transform 0.2s ease",
  boxShadow: theme.cardGlow
};
