import { useEffect, useState, type CSSProperties } from "react";
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
  const [masterDb, setMasterDb] = useState(-120);

  useEffect(() => {
    const handlePlay = () => setIsPlaying(true);
    const handleStop = () => setIsPlaying(false);
    window.addEventListener("audio-play", handlePlay);
    window.addEventListener("audio-stop", handleStop);
    return () => {
      window.removeEventListener("audio-play", handlePlay);
      window.removeEventListener("audio-stop", handleStop);
    };
  }, []);

  useEffect(() => {
    let raf: number | null = null;
    const tick = () => {
      const { rms } = audioEngine.getMasterLevels();
      const db = rms > 0 ? 20 * Math.log10(rms) : -120;
      setMasterDb(db);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { if (raf) cancelAnimationFrame(raf); };
  }, []);

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
            } else {
              const firstSample = [...project.samples].sort(
                (a, b) => a.position - b.position
              )[0];
              if (firstSample) {
                void audioEngine.play(firstSample, firstSample.measures, {
                  timelineOffset: firstSample.position
                });
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
          onClick={async () => {
            try {
              setIsRendering(true);
              const clips = project.samples.filter((s) => s.isInTimeline !== false);
              const blob = await audioEngine.renderTimelineMix(
                clips,
                project.mastering,
                project.channels,
                { sampleRate: 48000 },
              );
              const url = URL.createObjectURL(blob);
              const anchor = document.createElement("a");
              anchor.href = url;
              anchor.download = `${project.name.replace(/\s+/g, "-")}-mixdown.wav`;
              anchor.click();
              URL.revokeObjectURL(url);
            } finally {
              setIsRendering(false);
            }
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
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ fontSize: "0.85rem", color: theme.textMuted }}>Master</span>
          <div style={{ width: "120px", height: "6px", borderRadius: "999px", overflow: "hidden", background: theme.surface, border: `1px solid ${theme.button.outline}` }}>
            {(() => {
              const pct = Math.max(0, Math.min(100, ((masterDb + 60) / 60) * 100));
              return <div style={{ height: "100%", width: `${pct}%`, background: theme.button.primary, transition: "width 0.1s linear" }} />;
            })()}
          </div>
          <span style={{ fontSize: "0.8rem", color: theme.textMuted }}>{masterDb.toFixed(1)} dB</span>
        </div>
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
