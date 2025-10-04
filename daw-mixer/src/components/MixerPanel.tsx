import { useMemo } from "react";
import type { Project } from "../types";
import { useProjectStore } from "../state/ProjectStore";
import { audioEngine } from "../lib/audioEngine";
import { theme } from "../theme";
import { MasteringPanel } from "./MasteringPanel";

interface MixerPanelProps {
  project: Project;
}

export function MixerPanel({ project }: MixerPanelProps) {
  const { currentProjectId, dispatch } = useProjectStore();
  const channels = useMemo(() => project.channels, [project.channels]);

  return (
    <div
      style={{
        display: "grid",
        gap: "12px",
        color: theme.text,
        minWidth: "min(960px, 100%)"
      }}
    >
      <section style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2
            style={{
              margin: 0,
              fontSize: "0.95rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em"
            }}
          >
            Channel Mixer
          </h2>
          <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
            Adjust gain, pan and FX routing per lane
          </span>
        </header>
        <div
          style={{
            display: "grid",
            gap: "8px",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))"
          }}
        >
          {channels.map((channel) => (
            <div
              key={channel.id}
              style={{
                border: `1px solid ${theme.border}`,
                borderRadius: "10px",
                padding: "10px 12px",
                background: theme.surfaceOverlay,
                display: "grid",
                gap: "6px"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong style={{ fontSize: "0.8rem" }}>{channel.name}</strong>
                <span style={{ fontSize: "0.65rem", color: theme.textMuted }}>{channel.type.toUpperCase()}</span>
              </div>
              <label style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "0.7rem" }}>
                Volume
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={channel.volume}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    audioEngine.setChannelMix(channel.id, { volume: value, pan: channel.pan });
                    dispatch({
                      type: "update-channel",
                      projectId: currentProjectId,
                      channelId: channel.id,
                      patch: { volume: value }
                    });
                  }}
                />
              </label>
              <label style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "0.7rem" }}>
                Pan
                <input
                  type="range"
                  min={-1}
                  max={1}
                  step={0.01}
                  value={channel.pan}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    audioEngine.setChannelMix(channel.id, { volume: channel.volume, pan: value });
                    dispatch({
                      type: "update-channel",
                      projectId: currentProjectId,
                      channelId: channel.id,
                      patch: { pan: value }
                    });
                  }}
                />
              </label>
              <button
                type="button"
                onClick={() =>
                  dispatch({
                    type: "update-channel",
                    projectId: currentProjectId,
                    channelId: channel.id,
                    patch: { isFxEnabled: !channel.isFxEnabled }
                  })
                }
                style={{
                  borderRadius: "8px",
                  padding: "6px 10px",
                  border: `1px solid ${theme.button.outline}`,
                  background: channel.isFxEnabled ? theme.button.primary : theme.surface,
                  color: channel.isFxEnabled ? theme.button.primaryText : theme.text,
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  cursor: "pointer"
                }}
              >
                {channel.isFxEnabled ? "FX Enabled" : "FX Bypassed"}
              </button>
            </div>
          ))}
        </div>
      </section>
      <MasteringPanel project={project} variant="inline" />
    </div>
  );
}
