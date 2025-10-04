import { useEffect, useMemo, useState } from "react";
import type { SampleClip, TrackEffects } from "../types";
import { theme } from "../theme";
import { cloneTrackEffects } from "../lib/effectPresets";

interface TrackEffectsPanelProps {
  sample: SampleClip;
  onClose: () => void;
  onUpdate: (effects: TrackEffects) => void;
}

type EffectKey = keyof TrackEffects;

type NumericKey<T> = {
  [K in keyof T]: T[K] extends number ? K : never;
}[keyof T];

function SliderControl<T extends Record<string, unknown>>({
  label,
  min,
  max,
  step,
  value,
  onChange
}: {
  label: string;
  min: number;
  max: number;
  step?: number;
  value: number;
  onChange: (next: number) => void;
}) {
  return (
    <label
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "4px",
        fontSize: "0.75rem",
        color: theme.text
      }}
    >
      <span style={{ display: "flex", justifyContent: "space-between" }}>
        {label}
        <span style={{ color: theme.textMuted }}>{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step ?? 0.01}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ width: "100%" }}
      />
    </label>
  );
}

function ToggleRow({
  label,
  enabled,
  onToggle,
  description
}: {
  label: string;
  enabled: boolean;
  onToggle: (value: boolean) => void;
  description?: string;
}) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
      <div>
        <strong style={{ fontSize: "0.85rem" }}>{label}</strong>
        {description && (
          <p style={{ margin: "4px 0 0", fontSize: "0.72rem", color: theme.textMuted }}>{description}</p>
        )}
      </div>
      <label
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          fontSize: "0.75rem",
          cursor: "pointer"
        }}
      >
        <input
          type="checkbox"
          checked={enabled}
          onChange={(event) => onToggle(event.target.checked)}
        />
        {enabled ? "On" : "Off"}
      </label>
    </div>
  );
}

export function TrackEffectsPanel({ sample, onClose, onUpdate }: TrackEffectsPanelProps) {
  const [effects, setEffects] = useState<TrackEffects>(() => cloneTrackEffects(sample.effects));

  useEffect(() => {
    setEffects(cloneTrackEffects(sample.effects));
  }, [sample]);

  const updateEffect = <K extends EffectKey, P extends Partial<TrackEffects[K]>>(key: K, patch: P) => {
    setEffects((previous) => {
      const next = { ...previous, [key]: { ...previous[key], ...patch } } as TrackEffects;
      onUpdate(next);
      return next;
    });
  };

  const renderFilterControls = (key: "hiPass" | "lowPass") => {
    const settings = effects[key];
    return (
      <div
        key={key}
        style={{
          display: "grid",
          gap: "12px",
          padding: "14px",
          borderRadius: "12px",
          background: theme.surface,
          border: `1px solid ${theme.border}`
        }}
      >
        <ToggleRow
          label={key === "hiPass" ? "Hi-pass filter" : "Low-pass filter"}
          enabled={settings.enabled}
          onToggle={(value) => updateEffect(key, { enabled: value })}
          description={
            key === "hiPass"
              ? "Carve out muddy lows with a resonant high-pass"
              : "Darken the sample with a smooth low-pass"
          }
        />
        <SliderControl
          label="Mix"
          min={0}
          max={1}
          value={settings.mix}
          onChange={(value) => updateEffect(key, { mix: value })}
        />
        <SliderControl
          label="Cutoff"
          min={key === "hiPass" ? 20 : 200}
          max={key === "hiPass" ? 1000 : 20000}
          step={key === "hiPass" ? 1 : 10}
          value={settings.cutoff}
          onChange={(value) => updateEffect(key, { cutoff: value })}
        />
        <SliderControl
          label="Resonance"
          min={0}
          max={1}
          value={settings.resonance}
          onChange={(value) => updateEffect(key, { resonance: value })}
        />
      </div>
    );
  };

  const gateRates: Array<TrackEffects["rhythmicGate"]["rate"]> = ["1/4", "1/8", "1/16", "1/32"];
  const slicerDivisions: Array<TrackEffects["slicer"]["division"]> = ["1/2", "1/4", "1/8", "1/16"];
  const delayTimes: Array<TrackEffects["delay"]["time"]> = ["1/8", "1/4", "1/2", "1"];

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(5, 15, 22, 0.65)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 50,
        padding: "40px"
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(900px, 90vw)",
          maxHeight: "90vh",
          overflowY: "auto",
          background: theme.surfaceOverlay,
          borderRadius: "20px",
          border: `1px solid ${theme.button.outline}`,
          boxShadow: theme.cardGlow,
          padding: "28px",
          color: theme.text,
          display: "grid",
          gap: "20px"
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "1.1rem", letterSpacing: "0.08em" }}>
              {sample.name}
            </h2>
            <p style={{ margin: 0, fontSize: "0.78rem", color: theme.textMuted }}>
              Sculpt this lane with per-track processors and a dedicated VST link.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              borderRadius: "999px",
              padding: "8px 16px",
              border: `1px solid ${theme.button.outline}`,
              background: theme.button.base,
              color: theme.text,
              cursor: "pointer"
            }}
          >
            Close
          </button>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: "18px"
          }}
        >
          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Glitch shuffler"
              enabled={effects.glitch.enabled}
              onToggle={(value) => updateEffect("glitch", { enabled: value })}
              description="Buffer jumps and granular stutters"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.glitch.mix}
              onChange={(value) => updateEffect("glitch", { mix: value })}
            />
            <SliderControl
              label="Density"
              min={0}
              max={1}
              value={effects.glitch.density}
              onChange={(value) => updateEffect("glitch", { density: value })}
            />
            <SliderControl
              label="Scatter"
              min={0}
              max={1}
              value={effects.glitch.scatter}
              onChange={(value) => updateEffect("glitch", { scatter: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Rhythmic gate"
              enabled={effects.rhythmicGate.enabled}
              onToggle={(value) => updateEffect("rhythmicGate", { enabled: value })}
              description="Chop pads or tails into tempo-synced pulses"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.rhythmicGate.mix}
              onChange={(value) => updateEffect("rhythmicGate", { mix: value })}
            />
            <label style={{ display: "grid", gap: "4px", fontSize: "0.75rem" }}>
              Rate
              <select
                value={effects.rhythmicGate.rate}
                onChange={(event) => updateEffect("rhythmicGate", { rate: event.target.value as TrackEffects["rhythmicGate"]["rate"] })}
                style={{
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  padding: "6px 10px",
                  background: theme.surface,
                  color: theme.text
                }}
              >
                {gateRates.map((rate) => (
                  <option key={rate} value={rate}>
                    {rate} note
                  </option>
                ))}
              </select>
            </label>
            <SliderControl
              label="Swing"
              min={0}
              max={0.5}
              value={effects.rhythmicGate.swing}
              onChange={(value) => updateEffect("rhythmicGate", { swing: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Slicer"
              enabled={effects.slicer.enabled}
              onToggle={(value) => updateEffect("slicer", { enabled: value })}
              description="Tempo synced slicing with micro-jitter"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.slicer.mix}
              onChange={(value) => updateEffect("slicer", { mix: value })}
            />
            <label style={{ display: "grid", gap: "4px", fontSize: "0.75rem" }}>
              Division
              <select
                value={effects.slicer.division}
                onChange={(event) => updateEffect("slicer", { division: event.target.value as TrackEffects["slicer"]["division"] })}
                style={{
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  padding: "6px 10px",
                  background: theme.surface,
                  color: theme.text
                }}
              >
                {slicerDivisions.map((division) => (
                  <option key={division} value={division}>
                    {division}
                  </option>
                ))}
              </select>
            </label>
            <SliderControl
              label="Jitter"
              min={0}
              max={0.5}
              value={effects.slicer.jitter}
              onChange={(value) => updateEffect("slicer", { jitter: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Bitcrush"
              enabled={effects.bitcrush.enabled}
              onToggle={(value) => updateEffect("bitcrush", { enabled: value })}
              description="Lo-fi grit with resolution reduction"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.bitcrush.mix}
              onChange={(value) => updateEffect("bitcrush", { mix: value })}
            />
            <SliderControl
              label="Depth (bits)"
              min={4}
              max={16}
              step={1}
              value={effects.bitcrush.depth}
              onChange={(value) => updateEffect("bitcrush", { depth: value })}
            />
            <SliderControl
              label="Downsample"
              min={1}
              max={12}
              step={1}
              value={effects.bitcrush.downsample}
              onChange={(value) => updateEffect("bitcrush", { downsample: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Reverb"
              enabled={effects.reverb.enabled}
              onToggle={(value) => updateEffect("reverb", { enabled: value })}
              description="Shimmering tails for space and depth"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.reverb.mix}
              onChange={(value) => updateEffect("reverb", { mix: value })}
            />
            <SliderControl
              label="Size"
              min={0}
              max={1}
              value={effects.reverb.size}
              onChange={(value) => updateEffect("reverb", { size: value })}
            />
            <SliderControl
              label="Decay"
              min={0}
              max={1.5}
              value={effects.reverb.decay}
              onChange={(value) => updateEffect("reverb", { decay: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Delay"
              enabled={effects.delay.enabled}
              onToggle={(value) => updateEffect("delay", { enabled: value })}
              description="Ping-pong echoes quantised to tempo"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.delay.mix}
              onChange={(value) => updateEffect("delay", { mix: value })}
            />
            <label style={{ display: "grid", gap: "4px", fontSize: "0.75rem" }}>
              Time
              <select
                value={effects.delay.time}
                onChange={(event) => updateEffect("delay", { time: event.target.value as TrackEffects["delay"]["time"] })}
                style={{
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  padding: "6px 10px",
                  background: theme.surface,
                  color: theme.text
                }}
              >
                {delayTimes.map((time) => (
                  <option key={time} value={time}>
                    {time} note
                  </option>
                ))}
              </select>
            </label>
            <SliderControl
              label="Feedback"
              min={0}
              max={0.95}
              value={effects.delay.feedback}
              onChange={(value) => updateEffect("delay", { feedback: value })}
            />
          </div>

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Chorus"
              enabled={effects.chorus.enabled}
              onToggle={(value) => updateEffect("chorus", { enabled: value })}
              description="Wide detuned motion"
            />
            <SliderControl
              label="Mix"
              min={0}
              max={1}
              value={effects.chorus.mix}
              onChange={(value) => updateEffect("chorus", { mix: value })}
            />
            <SliderControl
              label="Rate"
              min={0.1}
              max={5}
              value={effects.chorus.rate}
              onChange={(value) => updateEffect("chorus", { rate: value })}
            />
            <SliderControl
              label="Depth"
              min={0}
              max={1}
              value={effects.chorus.depth}
              onChange={(value) => updateEffect("chorus", { depth: value })}
            />
          </div>

          {renderFilterControls("hiPass")}
          {renderFilterControls("lowPass")}

          <div
            style={{
              padding: "14px",
              borderRadius: "12px",
              background: theme.surface,
              border: `1px solid ${theme.border}`,
              display: "grid",
              gap: "12px"
            }}
          >
            <ToggleRow
              label="Link VST"
              enabled={effects.vst.enabled}
              onToggle={(value) => updateEffect("vst", { enabled: value })}
              description="Route this lane through a VST from the rack"
            />
            <label style={{ display: "grid", gap: "4px", fontSize: "0.75rem" }}>
              Plugin alias
              <input
                type="text"
                value={effects.vst.pluginName}
                placeholder="eg. Chromatic Delay"
                onChange={(event) => updateEffect("vst", { pluginName: event.target.value })}
                style={{
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  padding: "6px 10px",
                  background: theme.surface,
                  color: theme.text
                }}
              />
            </label>
            <label style={{ display: "grid", gap: "4px", fontSize: "0.75rem" }}>
              Notes
              <textarea
                value={effects.vst.notes ?? ""}
                onChange={(event) => updateEffect("vst", { notes: event.target.value })}
                placeholder="Describe the chain or modulation routing"
                rows={3}
                style={{
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  padding: "8px",
                  background: theme.surface,
                  color: theme.text,
                  resize: "vertical"
                }}
              />
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
