import { useMemo } from "react";
import type { Preferences, PreferencesUpdate } from "../state/ProjectStore";
import { theme } from "../theme";
import {
  STEM_ENGINES,
  describeHeuristics,
  getStemEngineDefinition,
  type StemEngineDefinition,
} from "../stem_engines";

interface SettingsDialogProps {
  isOpen: boolean;
  preferences: Preferences;
  onClose: () => void;
  onUpdate: (update: PreferencesUpdate) => void;
}

function HeuristicToggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: "10px",
        padding: "10px 12px",
        borderRadius: "10px",
        border: `1px solid ${checked ? theme.button.primary : theme.button.outline}`,
        background: checked ? "rgba(76, 199, 194, 0.1)" : theme.surface,
        cursor: "pointer",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        style={{ width: "16px", height: "16px", marginTop: "2px" }}
      />
      <span style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
        <strong style={{ fontSize: "0.75rem" }}>{label}</strong>
        <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>{description}</span>
      </span>
    </label>
  );
}

export function SettingsDialog({ isOpen, preferences, onClose, onUpdate }: SettingsDialogProps) {
  const engineDefinition = useMemo<StemEngineDefinition>(
    () => getStemEngineDefinition(preferences.stemEngine),
    [preferences.stemEngine],
  );

  const heuristicSummary = useMemo(
    () => describeHeuristics(engineDefinition, preferences.heuristics),
    [engineDefinition, preferences.heuristics],
  );

  if (!isOpen) {
    return null;
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(7, 10, 18, 0.78)",
        backdropFilter: "blur(6px)",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        padding: "72px 24px 24px",
        zIndex: 30,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "min(720px, 96vw)",
          background: theme.surfaceRaised,
          borderRadius: "18px",
          border: `1px solid ${theme.border}`,
          boxShadow: theme.shadow,
          display: "flex",
          flexDirection: "column",
          gap: "18px",
          padding: "22px 24px 28px",
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "1rem", letterSpacing: "0.04em" }}>Settings</h2>
            <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>
              Configure stem engines and heuristic focus for new separations.
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              border: `1px solid ${theme.button.outline}`,
              background: theme.button.base,
              color: theme.text,
              borderRadius: "999px",
              padding: "4px 10px",
              cursor: "pointer",
              fontSize: "0.75rem",
            }}
          >
            Close
          </button>
        </div>

        <section style={{ display: "grid", gap: "12px" }}>
          <strong style={{ fontSize: "0.78rem", letterSpacing: "0.05em" }}>Stem generation engine</strong>
          <div style={{ display: "grid", gap: "10px" }}>
            {STEM_ENGINES.map((engine) => {
              const isSelected = engine.id === preferences.stemEngine;
              return (
                <label
                  key={engine.id}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "12px",
                    padding: "12px 14px",
                    borderRadius: "12px",
                    border: `1px solid ${
                      isSelected ? theme.button.primary : theme.button.outline
                    }`,
                    background: isSelected ? "rgba(76, 199, 194, 0.14)" : theme.surface,
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="radio"
                    name="stem-engine"
                    checked={isSelected}
                    onChange={() => onUpdate({ stemEngine: engine.id })}
                    style={{ width: "16px", height: "16px", marginTop: "2px" }}
                  />
                  <span style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                    <strong style={{ fontSize: "0.78rem" }}>{engine.name}</strong>
                    <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{engine.description}</span>
                  </span>
                </label>
              );
            })}
          </div>
        </section>

        <section style={{ display: "grid", gap: "12px" }}>
          <strong style={{ fontSize: "0.78rem", letterSpacing: "0.05em" }}>
            Heuristic focus
          </strong>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: "12px",
            }}
          >
            <HeuristicToggle
              label="Percussion attenuation"
              description={engineDefinition.heuristics.percussion.attenuation}
              checked={preferences.heuristics.percussion.attenuation}
              onChange={(next) =>
                onUpdate({ heuristics: { percussion: { attenuation: next } } })
              }
            />
            <HeuristicToggle
              label="Percussion tonal cut"
              description={engineDefinition.heuristics.percussion.tonalCut}
              checked={preferences.heuristics.percussion.tonalCut}
              onChange={(next) =>
                onUpdate({ heuristics: { percussion: { tonalCut: next } } })
              }
            />
            <HeuristicToggle
              label="Vocal bleed reduction"
              description={engineDefinition.heuristics.vocals.attenuationCut}
              checked={preferences.heuristics.vocals.attenuationCut}
              onChange={(next) =>
                onUpdate({ heuristics: { vocals: { attenuationCut: next } } })
              }
            />
            <HeuristicToggle
              label="Vocal profile matching"
              description={engineDefinition.heuristics.vocals.profileMatch}
              checked={preferences.heuristics.vocals.profileMatch}
              onChange={(next) =>
                onUpdate({ heuristics: { vocals: { profileMatch: next } } })
              }
            />
            <HeuristicToggle
              label="Frequency-shift alignment"
              description={engineDefinition.heuristics.vocals.frequencyShift}
              checked={preferences.heuristics.vocals.frequencyShift}
              onChange={(next) =>
                onUpdate({ heuristics: { vocals: { frequencyShift: next } } })
              }
            />
          </div>
          <div
            style={{
              padding: "12px 14px",
              borderRadius: "12px",
              border: `1px solid ${theme.button.outline}`,
              background: theme.surface,
              fontSize: "0.68rem",
              color: theme.textMuted,
              display: "grid",
              gap: "4px",
            }}
          >
            <span>
              Active percussion focus: {heuristicSummary.percussion.join(" • ") || "Disabled"}
            </span>
            <span>Active vocal focus: {heuristicSummary.vocals.join(" • ") || "Disabled"}</span>
          </div>
        </section>
      </div>
    </div>
  );
}
