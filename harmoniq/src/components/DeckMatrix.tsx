import { Fragment, useMemo, useRef } from "react";
import type { PointerEvent } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";
import { WaveformPreview } from "@daw/shared/WaveformPreview";

export type DeckId = "A" | "B" | "C" | "D";

export type LoopSlotStatus = "idle" | "queued" | "recording" | "playing";

export interface LoopSlot {
  id: string;
  label: string;
  status: LoopSlotStatus;
  length: "bar" | "half";
}

export type StemStatus = "standby" | "active" | "muted";

export interface DeckStem {
  id: string;
  label: string;
  status: StemStatus;
}

const STEM_BADGE_COLORS: Record<StemStatus, { background: string; border: string; text: string }> = {
  standby: {
    background: "rgba(21, 74, 98, 0.6)",
    border: "rgba(103, 255, 230, 0.4)",
    text: theme.button.primaryText,
  },
  active: {
    background: "rgba(124, 84, 255, 0.5)",
    border: "rgba(214, 189, 255, 0.6)",
    text: theme.button.primaryText,
  },
  muted: {
    background: "rgba(10, 32, 44, 0.7)",
    border: "rgba(120, 203, 220, 0.25)",
    text: theme.textMuted,
  },
};

export interface DeckPerformance {
  id: DeckId;
  loopName: string;
  waveform: Float32Array;
  filter: number;
  resonance: number;
  zoom: number;
  fxStack: string[];
  isFocused: boolean;
  level: number;
  bpm?: number;
  scale?: string;
  stems?: DeckStem[];
  source?: string;
}

export interface CrossfadeState {
  x: number;
  y: number;
}

export interface DeckMatrixProps {
  decks: DeckPerformance[];
  crossfade: CrossfadeState;
  onCrossfadeChange: (next: CrossfadeState) => void;
  onUpdateDeck: (deckId: DeckId, patch: Partial<DeckPerformance>) => void;
  onNudge: (deckId: DeckId, direction: "forward" | "back") => void;
  onFocusDeck: (deckId: DeckId) => void;
  loopSlots: Record<DeckId, LoopSlot[]>;
  onToggleLoopSlot: (deckId: DeckId, slotId: string) => void;
  masterTempo: number;
  masterPitch: number;
  onMasterTempoChange: (value: number) => void;
  onMasterPitchChange: (value: number) => void;
}

type PointerInfo = {
  startX: number;
  startY: number;
  x: number;
  y: number;
};

type GestureState = {
  deckId: DeckId;
  startFilter: number;
  startZoom: number;
  baseDistance: number;
  pointers: Map<number, PointerInfo>;
};

type DragState = {
  id: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function round(value: number) {
  return Number(value.toFixed(2));
}

function MasterControlSlider({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  const formattedBase = step < 1 ? value.toFixed(1) : Math.round(value).toString();
  const display = suffix === "st" && value > 0 ? `+${formattedBase}` : formattedBase;
  const sliderClass = "harmoniq-master-slider";
  return (
    <label
      style={{
        display: "grid",
        gap: "8px",
        padding: "12px 16px",
        borderRadius: "12px",
        border: "1px solid rgba(120, 203, 220, 0.25)",
        background: "rgba(6, 24, 34, 0.75)",
      }}
    >
      <span
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontSize: "0.68rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: theme.textMuted,
        }}
      >
        {label}
        <span style={{ color: theme.button.primaryText, fontWeight: 600 }}>
          {display} {suffix}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          const raw = Number(event.currentTarget.value);
          onChange(step < 1 ? Number(raw.toFixed(1)) : raw);
        }}
        className={sliderClass}
        style={{
          width: "100%",
          appearance: "none",
          height: "6px",
          borderRadius: "999px",
          background: "linear-gradient(90deg, rgba(132, 94, 255, 0.75), rgba(103, 255, 230, 0.75))",
          position: "relative",
        }}
      />
      <style>
        {`
          .${sliderClass}::-webkit-slider-thumb {
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: ${theme.button.primary};
            box-shadow: ${theme.shadow};
            border: none;
          }
          .${sliderClass}::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: ${theme.button.primary};
            box-shadow: ${theme.shadow};
            border: none;
          }
          .${sliderClass}::-webkit-slider-runnable-track {
            height: 6px;
            border-radius: 999px;
            background: transparent;
          }
          .${sliderClass}::-moz-range-track {
            height: 6px;
            border-radius: 999px;
            background: transparent;
          }
        `}
      </style>
    </label>
  );
}

function LoopRecorderStrip({
  slots,
  onToggle,
}: {
  slots: LoopSlot[];
  onToggle: (slotId: string) => void;
}) {
  const palette: Record<LoopSlotStatus, { background: string; border: string; indicator: string; shadow?: string }>
    = {
      idle: {
        background: "rgba(10, 32, 44, 0.85)",
        border: "rgba(120, 203, 220, 0.22)",
        indicator: "rgba(126, 154, 230, 0.4)",
      },
      queued: {
        background: "rgba(12, 46, 63, 0.95)",
        border: theme.button.primary,
        indicator: theme.button.primary,
        shadow: "0 0 16px rgba(120, 203, 220, 0.45)",
      },
      recording: {
        background: "rgba(255, 86, 134, 0.85)",
        border: "rgba(255, 173, 212, 0.8)",
        indicator: "rgba(255, 227, 236, 0.9)",
        shadow: "0 0 18px rgba(255, 126, 170, 0.55)",
      },
      playing: {
        background: "rgba(68, 207, 255, 0.82)",
        border: "rgba(198, 244, 255, 0.8)",
        indicator: "rgba(9, 43, 60, 0.85)",
        shadow: "0 0 14px rgba(68, 207, 255, 0.45)",
      },
    };

  return (
    <div
      style={{
        display: "grid",
        gap: "8px",
        alignContent: "start",
        minWidth: "108px",
      }}
    >
      <div
        style={{
          background: "linear-gradient(90deg, rgba(12, 80, 98, 0.65) 0%, rgba(9, 43, 60, 0.8) 100%)",
          borderRadius: "6px",
          padding: "6px 8px",
          fontSize: "0.62rem",
          textTransform: "uppercase",
          letterSpacing: "0.1em",
          textAlign: "center",
          color: theme.button.primaryText,
        }}
      >
        Loop Capture
      </div>
      <div style={{ display: "grid", gap: "6px" }}>
        {slots.map((slot) => {
          const swatch = palette[slot.status];
          return (
            <div
              key={slot.id}
              style={{
                display: "grid",
                gridTemplateColumns: "24px minmax(0, 1fr)",
                gap: "8px",
                alignItems: "center",
              }}
            >
              <button
                type="button"
                onClick={() => onToggle(slot.id)}
                title={
                  slot.status === "idle"
                    ? `Record ${slot.label} at the next ${slot.length === "bar" ? "bar" : "half"}-measure`
                    : `Cancel ${slot.status} clip`
                }
                style={{
                  width: "24px",
                  height: "24px",
                  borderRadius: "4px",
                  border: `1px solid ${swatch.border}`,
                  background: swatch.background,
                  display: "grid",
                  placeItems: "center",
                  boxShadow: swatch.shadow ?? "none",
                  cursor: "pointer",
                  transition: "background 0.2s ease, box-shadow 0.2s ease",
                }}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: swatch.indicator,
                  }}
                />
              </button>
              <span
                style={{
                  fontSize: "0.6rem",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color:
                    slot.status === "playing"
                      ? theme.button.primaryText
                      : slot.status === "recording"
                      ? "rgba(255, 196, 220, 0.85)"
                      : theme.textMuted,
                }}
              >
                {slot.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function XYCrossfaderPad({
  value,
  onChange,
  weights,
}: {
  value: CrossfadeState;
  onChange: (next: CrossfadeState) => void;
  weights: Record<DeckId, number>;
}) {
  const drag = useRef<DragState | null>(null);

  const updateFromEvent = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratioX = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    const ratioY = clamp((event.clientY - bounds.top) / bounds.height, 0, 1);
    onChange({ x: round(ratioX), y: round(ratioY) });
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { id: event.pointerId };
    updateFromEvent(event);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!drag.current || drag.current.id !== event.pointerId) return;
    updateFromEvent(event);
  };

  const handlePointerUp = () => {
    drag.current = null;
  };

  const indicatorX = `${value.x * 100}%`;
  const indicatorY = `${value.y * 100}%`;

  const gradient = `radial-gradient(circle at ${indicatorX} ${indicatorY}, rgba(132, 94, 255, 0.45) 0%, rgba(18, 46, 63, 0.7) 48%, rgba(8, 24, 33, 0.95) 70%)`;

  return (
    <div
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      style={{
        position: "relative",
        width: "100%",
        aspectRatio: "1",
        minHeight: "340px",
        borderRadius: "32px",
        border: `1px solid rgba(120, 203, 220, 0.5)`,
        background: gradient,
        cursor: "pointer",
        overflow: "hidden",
        boxShadow: "0 26px 72px rgba(4, 18, 28, 0.55)",
      }}
    >
      <div
        style={{
          position: "absolute",
          left: indicatorX,
          top: indicatorY,
          transform: "translate(-50%, -50%)",
          width: "118px",
          height: "118px",
          borderRadius: "50%",
          border: `1px solid ${theme.button.primary}`,
          background: "rgba(9, 43, 60, 0.85)",
          display: "grid",
          placeItems: "center",
          color: theme.button.primaryText,
          letterSpacing: "0.08em",
          fontSize: "0.82rem",
          fontWeight: 600,
          boxShadow: "0 18px 34px rgba(12, 44, 63, 0.6)",
        }}
      >
        {Math.round(value.x * 100)}/{Math.round(value.y * 100)}
      </div>
      {([
        { id: "A", x: 12, y: 12 },
        { id: "B", x: 88, y: 12 },
        { id: "C", x: 12, y: 88 },
        { id: "D", x: 88, y: 88 },
      ] as Array<{ id: DeckId; x: number; y: number }>).map(({ id, x, y }) => {
        const emphasis = weights[id];
        return (
          <span
            key={id}
            style={{
              position: "absolute",
              left: `${x}%`,
              top: `${y}%`,
              transform: "translate(-50%, -50%)",
              fontSize: "0.7rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: emphasis > 0.4 ? theme.button.primaryText : theme.text,
              background: emphasis > 0.25 ? "rgba(21, 74, 98, 0.8)" : "transparent",
              padding: emphasis > 0.25 ? "4px 8px" : undefined,
              borderRadius: "999px",
              border:
                emphasis > 0.4
                  ? `1px solid ${theme.button.primary}`
                  : emphasis > 0.25
                  ? `1px solid rgba(120, 203, 220, 0.4)`
                  : "none",
            }}
          >
            Deck {id}
          </span>
        );
      })}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 0,
          bottom: 0,
          width: "1px",
          background: "rgba(120, 203, 220, 0.2)",
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: 0,
          right: 0,
          height: "1px",
          background: "rgba(120, 203, 220, 0.2)",
        }}
      />
    </div>
  );
}

function MasterStrip({
  label,
  level,
  focus,
}: {
  label: string;
  level: number;
  focus: number;
}) {
  const barHeight = `${Math.round(level * 100)}%`;
  const focusHeight = `${Math.round(focus * 100)}%`;
  return (
    <div
      style={{
        display: "grid",
        gap: "12px",
        justifyItems: "center",
        width: "88px",
      }}
    >
      <div
        style={{
          fontSize: "0.7rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: theme.textMuted,
          textAlign: "center",
        }}
      >
        {label}
      </div>
      <div
        style={{
          position: "relative",
          width: "28px",
          height: "220px",
          borderRadius: "999px",
          border: `1px solid rgba(120, 203, 220, 0.35)`,
          background: "rgba(7, 29, 40, 0.8)",
          overflow: "hidden",
        }}
      >
        <span
          style={{
            position: "absolute",
            left: "6px",
            right: "6px",
            bottom: "8px",
            height: focusHeight,
            borderRadius: "999px",
            background: `linear-gradient(180deg, rgba(255, 97, 146, 0.8) 0%, rgba(99, 238, 205, 0.85) 80%)`,
            opacity: 0.6,
          }}
        />
        <span
          style={{
            position: "absolute",
            left: "4px",
            right: "4px",
            bottom: 0,
            height: barHeight,
            borderRadius: "999px",
            background: `linear-gradient(180deg, rgba(99, 238, 205, 0.9) 0%, rgba(132, 94, 255, 0.85) 90%)`,
          }}
        />
      </div>
      <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>
        {(level * 100).toFixed(0)}%
      </span>
    </div>
  );
}

export function DeckMatrix({
  decks,
  crossfade,
  onCrossfadeChange,
  onUpdateDeck,
  onNudge,
  onFocusDeck,
  loopSlots,
  onToggleLoopSlot,
  masterTempo,
  masterPitch,
  onMasterTempoChange,
  onMasterPitchChange,
}: DeckMatrixProps) {
  const gesture = useRef<GestureState | null>(null);
  const deckLookup = useMemo(() => new Map(decks.map((deck) => [deck.id, deck])), [decks]);

  const crossWeights = useMemo(() => {
    const left = 1 - crossfade.x;
    const right = crossfade.x;
    const top = 1 - crossfade.y;
    const bottom = crossfade.y;
    const weights: Record<DeckId, number> = {
      A: round(left * top),
      B: round(right * top),
      C: round(left * bottom),
      D: round(right * bottom),
    };
    return weights;
  }, [crossfade]);

  const masterLevels = useMemo(() => {
    const deck = (id: DeckId) => deckLookup.get(id)?.level ?? 0;
    const topBlend = crossWeights.A * deck("A") + crossWeights.B * deck("B");
    const bottomBlend = crossWeights.C * deck("C") + crossWeights.D * deck("D");
    const leftBlend = crossWeights.A * deck("A") + crossWeights.C * deck("C");
    const rightBlend = crossWeights.B * deck("B") + crossWeights.D * deck("D");
    return {
      left: clamp(leftBlend, 0, 1),
      right: clamp(rightBlend, 0, 1),
      top: clamp(topBlend, 0, 1),
      bottom: clamp(bottomBlend, 0, 1),
    };
  }, [crossWeights, deckLookup]);

  const ensureGesture = (deckId: DeckId) => {
    const deck = deckLookup.get(deckId);
    if (!deck) return null;
    if (!gesture.current || gesture.current.deckId !== deckId) {
      gesture.current = {
        deckId,
        startFilter: deck.filter,
        startZoom: deck.zoom,
        baseDistance: 0,
        pointers: new Map(),
      };
    }
    return gesture.current;
  };

  const handlePointerDown = (deckId: DeckId) => (event: PointerEvent<HTMLDivElement>) => {
    const state = ensureGesture(deckId);
    if (!state) return;
    const element = event.currentTarget;
    element.setPointerCapture(event.pointerId);
    state.pointers.set(event.pointerId, {
      startX: event.clientX,
      startY: event.clientY,
      x: event.clientX,
      y: event.clientY,
    });
    if (state.pointers.size === 1) {
      const deck = deckLookup.get(deckId);
      if (deck) {
        state.startFilter = deck.filter;
      }
    }
    if (state.pointers.size === 2) {
      const values = Array.from(state.pointers.values());
      const distance = Math.hypot(values[0].x - values[1].x, values[0].y - values[1].y);
      state.baseDistance = distance || 1;
      const deck = deckLookup.get(deckId);
      if (deck) {
        state.startZoom = deck.zoom;
      }
    }
  };

  const handlePointerMove = (deckId: DeckId) => (event: PointerEvent<HTMLDivElement>) => {
    const state = gesture.current;
    if (!state || state.deckId !== deckId) return;
    const info = state.pointers.get(event.pointerId);
    if (!info) return;
    info.x = event.clientX;
    info.y = event.clientY;
    const deck = deckLookup.get(deckId);
    if (!deck) return;

    if (state.pointers.size >= 2) {
      const pointers = Array.from(state.pointers.values());
      const distance = Math.hypot(pointers[0].x - pointers[1].x, pointers[0].y - pointers[1].y);
      if (state.baseDistance > 0) {
        const ratio = clamp(distance / state.baseDistance, 0.5, 1.8);
        const nextZoom = clamp(state.startZoom * ratio, 0.5, 3);
        onUpdateDeck(deckId, { zoom: round(nextZoom) });
      }
      return;
    }

    const deltaY = info.startY - info.y;
    const nextFilter = clamp(state.startFilter + deltaY / 260, 0, 1);
    onUpdateDeck(deckId, { filter: round(nextFilter) });
  };

  const handlePointerUp = (deckId: DeckId) => (event: PointerEvent<HTMLDivElement>) => {
    const state = gesture.current;
    if (!state || state.deckId !== deckId) return;
    state.pointers.delete(event.pointerId);
    if (state.pointers.size === 0) {
      gesture.current = null;
    }
  };

  const deckOrder: DeckId[] = ["A", "B", "C", "D"];
  const deckPositions: Record<DeckId, { gridColumn: string; gridRow: string }> = {
    A: { gridColumn: "2", gridRow: "1" },
    B: { gridColumn: "4", gridRow: "1" },
    C: { gridColumn: "2", gridRow: "2" },
    D: { gridColumn: "4", gridRow: "2" },
  };

  const loopStripPositions: Record<DeckId, { gridColumn: string; gridRow: string; justify: "start" | "end" }> = {
    A: { gridColumn: "1", gridRow: "1", justify: "start" },
    C: { gridColumn: "1", gridRow: "2", justify: "start" },
    B: { gridColumn: "5", gridRow: "1", justify: "end" },
    D: { gridColumn: "5", gridRow: "2", justify: "end" },
  };

  return (
    <section
      style={{
        ...cardSurfaceStyle,
        display: "grid",
        gap: "18px",
        padding: "22px 24px",
        position: "relative",
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.92rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            Quad Deck Matrix
          </h2>
          <p style={{ margin: 0, fontSize: "0.7rem", color: theme.textMuted }}>
            Pinch to zoom, drag vertically to shape filters, steer blends with the XY crossfader.
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", justifyContent: "flex-end" }}>
          {deckOrder.map((id) => {
            const deck = deckLookup.get(id);
            if (!deck) return null;
            return (
              <button
                key={deck.id}
                type="button"
                style={{
                  ...toolbarButtonStyle,
                  padding: "6px 12px",
                  background: deck.isFocused ? theme.button.primary : theme.button.base,
                  color: deck.isFocused ? theme.button.primaryText : theme.text,
                  borderColor: deck.isFocused ? theme.button.primary : theme.button.outline,
                }}
                onClick={() => onFocusDeck(deck.id)}
              >
                Focus {deck.id}
              </button>
            );
          })}
        </div>
      </header>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto minmax(0, 1fr) 460px minmax(0, 1fr) auto",
          gridTemplateRows: "minmax(0, 1fr) minmax(0, 1fr)",
          gap: "20px",
        }}
      >
        {deckOrder.map((id) => {
          const deck = deckLookup.get(id);
          if (!deck) return null;
          const position = deckPositions[id];
          const loopPosition = loopStripPositions[id];
          const weight = crossWeights[id];
          const levelHeight = `${Math.round(deck.level * 100)}%`;
          const loops = loopSlots[deck.id] ?? [];
          return (
            <Fragment key={deck.id}>
              <div
                style={{
                  gridColumn: loopPosition.gridColumn,
                  gridRow: loopPosition.gridRow,
                  justifySelf: loopPosition.justify,
                  alignSelf: "stretch",
                  display: "flex",
                  alignItems: "stretch",
                }}
              >
                <LoopRecorderStrip slots={loops} onToggle={(slotId) => onToggleLoopSlot(deck.id, slotId)} />
              </div>
              <article
                onPointerDown={handlePointerDown(deck.id)}
                onPointerMove={handlePointerMove(deck.id)}
                onPointerUp={handlePointerUp(deck.id)}
                onPointerCancel={handlePointerUp(deck.id)}
                style={{
                  border: `1px solid ${
                    deck.isFocused ? theme.button.primary : "rgba(120, 203, 220, 0.32)"
                  }`,
                  borderRadius: "16px",
                  padding: "18px",
                  background: `linear-gradient(180deg, rgba(12, 46, 63, ${0.85 + weight * 0.1}) 0%, rgba(9, 33, 45, ${
                    0.92 + weight * 0.05
                  }) 100%)`,
                  display: "grid",
                  gap: "14px",
                  position: "relative",
                  cursor: "pointer",
                  gridColumn: position.gridColumn,
                  gridRow: position.gridRow,
                }}
              >
                <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ display: "grid", gap: "4px" }}>
                    <strong style={{ fontSize: "0.9rem" }}>Deck {deck.id}</strong>
                    <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>{deck.loopName}</span>
                    {deck.source ? (
                      <span style={{ fontSize: "0.64rem", color: theme.button.primaryText }}>
                        {deck.source}
                      </span>
                    ) : null}
                  </div>
                  <div style={{ textAlign: "right", display: "grid", gap: "4px" }}>
                    {deck.bpm ? (
                      <span style={{ fontSize: "0.67rem", color: theme.button.primaryText }}>
                        {deck.bpm} BPM
                      </span>
                    ) : null}
                    {deck.scale ? (
                      <span style={{ fontSize: "0.67rem", color: theme.textMuted }}>Scale {deck.scale}</span>
                    ) : null}
                    <span style={{ fontSize: "0.67rem", color: theme.textMuted }}>Zoom ×{deck.zoom.toFixed(2)}</span>
                    <span style={{ fontSize: "0.67rem", color: theme.textMuted }}>
                      Filter {Math.round(deck.filter * 100)}%
                    </span>
                    <span style={{ fontSize: "0.67rem", color: theme.button.primary }}>
                      Blend {Math.round(weight * 100)}%
                    </span>
                  </div>
                </header>
                {deck.stems && deck.stems.length ? (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
                    {deck.stems.map((stem) => {
                      const colors = STEM_BADGE_COLORS[stem.status];
                      return (
                        <span
                          key={stem.id}
                          style={{
                            padding: "6px 10px",
                            borderRadius: "999px",
                            border: colors.border,
                            background: colors.background,
                            color: colors.text,
                            fontSize: "0.64rem",
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                          }}
                        >
                          {stem.label} · {stem.status === "standby" ? "Standby" : stem.status === "active" ? "On" : "Muted"}
                        </span>
                      );
                    })}
                  </div>
                ) : null}
                <div
                  style={{
                    position: "relative",
                    height: "140px",
                    borderRadius: "14px",
                    background: `linear-gradient(180deg, rgba(124, 84, 255, 0.35) 0%, rgba(76, 199, 194, 0.25) 100%)`,
                    border: `1px solid rgba(126, 154, 230, 0.28)`,
                    overflow: "hidden",
                  }}
                >
                  <WaveformPreview
                    waveform={deck.waveform}
                    fillColor="rgba(132, 94, 255, 0.55)"
                    strokeColor="rgba(255, 189, 255, 0.9)"
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: "14px",
                      right: "14px",
                      width: "12px",
                      height: "92px",
                      borderRadius: "999px",
                      background: "rgba(9, 43, 60, 0.8)",
                      border: `1px solid rgba(120, 203, 220, 0.4)`,
                      display: "flex",
                      alignItems: "flex-end",
                      padding: "2px",
                    }}
                  >
                    <span
                      style={{
                        width: "100%",
                        height: levelHeight,
                        background: `linear-gradient(180deg, rgba(103, 255, 230, 0.9) 0%, rgba(255, 114, 177, 0.8) 100%)`,
                        borderRadius: "999px",
                        transition: "height 0.2s ease",
                      }}
                    />
                  </div>
                  <div
                    style={{
                      position: "absolute",
                      left: "14px",
                      top: "14px",
                      bottom: "14px",
                      width: "10px",
                      borderRadius: "999px",
                      background: `linear-gradient(180deg, rgba(68, 207, 255, 0.8) 0%, rgba(241, 139, 180, 0.7) 100%)`,
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        left: "-4px",
                        width: "18px",
                        height: "18px",
                        borderRadius: "999px",
                        background: theme.button.primary,
                        top: `${(1 - deck.filter) * 100}%`,
                        transform: "translateY(-50%)",
                        boxShadow: theme.shadow,
                      }}
                    />
                  </div>
                </div>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "10px", alignItems: "center" }}>
                <div style={{ display: "flex", gap: "8px" }}>
                  <button
                    type="button"
                    style={{ ...toolbarButtonStyle, padding: "6px 12px" }}
                    onClick={() => onNudge(deck.id, "back")}
                  >
                    Nudge -
                  </button>
                  <button
                    type="button"
                    style={{ ...toolbarButtonStyle, padding: "6px 12px" }}
                    onClick={() => onNudge(deck.id, "forward")}
                  >
                    Nudge +
                  </button>
                </div>
                <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", justifyContent: "flex-end" }}>
                  {deck.fxStack.map((fx) => (
                    <span
                      key={fx}
                      style={{
                        padding: "4px 10px",
                        borderRadius: "999px",
                        background: "rgba(30, 72, 102, 0.7)",
                        border: `1px solid rgba(120, 203, 220, 0.4)`,
                        fontSize: "0.65rem",
                        letterSpacing: "0.06em",
                        textTransform: "uppercase",
                      }}
                    >
                      {fx}
                    </span>
                  ))}
                </div>
              </div>
              </article>
            </Fragment>
          );
        })}
        <div
          style={{
            gridColumn: "3",
            gridRow: "1 / span 2",
            border: `1px solid rgba(120, 203, 220, 0.32)`,
            borderRadius: "18px",
            background: "rgba(8, 29, 40, 0.85)",
            display: "grid",
            gridTemplateRows: "auto auto 1fr",
            padding: "18px 16px",
            gap: "18px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "0.75rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: theme.textMuted,
            }}
          >
            <span>Master</span>
            <span style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <span>{Math.round(masterTempo)} BPM</span>
              <span>
                Pitch {masterPitch > 0 ? `+${masterPitch.toFixed(1)}` : masterPitch.toFixed(1)} st
              </span>
            </span>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: "12px",
            }}
          >
            <MasterControlSlider
              label="Tempo"
              value={masterTempo}
              min={80}
              max={160}
              step={1}
              suffix="BPM"
              onChange={onMasterTempoChange}
            />
            <MasterControlSlider
              label="Pitch"
              value={masterPitch}
              min={-12}
              max={12}
              step={0.1}
              suffix="st"
              onChange={onMasterPitchChange}
            />
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "auto 1fr auto",
              alignItems: "center",
              gap: "16px",
            }}
          >
            <MasterStrip label="Left Sum" level={masterLevels.left} focus={masterLevels.top} />
            <XYCrossfaderPad value={crossfade} onChange={onCrossfadeChange} weights={crossWeights} />
            <MasterStrip label="Right Sum" level={masterLevels.right} focus={masterLevels.bottom} />
          </div>
        </div>
      </div>
    </section>
  );
}
