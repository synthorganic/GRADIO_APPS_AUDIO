import { useMemo, useRef } from "react";
import { theme } from "@daw-theme";

export interface HarmonicWheelProps {
  selectedKey: string;
  onSelect: (key: string) => void;
  rotation: number; // degrees
  onRotationChange: (rotation: number) => void;
  emphasis: number;
  onEmphasisChange: (value: number) => void;
}

type WheelGesture =
  | {
      type: "rotate";
      pointerId: number;
      baseAngle: number;
      baseRotation: number;
    }
  | {
      type: "pinch";
      pointerIds: [number, number];
      baseDistance: number;
      baseEmphasis: number;
    };

const OUTER_KEYS = Array.from({ length: 12 }, (_, index) => `${((index + 1 - 1 + 12) % 12) + 1}B`);
const INNER_KEYS = Array.from({ length: 12 }, (_, index) => `${((index + 1 - 1 + 12) % 12) + 1}A`);

export function HarmonicWheel({
  selectedKey,
  onSelect,
  rotation,
  onRotationChange,
  emphasis,
  onEmphasisChange,
}: HarmonicWheelProps) {
  const wheelRef = useRef<HTMLDivElement | null>(null);
  const gesture = useRef<WheelGesture | null>(null);
  const pointerPositions = useRef(new Map<number, { x: number; y: number }>());

  const spokes = useMemo(() => {
    const colors = theme.accentBeam;
    return OUTER_KEYS.map((outerKey, index) => {
      const color = colors[index % colors.length];
      return {
        outerKey,
        innerKey: INNER_KEYS[index],
        color,
        angle: (index / OUTER_KEYS.length) * 360,
      };
    });
  }, []);

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    pointerPositions.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointerPositions.current.size === 1) {
      const angle = pointerAngle(event.clientX, event.clientY);
      if (angle == null) return;
      gesture.current = {
        type: "rotate",
        pointerId: event.pointerId,
        baseAngle: angle,
        baseRotation: rotation,
      };
    } else if (pointerPositions.current.size === 2) {
      const [first, second] = Array.from(pointerPositions.current.values());
      const distance = Math.max(distanceBetween(first, second), 1);
      const ids = Array.from(pointerPositions.current.keys()) as [number, number];
      gesture.current = {
        type: "pinch",
        pointerIds: ids,
        baseDistance: distance,
        baseEmphasis: emphasis,
      };
    }
  }

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const stored = pointerPositions.current.get(event.pointerId);
    if (!stored) return;
    stored.x = event.clientX;
    stored.y = event.clientY;
    pointerPositions.current.set(event.pointerId, stored);
    const state = gesture.current;
    if (!state) return;
    if (state.type === "rotate" && state.pointerId === event.pointerId) {
      const angle = pointerAngle(event.clientX, event.clientY);
      if (angle == null) return;
      const delta = angle - state.baseAngle;
      onRotationChange(normalizeDegrees(state.baseRotation + (delta * 180) / Math.PI));
    }
    if (state.type === "pinch" && state.pointerIds.includes(event.pointerId)) {
      const [firstId, secondId] = state.pointerIds;
      const first = pointerPositions.current.get(firstId);
      const second = pointerPositions.current.get(secondId);
      if (!first || !second) return;
      const distance = Math.max(distanceBetween(first, second), 1);
      const ratio = clamp(distance / state.baseDistance, 0.6, 1.6);
      onEmphasisChange(clamp(state.baseEmphasis * ratio, 0.5, 1.6));
    }
  }

  function handlePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    pointerPositions.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const state = gesture.current;
    if (!state) return;
    if (state.type === "rotate" && state.pointerId === event.pointerId) {
      gesture.current = null;
    }
    if (state.type === "pinch" && state.pointerIds.includes(event.pointerId)) {
      gesture.current = null;
      const remaining = Array.from(pointerPositions.current.entries());
      if (remaining.length === 1) {
        const [id, pos] = remaining[0];
        const angle = pointerAngle(pos.x, pos.y);
        if (angle == null) return;
        gesture.current = {
          type: "rotate",
          pointerId: id,
          baseAngle: angle,
          baseRotation: rotation,
        };
      }
    }
  }

  return (
    <section
      style={{
        padding: "22px",
        borderRadius: "18px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: theme.cardGlow,
        color: theme.text,
        display: "flex",
        flexDirection: "column",
        gap: "16px",
      }}
    >
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.95rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Harmonic Wheel
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
            Rotate to shift compatibilities · Pinch to widen blends
          </p>
        </div>
        <div
          style={{
            padding: "6px 14px",
            borderRadius: "999px",
            border: `1px solid ${theme.button.outline}`,
            background: theme.surface,
            fontSize: "0.72rem",
            letterSpacing: "0.06em",
          }}
        >
          {selectedKey}
        </div>
      </header>
      <div
        ref={wheelRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "1 / 1",
          maxWidth: "340px",
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          touchAction: "none",
        }}
      >
        <div
          style={{
            position: "absolute",
            width: "76%",
            height: "76%",
            borderRadius: "50%",
            border: `1px solid rgba(255, 255, 255, 0.12)`,
            boxShadow: "inset 0 0 24px rgba(120, 203, 220, 0.25)",
            transform: `scale(${emphasis})`,
            transition: "transform 160ms ease",
          }}
        />
        {spokes.map((spoke) => {
          const outerAngle = spoke.angle + rotation;
          const innerAngle = spoke.angle + rotation + 15;
          const outerSelected = spoke.outerKey === selectedKey;
          const innerSelected = spoke.innerKey === selectedKey;
          return (
            <div key={spoke.outerKey} style={{ position: "absolute", inset: 0 }}>
              <WheelKey
                label={spoke.outerKey}
                angle={outerAngle}
                radius={130}
                color={spoke.color}
                selected={outerSelected}
                onSelect={() => onSelect(spoke.outerKey)}
              />
              <WheelKey
                label={spoke.innerKey}
                angle={innerAngle}
                radius={90}
                color={spoke.color}
                selected={innerSelected}
                onSelect={() => onSelect(spoke.innerKey)}
              />
            </div>
          );
        })}
      </div>
    </section>
  );

  function pointerAngle(x: number, y: number) {
    const element = wheelRef.current;
    if (!element) return null;
    const bounds = element.getBoundingClientRect();
    const centerX = bounds.left + bounds.width / 2;
    const centerY = bounds.top + bounds.height / 2;
    return Math.atan2(y - centerY, x - centerX);
  }
}

function WheelKey({
  label,
  angle,
  radius,
  color,
  selected,
  onSelect,
}: {
  label: string;
  angle: number;
  radius: number;
  color: string;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      style={{
        position: "absolute",
        left: "50%",
        top: "50%",
        transform: `rotate(${angle}deg) translate(0, -${radius}px) rotate(${-angle}deg)`,
        transformOrigin: "center",
        padding: "6px 12px",
        borderRadius: "12px",
        border: `1px solid ${color}`,
        background: selected
          ? `${color}88`
          : "linear-gradient(180deg, rgba(12, 28, 40, 0.9), rgba(4, 12, 20, 0.9))",
        color: selected ? theme.button.primaryText : color,
        fontSize: "0.7rem",
        letterSpacing: "0.08em",
        fontWeight: 600,
        cursor: "pointer",
        boxShadow: selected ? `0 0 18px ${color}` : "0 2px 8px rgba(0, 0, 0, 0.35)",
      }}
    >
      {label}
    </button>
  );
}

function clamp(value: number, min: number, max: number) {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function distanceBetween(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function normalizeDegrees(value: number) {
  let result = value % 360;
  if (result < 0) result += 360;
  return result;
}
