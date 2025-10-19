import { useMemo, useRef } from "react";
import type { PointerEvent } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle } from "@daw/components/layout/styles";

const OUTER_RING = ["12A", "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "11A"];
const INNER_RING = ["12B", "1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B", "9B", "10B", "11B"];
const WHEEL_SIZE = 820;
const VISIBLE_OVERLAP = 48;
const WINDOW_HEIGHT = Math.round(WHEEL_SIZE / 2 + VISIBLE_OVERLAP);

export interface HarmonicWheelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  onOpenKey?: (value: string) => void;
}

type ActivePointer = {
  id: number;
  ring: "outer" | "inner";
};

export function HarmonicWheelSelector({ value, onChange, onOpenKey }: HarmonicWheelSelectorProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pointer = useRef<ActivePointer | null>(null);

  const segments = useMemo(() => {
    const entries: Array<{ key: string; ring: "outer" | "inner"; angle: number }> = [];
    OUTER_RING.forEach((key, index) => {
      entries.push({ key, ring: "outer", angle: (index / OUTER_RING.length) * 360 });
    });
    INNER_RING.forEach((key, index) => {
      entries.push({ key, ring: "inner", angle: (index / INNER_RING.length) * 360 });
    });
    return entries;
  }, []);

  const selectFromEvent = (event: PointerEvent<Element>, shouldOpen = false) => {
    const element = containerRef.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    const centerX = bounds.left + bounds.width / 2;
    const centerY = bounds.top + bounds.height / 2;
    const dx = event.clientX - centerX;
    const dy = event.clientY - centerY;
    const radius = Math.hypot(dx, dy);
    if (radius < 36 || radius > bounds.width / 2) return;

    const ring: "outer" | "inner" = radius > bounds.width * 0.28 ? "outer" : "inner";
    pointer.current = { id: event.pointerId, ring };
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    const normalized = (angle + 450) % 360;
    const index = Math.round(normalized / 30) % 12;
    const key = ring === "outer" ? OUTER_RING[index] : INNER_RING[index];
    onChange(key);
    if (shouldOpen) {
      onOpenKey?.(key);
    }
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    selectFromEvent(event, true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!pointer.current || pointer.current.id !== event.pointerId) {
      return;
    }
    selectFromEvent(event);
  };

  const handlePointerUp = () => {
    pointer.current = null;
  };

  return (
    <section
      style={{
        ...cardSurfaceStyle,
        padding: "24px",
        display: "grid",
        gap: "20px",
        justifyItems: "center",
      }}
    >
      <header style={{ textAlign: "center" }}>
        <h2 style={{ margin: 0, fontSize: "0.95rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Harmonic Wheel
        </h2>
        <p style={{ margin: "8px 0 0", fontSize: "0.7rem", color: theme.textMuted }}>
          Sweep the Camelot circle—only half is visible so you focus on neighbouring keys.
        </p>
      </header>
      <div
        style={{
          width: "100%",
          maxWidth: `${WHEEL_SIZE + 120}px`,
          height: `${WINDOW_HEIGHT}px`,
          borderRadius: `${WHEEL_SIZE}px ${WHEEL_SIZE}px 0 0`,
          border: `1px solid rgba(120, 203, 220, 0.26)`,
          background: "linear-gradient(180deg, rgba(9, 29, 42, 0.9) 0%, rgba(5, 18, 27, 0.98) 90%)",
          position: "relative",
          overflow: "hidden",
          boxShadow: "inset 0 22px 60px rgba(4, 18, 28, 0.5)",
        }}
      >
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{
            position: "absolute",
            width: `${WHEEL_SIZE}px`,
            height: `${WHEEL_SIZE}px`,
            borderRadius: "50%",
            border: `1px solid rgba(120, 203, 220, 0.32)`,
            background: "radial-gradient(circle, rgba(13, 55, 72, 0.85) 0%, rgba(3, 18, 26, 0.95) 70%)",
            display: "grid",
            placeItems: "center",
            left: "50%",
            transform: "translateX(-50%)",
            bottom: `${-Math.round(WHEEL_SIZE / 2 - VISIBLE_OVERLAP)}px`,
          }}
        >
          <div
            style={{
              position: "absolute",
              width: "72px",
              height: "72px",
              borderRadius: "50%",
              border: `1px solid rgba(132, 94, 255, 0.5)`,
              background: "rgba(12, 44, 63, 0.85)",
              display: "grid",
              placeItems: "center",
              fontSize: "1.1rem",
              fontWeight: 700,
              letterSpacing: "0.08em",
            }}
          >
            {value}
          </div>
          <div
            style={{
              position: "absolute",
              width: "220px",
              height: "220px",
              borderRadius: "50%",
              border: `1px solid rgba(126, 154, 230, 0.35)`,
            }}
          />
          <div
            style={{
              position: "absolute",
              width: "320px",
              height: "320px",
              borderRadius: "50%",
              border: `1px solid rgba(255, 148, 241, 0.35)`,
            }}
          />
          {segments.map(({ key, ring, angle }) => {
            const radius = ring === "outer" ? 210 : 150;
            const radians = ((angle - 90) * Math.PI) / 180;
            const x = Math.cos(radians) * radius;
            const y = Math.sin(radians) * radius;
            const isActive = value === key;
            return (
              <button
                key={`${ring}-${key}`}
                type="button"
                onPointerDown={(event) => {
                  selectFromEvent(event, true);
                }}
                onClick={() => {
                  onOpenKey?.(key);
                }}
                style={{
                  position: "absolute",
                  left: "50%",
                  top: "50%",
                  transform: `translate(-50%, -50%) translate(${x}px, ${y}px)`,
                  padding: "10px 14px",
                  borderRadius: "999px",
                  border: `1px solid ${isActive ? theme.button.primary : "rgba(120, 203, 220, 0.3)"}`,
                  background: isActive ? theme.button.primary : "rgba(9, 43, 60, 0.8)",
                  color: isActive ? theme.button.primaryText : theme.text,
                  fontSize: "0.72rem",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                {key}
              </button>
            );
          })}
          <div
            style={{
              position: "absolute",
              bottom: `${-Math.round(WHEEL_SIZE / 2)}px`,
              left: "-20%",
              width: "140%",
              height: `${Math.round(WHEEL_SIZE / 2 + 120)}px`,
              background: "linear-gradient(180deg, rgba(9, 29, 42, 0) 0%, rgba(9, 29, 42, 0.95) 85%)",
              pointerEvents: "none",
            }}
          />
        </div>
      </div>
      <footer style={{ fontSize: "0.68rem", color: theme.textMuted }}>
        Rotate or tap to choose a compatible mix key.
      </footer>
    </section>
  );
}
