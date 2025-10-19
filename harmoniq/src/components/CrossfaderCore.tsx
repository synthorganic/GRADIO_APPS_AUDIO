import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import { theme } from "@daw-theme";

export interface CrossfaderCoreProps {
  value: number; // -1 (Deck A) to +1 (Deck B)
  onChange: (value: number) => void;
  filters: Record<"A" | "B", number>;
  onFilterChange: (deck: "A" | "B", value: number) => void;
  focus: number;
  onFocusChange: (value: number) => void;
}

type PadState =
  | {
      type: "drag";
      pointerId: number;
      originX: number;
      originY: number;
      originValue: number;
      originFilters: Record<"A" | "B", number>;
    }
  | {
      type: "pinch";
      pointerIds: [number, number];
      baseDistance: number;
      baseFocus: number;
    };

export function CrossfaderCore({
  value,
  onChange,
  filters,
  onFilterChange,
  focus,
  onFocusChange,
}: CrossfaderCoreProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const handleRef = useRef<HTMLDivElement | null>(null);
  const padRef = useRef<HTMLDivElement | null>(null);
  const padState = useRef<PadState | null>(null);
  const pointerPositions = useRef(new Map<number, { x: number; y: number }>());

  const leftClip = value < -0.78;
  const rightClip = value > 0.78;

  function handleHandlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!trackRef.current) return;
    padState.current = {
      type: "drag",
      pointerId: event.pointerId,
      originX: event.clientX,
      originY: event.clientY,
      originValue: value,
      originFilters: { ...filters },
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    pointerPositions.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
  }

  function handleHandlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const state = padState.current;
    if (!state || state.type !== "drag" || state.pointerId !== event.pointerId) return;
    pointerPositions.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (!trackRef.current) return;
    const bounds = trackRef.current.getBoundingClientRect();
    const deltaX = (event.clientX - state.originX) / bounds.width;
    const nextValue = clamp(state.originValue + deltaX * 2, -1, 1);
    onChange(nextValue);
  }

  function handleHandlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (padState.current?.type === "drag" && padState.current.pointerId === event.pointerId) {
      padState.current = null;
    }
    pointerPositions.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function handlePadPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    pointerPositions.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointerPositions.current.size === 1) {
      padState.current = {
        type: "drag",
        pointerId: event.pointerId,
        originX: event.clientX,
        originY: event.clientY,
        originValue: value,
        originFilters: { ...filters },
      };
    } else if (pointerPositions.current.size === 2) {
      const entries = Array.from(pointerPositions.current.entries());
      const first = entries[0];
      const second = entries[1];
      const distance = distanceBetween(first[1], second[1]);
      padState.current = {
        type: "pinch",
        pointerIds: [first[0], second[0]],
        baseDistance: Math.max(distance, 1),
        baseFocus: focus,
      };
    }
  }

  function handlePadPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const position = pointerPositions.current.get(event.pointerId);
    if (!position) return;
    position.x = event.clientX;
    position.y = event.clientY;
    pointerPositions.current.set(event.pointerId, position);

    const state = padState.current;
    if (!state) return;

    if (state.type === "drag" && state.pointerId === event.pointerId) {
      if (!padRef.current) return;
      const bounds = padRef.current.getBoundingClientRect();
      const deltaX = (event.clientX - state.originX) / bounds.width;
      const deltaY = (state.originY - event.clientY) / bounds.height;
      const nextValue = clamp(state.originValue + deltaX * 2, -1, 1);
      onChange(nextValue);
      const nextFilterA = clamp(state.originFilters.A + deltaY);
      const nextFilterB = clamp(state.originFilters.B - deltaY);
      onFilterChange("A", nextFilterA);
      onFilterChange("B", nextFilterB);
    }

    if (state.type === "pinch" && state.pointerIds.includes(event.pointerId)) {
      const [firstId, secondId] = state.pointerIds;
      const first = pointerPositions.current.get(firstId);
      const second = pointerPositions.current.get(secondId);
      if (!first || !second) return;
      const distance = Math.max(distanceBetween(first, second), 1);
      const ratio = clamp(distance / state.baseDistance, 0.4, 1.8);
      onFocusChange(clamp(state.baseFocus * ratio, 0.45, 1.75));
    }
  }

  function handlePadPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    pointerPositions.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const state = padState.current;
    if (!state) return;
    if (state.type === "drag" && state.pointerId === event.pointerId) {
      padState.current = null;
      return;
    }
    if (state.type === "pinch" && state.pointerIds.includes(event.pointerId)) {
      padState.current = null;
      const remaining = Array.from(pointerPositions.current.entries());
      if (remaining.length === 1) {
        const [id, pos] = remaining[0];
        padState.current = {
          type: "drag",
          pointerId: id,
          originX: pos.x,
          originY: pos.y,
          originValue: value,
          originFilters: { ...filters },
        };
      }
    }
  }

  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        padding: "20px",
        borderRadius: "18px",
        background: theme.surfaceOverlay,
        border: `1px solid ${theme.border}`,
        boxShadow: theme.cardGlow,
        color: theme.text,
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <div>
          <h2
            style={{
              margin: 0,
              fontSize: "0.95rem",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
            }}
          >
            XY Crossfader Core
          </h2>
          <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
            Drag to blend · Pinch to tune focus
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <ClippingPulse active={leftClip} color={theme.accentBeam[0]} label="Deck A" />
          <ClippingPulse active={rightClip} color={theme.accentBeam[4]} label="Deck B" />
        </div>
      </header>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr auto",
          alignItems: "center",
          gap: "20px",
        }}
      >
        <FilterSlider deck="A" value={filters.A} color={theme.accentBeam[0]} onChange={(next) => onFilterChange("A", next)} />
        <div
          ref={padRef}
          onPointerDown={handlePadPointerDown}
          onPointerMove={handlePadPointerMove}
          onPointerUp={handlePadPointerUp}
          onPointerLeave={handlePadPointerUp}
          style={{
            position: "relative",
            height: "160px",
            borderRadius: "18px",
            border: `1px solid ${theme.border}`,
            background:
              "radial-gradient(circle at 50% 40%, rgba(124, 203, 220, 0.18), transparent 70%), rgba(7, 28, 36, 0.9)",
            overflow: "hidden",
            touchAction: "none",
          }}
        >
          <div
            ref={trackRef}
            style={{
              position: "absolute",
              top: "50%",
              left: "6%",
              right: "6%",
              height: "4px",
              borderRadius: "999px",
              background:
                "linear-gradient(90deg, rgba(241, 139, 180, 0.9), rgba(120, 203, 220, 0.35), rgba(129, 203, 255, 0.9))",
              transform: "translateY(-50%)",
              boxShadow: "0 0 24px rgba(120, 203, 220, 0.35)",
            }}
          />
          <div
            ref={handleRef}
            role="slider"
            aria-valuemin={-1}
            aria-valuemax={1}
            aria-valuenow={Number(value.toFixed(2))}
            onPointerDown={handleHandlePointerDown}
            onPointerMove={handleHandlePointerMove}
            onPointerUp={handleHandlePointerUp}
            onPointerLeave={handleHandlePointerUp}
            style={{
              position: "absolute",
              top: "50%",
              left: `${((value + 1) / 2) * 100}%`,
              width: "42px",
              height: "42px",
              borderRadius: "50%",
              border: `2px solid ${theme.button.outline}`,
              background:
                "linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0))",
              transform: "translate(-50%, -50%)",
              boxShadow: "0 10px 24px -12px rgba(0, 0, 0, 0.6)",
              cursor: "pointer",
              display: "grid",
              placeItems: "center",
              color: theme.text,
              fontSize: "0.72rem",
              fontWeight: 600,
              letterSpacing: "0.06em",
              userSelect: "none",
            }}
          >
            {value >= 0 ? `B ${(value * 50 + 50).toFixed(0)}%` : `A ${((1 + value) * 50).toFixed(0)}%`}
          </div>
          <FocusRing focus={focus} />
        </div>
        <FilterSlider deck="B" value={filters.B} color={theme.accentBeam[4]} onChange={(next) => onFilterChange("B", next)} />
      </div>
      <footer
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "0.72rem",
          color: theme.textMuted,
        }}
      >
        <span>Focus {focus.toFixed(2)} · pinch inward to tighten harmonics</span>
        <span>Filters offset to preserve stereo energy</span>
      </footer>
    </section>
  );
}

function FilterSlider({
  deck,
  value,
  color,
  onChange,
}: {
  deck: "A" | "B";
  value: number;
  color: string;
  onChange: (value: number) => void;
}) {
  const sliderRef = useRef<HTMLDivElement | null>(null);
  const dragState = useRef<{
    pointerId: number;
    originY: number;
    originValue: number;
  } | null>(null);

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!sliderRef.current) return;
    dragState.current = {
      pointerId: event.pointerId,
      originY: event.clientY,
      originValue: value,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const state = dragState.current;
    if (!state || state.pointerId !== event.pointerId || !sliderRef.current) return;
    const bounds = sliderRef.current.getBoundingClientRect();
    const delta = (state.originY - event.clientY) / bounds.height;
    onChange(clamp(state.originValue + delta, 0, 1));
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    if (dragState.current?.pointerId === event.pointerId) {
      dragState.current = null;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "12px",
      }}
    >
      <span style={{ fontSize: "0.7rem", letterSpacing: "0.08em", color }}>{deck} FILTER</span>
      <div
        ref={sliderRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{
          position: "relative",
          width: "46px",
          height: "160px",
          borderRadius: "24px",
          border: `1px solid ${theme.border}`,
          background: `linear-gradient(180deg, ${color}55 0%, rgba(12, 40, 52, 0.8) 100%)`,
          boxShadow: "inset 0 0 12px rgba(0, 0, 0, 0.45)",
          cursor: "pointer",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: "50%",
            transform: "translateX(-50%)",
            bottom: `${value * 100}%`,
            width: "26px",
            height: "26px",
            borderRadius: "50%",
            border: `2px solid ${color}`,
            background: "rgba(255, 255, 255, 0.16)",
            boxShadow: `${color}55 0 0 18px`,
            transition: "bottom 120ms ease", 
          }}
        />
      </div>
    </div>
  );
}

function FocusRing({ focus }: { focus: number }) {
  return (
    <div
      style={{
        position: "absolute",
        inset: "18%",
        borderRadius: "50%",
        border: `1px solid rgba(120, 203, 220, 0.28)`,
        transform: `scale(${focus})`,
        transition: "transform 160ms ease",
        boxShadow: "0 0 28px rgba(120, 203, 220, 0.35)",
        pointerEvents: "none",
      }}
    />
  );
}

function ClippingPulse({ active, color, label }: { active: boolean; color: string; label: string }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        fontSize: "0.68rem",
        color: active ? color : theme.textMuted,
      }}
    >
      <span
        style={{
          width: "14px",
          height: "14px",
          borderRadius: "50%",
          background: color,
          opacity: active ? 1 : 0.4,
          animation: active ? "clipPulse 1s ease-out infinite" : "none",
          boxShadow: active ? `0 0 16px ${color}` : `0 0 8px ${color}66`,
        }}
      />
      {label}
    </span>
  );
}

function clamp(value: number, min = 0, max = 1) {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function distanceBetween(
  first: { x: number; y: number },
  second: { x: number; y: number },
) {
  return Math.hypot(second.x - first.x, second.y - first.y);
}
