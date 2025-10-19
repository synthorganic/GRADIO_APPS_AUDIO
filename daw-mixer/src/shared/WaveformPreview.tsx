import { useMemo } from "react";

export interface WaveformPreviewProps {
  waveform: Float32Array;
  /**
   * Fill color for the waveform area. Defaults to a subtle white overlay.
   */
  fillColor?: string;
  /**
   * Stroke color for the waveform outline. Defaults to a semi-opaque white.
   */
  strokeColor?: string;
}

interface WaveformPathResult {
  commands: string;
  viewWidth: number;
}

function createWaveformPath(waveform: Float32Array): WaveformPathResult | null {
  const length = waveform.length;
  if (length === 0) return null;

  const viewWidth = Math.max(length - 1, 1);
  const halfHeight = 50;
  const scale = viewWidth === 0 ? 0 : viewWidth;
  const divisor = Math.max(length - 1, 1);

  let commands = `M 0 ${halfHeight}`;
  for (let index = 0; index < length; index += 1) {
    const amplitude = Math.min(1, Math.max(0, waveform[index] ?? 0));
    const x = (index / divisor) * scale;
    const y = halfHeight - amplitude * (halfHeight - 4);
    commands += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  for (let index = length - 1; index >= 0; index -= 1) {
    const amplitude = Math.min(1, Math.max(0, waveform[index] ?? 0));
    const x = (index / divisor) * scale;
    const y = halfHeight + amplitude * (halfHeight - 4);
    commands += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  commands += " Z";

  return { commands, viewWidth };
}

export function WaveformPreview({
  waveform,
  fillColor = "rgba(255, 255, 255, 0.25)",
  strokeColor = "rgba(255, 255, 255, 0.4)",
}: WaveformPreviewProps) {
  const path = useMemo(() => createWaveformPath(waveform), [waveform]);

  if (!path) return null;

  return (
    <svg
      width="100%"
      height="100%"
      viewBox={`0 0 ${Math.max(path.viewWidth, 1)} 100`}
      preserveAspectRatio="none"
      style={{ pointerEvents: "none" }}
    >
      <path d={path.commands} fill={fillColor} stroke={strokeColor} strokeWidth={0.5} />
    </svg>
  );
}

export type { WaveformPathResult };
export { createWaveformPath };
