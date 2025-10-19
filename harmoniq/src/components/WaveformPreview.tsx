import { memo, useMemo } from "react";
import type { CSSProperties } from "react";
import { theme } from "@daw/theme";

export interface WaveformPreviewProps {
  waveform: Float32Array;
  height?: number;
  fillColor?: string;
  strokeColor?: string;
  background?: string;
  style?: CSSProperties;
}

const SAMPLE_WIDTH = 160;

function createWaveformPath(waveform: Float32Array, height: number): string {
  const width = SAMPLE_WIDTH;
  const usableSamples = waveform.length;
  if (usableSamples === 0) {
    const center = height / 2;
    return `M 0 ${center} L ${width} ${center}`;
  }

  const targetPoints = Math.max(32, Math.min(usableSamples, 160));
  const step = usableSamples / targetPoints;
  const peaks: number[] = [];
  for (let index = 0; index < targetPoints; index += 1) {
    const start = Math.floor(index * step);
    const end = Math.min(usableSamples, Math.floor((index + 1) * step));
    let peak = 0;
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      const value = waveform[sampleIndex] ?? 0;
      peak = Math.max(peak, Math.abs(value));
    }
    peaks.push(peak);
  }

  const maxPeak = peaks.reduce((acc, value) => Math.max(acc, value), 0) || 1;
  const center = height / 2;
  const amplitudeScale = Math.max(center - 2, 0);

  let path = `M 0 ${center.toFixed(2)}`;
  const lastIndex = peaks.length - 1;
  peaks.forEach((peak, index) => {
    const normalized = peak / maxPeak;
    const amplitude = normalized * amplitudeScale;
    const x = (index / lastIndex) * width;
    const y = center - amplitude;
    path += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  path += ` L ${width.toFixed(2)} ${center.toFixed(2)}`;
  for (let index = peaks.length - 1; index >= 0; index -= 1) {
    const peak = peaks[index];
    const normalized = peak / maxPeak;
    const amplitude = normalized * amplitudeScale;
    const x = (index / lastIndex) * width;
    const y = center + amplitude;
    path += ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  path += " Z";
  return path;
}

export const WaveformPreview = memo(function WaveformPreview({
  waveform,
  height = 52,
  fillColor = "rgba(103, 255, 230, 0.75)",
  strokeColor = theme.border,
  background = "rgba(9, 33, 46, 0.75)",
  style,
}: WaveformPreviewProps) {
  const path = useMemo(() => createWaveformPath(waveform, height), [waveform, height]);

  return (
    <svg
      viewBox={`0 0 ${SAMPLE_WIDTH} ${height}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height, display: "block", borderRadius: "12px", background, ...style }}
      role="img"
      aria-label="Waveform preview"
    >
      <path d={path} fill={fillColor} fillOpacity={0.85} stroke={strokeColor} strokeOpacity={0.45} strokeWidth={0.5} />
    </svg>
  );
});
