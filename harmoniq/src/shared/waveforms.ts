export function createWaveform(seed: number, length = 256): Float32Array {
  const buffer = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const t = index / length;
    const sine = Math.sin(2 * Math.PI * (seed + 1) * t);
    const warp = Math.sin(2 * Math.PI * (seed * 0.35 + 0.5) * t * 2);
    const value = Math.abs((sine + warp * 0.5) * (0.6 + 0.4 * Math.sin(t * Math.PI * seed)));
    buffer[index] = Math.min(1, Math.max(0, value));
  }
  return buffer;
}

export function serializeWaveform(waveform: Float32Array): number[] {
  return Array.from(waveform);
}

export function deserializeWaveform(values: number[]): Float32Array {
  return new Float32Array(values);
}
