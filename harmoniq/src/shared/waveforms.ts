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

export function createWaveformFromAudioBuffer(buffer: AudioBuffer, length = 256): Float32Array {
  const result = new Float32Array(length);
  const totalSamples = buffer.length;
  const channels = buffer.numberOfChannels;
  if (!totalSamples || !channels) {
    return result;
  }
  const channelData = Array.from({ length: channels }, (_, channel) => buffer.getChannelData(channel));
  const step = totalSamples / length;
  for (let index = 0; index < length; index += 1) {
    const start = Math.floor(index * step);
    const end = Math.min(totalSamples, Math.floor((index + 1) * step));
    const width = Math.max(1, end - start);
    const stride = Math.max(1, Math.floor(width / 64));
    let accumulator = 0;
    let count = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const data = channelData[channel];
      for (let cursor = start; cursor < end; cursor += stride) {
        accumulator += Math.abs(data[cursor]);
        count += 1;
      }
    }
    result[index] = count ? Math.min(1, accumulator / count) : 0;
  }
  return result;
}

export function serializeWaveform(waveform: Float32Array): number[] {
  return Array.from(waveform);
}

export function deserializeWaveform(values: number[]): Float32Array {
  return new Float32Array(values);
}
