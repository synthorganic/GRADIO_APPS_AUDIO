import type { StemType } from "../types";

const KRUMHANSL_MAJOR_PROFILE = [
  6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

const KRUMHANSL_MINOR_PROFILE = [
  6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

const CAMEL0T_MAJOR_MAP = [
  "8B",
  "3B",
  "10B",
  "5B",
  "12B",
  "7B",
  "2B",
  "9B",
  "4B",
  "11B",
  "6B",
  "1B",
] as const;

const CAMEL0T_MINOR_MAP = [
  "5A",
  "12A",
  "7A",
  "2A",
  "9A",
  "4A",
  "11A",
  "6A",
  "1A",
  "8A",
  "3A",
  "10A",
] as const;

const PITCH_CLASS_COUNT = 12;
const DEFAULT_KEY = "8A";

const STEM_FREQUENCIES: Record<StemType, number[]> = {
  drums: [60, 90, 120, 180, 220, 260],
  synths: [320, 480, 720, 960, 1320, 1800],
  vocals: [2200, 3200, 4200, 5600, 7200, 8800],
};

const STEM_LABELS: Record<StemType, string> = {
  drums: "Drums",
  synths: "Synths",
  vocals: "Vocals",
};

export interface AudioAnalysisStem {
  type: StemType;
  label: string;
}

export interface AudioAnalysisResult {
  bpm: number;
  camelotKey: string;
  stems: AudioAnalysisStem[];
  durationSeconds: number;
}

let sharedContext: AudioContext | null = null;

async function ensureContext(): Promise<AudioContext> {
  if (sharedContext) {
    return sharedContext;
  }
  if (typeof AudioContext === "undefined") {
    throw new Error("AudioContext is not available in this environment");
  }
  sharedContext = new AudioContext();
  return sharedContext;
}

async function decodeAudioBuffer(data: ArrayBuffer): Promise<AudioBuffer> {
  const context = await ensureContext();
  const copy = data.slice(0);
  return context.decodeAudioData(copy);
}

function mixToMono(buffer: AudioBuffer, windowSeconds = 45): Float32Array {
  const sampleRate = buffer.sampleRate;
  const maxSamples = Math.min(buffer.length, Math.floor(windowSeconds * sampleRate));
  if (maxSamples <= 0) {
    return new Float32Array(0);
  }
  const channels = Math.max(1, buffer.numberOfChannels);
  const output = new Float32Array(maxSamples);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < maxSamples; index += 1) {
      output[index] += data[index] ?? 0;
    }
  }
  for (let index = 0; index < maxSamples; index += 1) {
    output[index] /= channels;
  }
  return output;
}

function downsample(signal: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (targetRate >= sourceRate) {
    return signal.slice();
  }
  const ratio = sourceRate / targetRate;
  const length = Math.max(1, Math.floor(signal.length / ratio));
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(signal.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    const width = Math.max(1, end - start);
    for (let cursor = start; cursor < end; cursor += 1) {
      sum += signal[cursor];
    }
    output[index] = sum / width;
  }
  return output;
}

function applyHannWindow(samples: Float32Array): Float32Array {
  const windowed = new Float32Array(samples.length);
  const lastIndex = samples.length - 1;
  for (let index = 0; index < samples.length; index += 1) {
    const weight = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, lastIndex));
    windowed[index] = samples[index] * weight;
  }
  return windowed;
}

function goertzelPower(samples: Float32Array, sampleRate: number, frequency: number): number {
  if (frequency <= 0 || frequency >= sampleRate / 2) {
    return 0;
  }
  const windowed = applyHannWindow(samples);
  const omega = (2 * Math.PI * frequency) / sampleRate;
  const cosine = Math.cos(omega);
  const coeff = 2 * cosine;
  let sPrev = 0;
  let sPrev2 = 0;
  for (let index = 0; index < windowed.length; index += 1) {
    const sample = windowed[index];
    const s = sample + coeff * sPrev - sPrev2;
    sPrev2 = sPrev;
    sPrev = s;
  }
  const power = sPrev2 * sPrev2 + sPrev * sPrev - coeff * sPrev * sPrev2;
  return Math.max(0, power);
}

function computeEnergyEnvelope(samples: Float32Array): Float32Array {
  const envelope = new Float32Array(samples.length);
  const smoothing = 0.92;
  let running = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const magnitude = Math.abs(samples[index]);
    running = smoothing * running + (1 - smoothing) * magnitude;
    envelope[index] = magnitude - running;
  }
  // Remove DC offset
  let sum = 0;
  for (let index = 0; index < envelope.length; index += 1) {
    sum += envelope[index];
  }
  const mean = sum / Math.max(1, envelope.length);
  for (let index = 0; index < envelope.length; index += 1) {
    envelope[index] -= mean;
  }
  return envelope;
}

function estimateTempo(samples: Float32Array, sampleRate: number): number {
  if (samples.length < 2) {
    return 120;
  }
  const envelope = computeEnergyEnvelope(samples);
  const minBpm = 70;
  const maxBpm = 180;
  const minLag = Math.max(2, Math.floor((sampleRate * 60) / maxBpm));
  const maxLag = Math.max(minLag + 1, Math.floor((sampleRate * 60) / minBpm));
  let bestLag = minLag;
  let bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let score = 0;
    for (let index = 0; index + lag < envelope.length; index += 1) {
      score += envelope[index] * envelope[index + lag];
    }
    const normalized = score / Math.max(1, envelope.length - lag);
    if (normalized > bestScore) {
      bestScore = normalized;
      bestLag = lag;
    }
  }
  let bpm = (60 * sampleRate) / bestLag;
  while (bpm < minBpm) bpm *= 2;
  while (bpm > maxBpm) bpm /= 2;
  return Math.round(bpm);
}

function computePitchClassProfile(samples: Float32Array, sampleRate: number): Float32Array {
  const profile = new Float32Array(PITCH_CLASS_COUNT);
  const minMidi = 36; // C2
  const maxMidi = 83; // B5
  for (let midi = minMidi; midi <= maxMidi; midi += 1) {
    const frequency = 440 * Math.pow(2, (midi - 69) / 12);
    const energy = goertzelPower(samples, sampleRate, frequency);
    const pitchClass = midi % PITCH_CLASS_COUNT;
    profile[pitchClass] += energy;
  }
  return profile;
}

function rotateProfile(profile: readonly number[], steps: number): number[] {
  const rotated: number[] = [];
  for (let index = 0; index < profile.length; index += 1) {
    rotated[index] = profile[(index + steps) % profile.length];
  }
  return rotated;
}

function pearsonCorrelation(a: Float32Array, b: number[]): number {
  let sumA = 0;
  let sumB = 0;
  let sumASq = 0;
  let sumBSq = 0;
  let sumProduct = 0;
  const length = a.length;
  for (let index = 0; index < length; index += 1) {
    const av = a[index];
    const bv = b[index];
    sumA += av;
    sumB += bv;
    sumASq += av * av;
    sumBSq += bv * bv;
    sumProduct += av * bv;
  }
  const numerator = length * sumProduct - sumA * sumB;
  const denominator = Math.sqrt(Math.max(length * sumASq - sumA * sumA, 1e-6)) *
    Math.sqrt(Math.max(length * sumBSq - sumB * sumB, 1e-6));
  if (denominator <= 0) {
    return 0;
  }
  return numerator / denominator;
}

function detectCamelotKey(samples: Float32Array, sampleRate: number): string {
  if (!samples.length) {
    return DEFAULT_KEY;
  }
  const profile = computePitchClassProfile(samples, sampleRate);
  let bestScore = -Infinity;
  let bestIndex = 0;
  let bestIsMajor = false;
  for (let pitch = 0; pitch < PITCH_CLASS_COUNT; pitch += 1) {
    const majorScore = pearsonCorrelation(profile, rotateProfile(KRUMHANSL_MAJOR_PROFILE, pitch));
    if (majorScore > bestScore) {
      bestScore = majorScore;
      bestIndex = pitch;
      bestIsMajor = true;
    }
    const minorScore = pearsonCorrelation(profile, rotateProfile(KRUMHANSL_MINOR_PROFILE, pitch));
    if (minorScore > bestScore) {
      bestScore = minorScore;
      bestIndex = pitch;
      bestIsMajor = false;
    }
  }
  if (!Number.isFinite(bestScore) || bestScore <= 0) {
    return DEFAULT_KEY;
  }
  return bestIsMajor ? CAMEL0T_MAJOR_MAP[bestIndex] : CAMEL0T_MINOR_MAP[bestIndex];
}

function computeStemEnergies(samples: Float32Array, sampleRate: number): Map<StemType, number> {
  const energies = new Map<StemType, number>();
  (Object.entries(STEM_FREQUENCIES) as Array<[StemType, number[]]>).forEach(([stem, freqs]) => {
    let total = 0;
    freqs.forEach((freq) => {
      total += goertzelPower(samples, sampleRate, freq);
    });
    energies.set(stem, total);
  });
  return energies;
}

function deriveStemOrder(samples: Float32Array, sampleRate: number): AudioAnalysisStem[] {
  const energies = computeStemEnergies(samples, sampleRate);
  const ordered = Array.from(energies.entries()).sort((a, b) => b[1] - a[1]);
  const strongest = ordered[0]?.[1] ?? 0;
  const stems: AudioAnalysisStem[] = [];
  ordered.forEach(([stem, energy]) => {
    // Always include the top stem, and include others when they carry at least 15% of peak energy
    if (!stems.length || strongest <= 0 || energy / strongest >= 0.15) {
      stems.push({ type: stem, label: STEM_LABELS[stem] });
    }
  });
  // Ensure all three categories are available to keep the UI consistent
  (Object.keys(STEM_LABELS) as StemType[]).forEach((stem) => {
    if (!stems.find((entry) => entry.type === stem)) {
      stems.push({ type: stem, label: STEM_LABELS[stem] });
    }
  });
  return stems;
}

export async function analyzeAudioFile(file: File): Promise<AudioAnalysisResult> {
  const arrayBuffer = await file.arrayBuffer();
  return analyzeAudioData(arrayBuffer);
}

export async function analyzeAudioData(arrayBuffer: ArrayBuffer): Promise<AudioAnalysisResult> {
  const buffer = await decodeAudioBuffer(arrayBuffer);
  const mono = mixToMono(buffer);
  const analysisWindowSeconds = Math.min(buffer.duration, 45);
  const truncatedSamples = mono.subarray(0, Math.min(mono.length, Math.floor(buffer.sampleRate * analysisWindowSeconds)));
  const tempoSamples = downsample(truncatedSamples, buffer.sampleRate, 2000);
  const pitchSamples = downsample(truncatedSamples, buffer.sampleRate, 4000);
  const bpm = estimateTempo(tempoSamples, 2000);
  const camelotKey = detectCamelotKey(pitchSamples, 4000);
  const stems = deriveStemOrder(pitchSamples, 4000);
  return {
    bpm,
    camelotKey,
    stems,
    durationSeconds: Number.isFinite(buffer.duration) ? buffer.duration : tempoSamples.length / 2000,
  };
}

export async function closeSharedContext(): Promise<void> {
  if (sharedContext) {
    const context = sharedContext;
    sharedContext = null;
    await context.close();
  }
}
