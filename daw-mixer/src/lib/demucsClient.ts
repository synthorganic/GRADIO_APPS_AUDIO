import { nanoid } from "nanoid";
import type { Measure, SampleClip, StemInfo } from "../types";

export const CAMEL0T_KEYS = [
  "1A",
  "2A",
  "3A",
  "4A",
  "5A",
  "6A",
  "7A",
  "8A",
  "9A",
  "10A",
  "11A",
  "12A",
  "1B",
  "2B",
  "3B",
  "4B",
  "5B",
  "6B",
  "7B",
  "8B",
  "9B",
  "10B",
  "11B",
  "12B"
];

const NOTE_NAMES = [
  "C",
  "C♯",
  "D",
  "E♭",
  "E",
  "F",
  "F♯",
  "G",
  "G♯",
  "A",
  "B♭",
  "B"
];

export interface DemucsResult {
  stems: StemInfo[];
  measures: Measure[];
  bpm: number;
  key: string;
}

const STEM_TYPES: Array<{
  type: StemInfo["type"];
  label: string;
  color: string;
  model?: string;
  notes?: string;
}> = [
  { type: "full", label: "Full Mix", color: "#f08ab9", notes: "Reference stereo bounce" },
  {
    type: "vocals",
    label: "Vocal",
    color: "#f2b08d",
    model: "UVR-MDX-NET Karaoke v2",
    notes: "De-essed to calm top-end shimmer"
  },
  {
    type: "leads",
    label: "Leads",
    color: "#f7d86d",
    model: "UVR-MDX-NET Main",
    notes: "High-pass focused to keep bass spill minimal"
  },
  { type: "percussion", label: "High Drums", color: "#87c7de" },
  {
    type: "kicks",
    label: "Kicks",
    color: "#7cd4c2",
    model: "UVR-MDX-NET Percussion v3",
    notes: "Sub energy trimmed for tighter punch"
  },
  { type: "bass", label: "Bassline", color: "#4c6edb" }
];

export async function runDemucs(file: File | undefined): Promise<DemucsResult> {
  if (!file) {
    throw new Error("DEMUCs requires a file input");
  }
  await new Promise((resolve) => setTimeout(resolve, 1200));

  const totalDuration = Math.max(file.size / 44100 / 2, 8);
  const bpm = Math.max(60, Math.min(160, Math.round((file.size % 80000) / 800 + 90)));
  const measureDuration = (60 / bpm) * 4;
  const measureCount = Math.max(2, Math.round(totalDuration / measureDuration));
  const measures: Measure[] = Array.from({ length: measureCount }).map((_, index) => {
    const start = measureDuration * index;
    const detectedPitch = NOTE_NAMES[(index * 3 + file.size) % NOTE_NAMES.length];
    return {
      id: nanoid(),
      start,
      end: start + measureDuration,
      beatCount: 4,
      isDownbeat: true,
      detectedPitch,
      tunedPitch: detectedPitch,
      energy: Math.round(((index + 1) / measureCount) * 100) / 100
    };
  });

  const stems: StemInfo[] = STEM_TYPES.map((stem) => ({
    id: nanoid(),
    name: stem.label,
    type: stem.type,
    color: stem.color,
    startOffset: 0,
    duration: measureDuration * measureCount,
    extractionModel: stem.model,
    processingNotes: stem.notes
  }));

  return {
    stems,
    measures,
    bpm,
    key: CAMEL0T_KEYS[file.size % CAMEL0T_KEYS.length]
  };
}

export function attachDemucsToSample(sample: SampleClip, result: DemucsResult): SampleClip {
  const lastMeasure = result.measures.length > 0 ? result.measures[result.measures.length - 1] : undefined;
  return {
    ...sample,
    stems: result.stems,
    measures: result.measures,
    bpm: result.bpm,
    key: result.key,
    length: lastMeasure ? lastMeasure.end : sample.length,
    duration: lastMeasure ? lastMeasure.end : sample.duration,
    startOffset: sample.startOffset ?? 0
  };
}
