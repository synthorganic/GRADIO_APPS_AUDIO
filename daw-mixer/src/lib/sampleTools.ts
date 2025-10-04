import { nanoid } from "nanoid";
import type { Beat, Measure, SampleClip, StemInfo } from "../types";
import { cloneTrackEffects } from "./effectPresets";

export interface SliceOptions {
  variantLabel?: string;
  position?: number;
  stems?: StemInfo[];
  isFragment?: boolean;
}

export function sliceStemsForRange(
  sample: SampleClip,
  start: number,
  end: number,
  stems: StemInfo[] | undefined = sample.stems
): StemInfo[] {
  const duration = Math.max(0, end - start);
  const baseOffset = sample.startOffset ?? 0;
  return stems.map((stem) => ({
    ...stem,
    id: `${stem.id}-slice-${start.toFixed(3)}-${end.toFixed(3)}`,
    sourceStemId: stem.sourceStemId ?? stem.id,
    startOffset: (stem.startOffset ?? baseOffset) + start,
    duration
  }));
}

export function sliceSampleSegment(
  sample: SampleClip,
  start: number,
  end: number,
  options: SliceOptions = {}
): SampleClip {
  const duration = Math.max(0, end - start);
  const measures = sample.measures
    .filter((measure) => measure.end > start && measure.start < end)
    .map((measure) => {
      const adjustedStart = Math.max(measure.start, start) - start;
      const adjustedEnd = Math.min(measure.end, end) - start;
      const beats = measure.beats
        ?.filter((beat) => beat.end > start && beat.start < end)
        .map((beat) => ({
          ...beat,
          id: `${beat.id}-slice-${start.toFixed(3)}-${end.toFixed(3)}`,
          start: Math.max(beat.start, start) - start,
          end: Math.min(beat.end, end) - start,
          stems: sliceStemsForRange(sample, beat.start, beat.end, beat.stems)
        }));
      return {
        ...measure,
        id: nanoid(),
        start: adjustedStart,
        end: adjustedEnd,
        beats,
        isDownbeat: measure.isDownbeat && adjustedStart === 0
      };
    });

  return {
    id: nanoid(),
    name: sample.name,
    variantLabel: options.variantLabel,
    file: sample.file,
    url: sample.url,
    bpm: sample.bpm,
    key: sample.key,
    measures,
    stems: sliceStemsForRange(sample, start, end, options.stems),
    position: options.position ?? 0,
    length: duration,
    duration,
    isLooping: sample.isLooping,
    startOffset: (sample.startOffset ?? 0) + start,
    originSampleId: sample.originSampleId ?? sample.id,
    isFragment: options.isFragment ?? true,
    retuneMap: sample.retuneMap,
    rekeyedAt: sample.rekeyedAt,
    effects: cloneTrackEffects(sample.effects)
  };
}

export function combineMeasures(measures: Measure[], groupSize: number): Measure[] {
  if (groupSize <= 1) return measures;
  const combined: Measure[] = [];
  let cursor = 0;
  for (let index = 0; index < measures.length; index += groupSize) {
    const bucket = measures.slice(index, index + groupSize);
    if (bucket.length === 0) continue;
    const duration = bucket[bucket.length - 1].end - bucket[0].start;
    const measure: Measure = {
      ...bucket[0],
      id: nanoid(),
      start: cursor,
      end: cursor + duration,
      beatCount: bucket.reduce((total, item) => total + item.beatCount, 0),
      detectedPitch: bucket[0].detectedPitch,
      tunedPitch: bucket[bucket.length - 1].tunedPitch ?? bucket[bucket.length - 1].detectedPitch,
      energy:
        bucket.reduce((total, item) => total + (item.energy ?? 0), 0) / Math.max(bucket.length, 1),
      beats: undefined,
      isDownbeat: index % groupSize === 0
    };
    combined.push(measure);
    cursor += duration;
  }

  return combined;
}

export function generateBeatsForMeasure(sample: SampleClip, measure: Measure): Beat[] {
  const beatCount = measure.beatCount || 4;
  const measureDuration = Math.max(0, measure.end - measure.start);
  if (measureDuration === 0) return [];
  const beatDuration = measureDuration / beatCount;
  return Array.from({ length: beatCount }).map((_, index) => {
    const start = measure.start + beatDuration * index;
    const end = start + beatDuration;
    return {
      id: nanoid(),
      start,
      end,
      index,
      stems: sliceStemsForRange(sample, start, end)
    };
  });
}
