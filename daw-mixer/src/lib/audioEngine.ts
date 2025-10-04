import type { Measure, SampleClip, StemInfo } from "../types";

type PlaybackTarget = StemInfo | SampleClip;

function isSampleClip(target: PlaybackTarget): target is SampleClip {
  return (target as SampleClip).file !== undefined || (target as SampleClip).url !== undefined;
}

function getPlaybackOffsets(target: PlaybackTarget) {
  return {
    startOffset: target.startOffset ?? 0,
    duration: target.duration
  };
}

export interface AudioEngineOptions {
  onMeasure?: (measure: Measure) => void;
}

export class AudioEngine {
  private context = new AudioContext();
  private gainNode = this.context.createGain();
  private currentSource: AudioBufferSourceNode | null = null;
  private startTime = 0;
  private startOffset = 0;
  private scheduledMeasures: Measure[] = [];
  private onMeasure?: (measure: Measure) => void;

  constructor(options: AudioEngineOptions = {}) {
    this.gainNode.connect(this.context.destination);
    this.onMeasure = options.onMeasure;
  }

  async decodeSample(target: PlaybackTarget): Promise<AudioBuffer | null> {
    try {
      if (isSampleClip(target) && target.file) {
        const arrayBuffer = await target.file.arrayBuffer();
        return this.context.decodeAudioData(arrayBuffer);
      }
      if (target.url) {
        const response = await fetch(target.url);
        const arrayBuffer = await response.arrayBuffer();
        return this.context.decodeAudioData(arrayBuffer);
      }
    } catch (error) {
      console.error("Failed to decode sample", error);
    }
    return null;
  }

  async play(target: PlaybackTarget, measures?: Measure[]) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop();
    this.currentSource = this.context.createBufferSource();
    this.currentSource.buffer = buffer;
    this.currentSource.connect(this.gainNode);
    this.startTime = this.context.currentTime;
    this.startOffset = 0;
    this.scheduledMeasures = measures ?? [];
    const { startOffset, duration } = getPlaybackOffsets(target);
    if (duration !== undefined) {
      this.currentSource.start(0, startOffset, duration);
    } else {
      this.currentSource.start(0, startOffset);
    }
    this.watchMeasures();
  }

  async playSegment(target: PlaybackTarget, segmentStart: number, segmentDuration: number) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop();
    this.currentSource = this.context.createBufferSource();
    this.currentSource.buffer = buffer;
    this.currentSource.connect(this.gainNode);
    const { startOffset } = getPlaybackOffsets(target);
    const start = startOffset + segmentStart;
    this.currentSource.start(0, start, segmentDuration);
    this.startTime = this.context.currentTime;
    this.startOffset = start;
    this.scheduledMeasures = [];
    this.watchMeasures();
  }

  stop() {
    if (this.currentSource) {
      this.currentSource.stop();
      this.currentSource.disconnect();
      this.currentSource = null;
    }
    this.startTime = 0;
    this.startOffset = 0;
    this.scheduledMeasures = [];
  }

  setVolume(value: number) {
    this.gainNode.gain.setTargetAtTime(value, this.context.currentTime, 0.05);
  }

  private watchMeasures() {
    if (!this.currentSource || this.scheduledMeasures.length === 0) return;
    const check = () => {
      if (!this.currentSource) return;
      const elapsed = this.context.currentTime - this.startTime + this.startOffset;
      const measure = this.scheduledMeasures.find((m) => elapsed >= m.start && elapsed < m.end);
      if (measure) {
        this.onMeasure?.(measure);
      }
      if (this.currentSource) {
        requestAnimationFrame(check);
      }
    };
    requestAnimationFrame(check);
  }
}

export const audioEngine = new AudioEngine();
