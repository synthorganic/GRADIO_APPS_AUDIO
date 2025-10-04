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
  private timelineOffset = 0;
  private rafId: number | null = null;

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

  async play(
    target: PlaybackTarget,
    measures?: Measure[],
    options: { timelineOffset?: number } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop(false);
    this.currentSource = this.context.createBufferSource();
    this.currentSource.buffer = buffer;
    this.currentSource.connect(this.gainNode);
    this.startTime = this.context.currentTime;
    this.startOffset = 0;
    this.scheduledMeasures = measures ?? [];
    this.timelineOffset = options.timelineOffset ?? 0;
    const { startOffset, duration } = getPlaybackOffsets(target);
    if (duration !== undefined) {
      this.currentSource.start(0, startOffset, duration);
    } else {
      this.currentSource.start(0, startOffset);
    }
    this.currentSource.onended = () => {
      this.finalizePlayback();
    };
    this.dispatchPlayEvent();
    this.watchMeasures();
  }

  async playSegment(
    target: PlaybackTarget,
    segmentStart: number,
    segmentDuration: number,
    options: { timelineOffset?: number } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop(false);
    this.currentSource = this.context.createBufferSource();
    this.currentSource.buffer = buffer;
    this.currentSource.connect(this.gainNode);
    const { startOffset } = getPlaybackOffsets(target);
    const start = startOffset + segmentStart;
    this.currentSource.start(0, start, segmentDuration);
    this.startTime = this.context.currentTime;
    this.startOffset = start;
    this.scheduledMeasures = [];
    this.timelineOffset = options.timelineOffset ?? 0;
    this.currentSource.onended = () => {
      this.finalizePlayback();
    };
    this.dispatchPlayEvent();
    this.watchMeasures();
  }

  stop(emitEvent = true) {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (error) {
        // Source may already be stopped; ignore.
      }
      this.currentSource.disconnect();
      this.currentSource.onended = null;
      this.currentSource = null;
    }
    this.finalizePlayback(emitEvent);
  }

  setVolume(value: number) {
    this.gainNode.gain.setTargetAtTime(value, this.context.currentTime, 0.05);
  }

  isPlaying() {
    return this.currentSource !== null;
  }

  getPlaybackPosition() {
    if (!this.currentSource) return null;
    return this.context.currentTime - this.startTime + this.startOffset;
  }

  private watchMeasures() {
    if (!this.currentSource) return;
    const check = () => {
      if (!this.currentSource) return;
      const elapsed = this.context.currentTime - this.startTime + this.startOffset;
      if (this.scheduledMeasures.length > 0) {
        const measure = this.scheduledMeasures.find((m) => elapsed >= m.start && elapsed < m.end);
        if (measure) {
          this.onMeasure?.(measure);
        }
      }
      this.dispatchTickEvent(elapsed);
      if (this.currentSource) {
        this.rafId = requestAnimationFrame(check);
      }
    };
    this.rafId = requestAnimationFrame(check);
  }

  private finalizePlayback(emitEvent = true) {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    if (emitEvent) {
      this.dispatchStopEvent();
    }
    this.startTime = 0;
    this.startOffset = 0;
    this.scheduledMeasures = [];
    this.timelineOffset = 0;
  }

  private dispatchPlayEvent() {
    if (typeof window === "undefined") return;
    window.dispatchEvent(
      new CustomEvent("audio-play", {
        detail: {
          timelineOffset: this.timelineOffset
        }
      })
    );
  }

  private dispatchStopEvent() {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent("audio-stop"));
  }

  private dispatchTickEvent(position: number) {
    if (typeof window === "undefined") return;
    window.dispatchEvent(
      new CustomEvent("audio-tick", {
        detail: {
          timelineOffset: this.timelineOffset,
          position
        }
      })
    );
  }
}

export const audioEngine = new AudioEngine();
