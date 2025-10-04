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
  private stemChain: AudioNode[] = [];

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
    this.connectThroughChain(this.currentSource, []);
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
    this.connectThroughChain(this.currentSource, []);
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
    this.cleanupStemChain();
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

  async playStem(
    sample: SampleClip,
    stem: StemInfo,
    measures?: Measure[],
    options: { timelineOffset?: number } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(sample);
    if (!buffer) return;

    this.stop(false);
    this.currentSource = this.context.createBufferSource();
    this.currentSource.buffer = buffer;

    const chain = this.buildStemChain(stem.type);
    this.stemChain = chain;
    this.connectThroughChain(this.currentSource, chain);

    const { startOffset, duration } = getPlaybackOffsets(stem);
    this.startTime = this.context.currentTime;
    this.startOffset = startOffset ?? 0;
    this.scheduledMeasures = measures ?? [];
    this.timelineOffset = options.timelineOffset ?? 0;

    if (duration !== undefined) {
      this.currentSource.start(0, startOffset ?? 0, duration);
    } else {
      this.currentSource.start(0, startOffset ?? 0);
    }

    this.currentSource.onended = () => {
      this.finalizePlayback();
    };

    this.dispatchPlayEvent();
    this.watchMeasures();
  }

  private finalizePlayback(emitEvent = true) {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.cleanupStemChain();
    if (emitEvent) {
      this.dispatchStopEvent();
    }
    this.startTime = 0;
    this.startOffset = 0;
    this.scheduledMeasures = [];
    this.timelineOffset = 0;
  }

  private cleanupStemChain() {
    if (this.stemChain.length === 0) return;
    this.stemChain.forEach((node) => {
      try {
        node.disconnect();
      } catch (error) {
        // Ignore disconnect errors when nodes are already detached.
      }
    });
    this.stemChain = [];
  }

  private connectThroughChain(source: AudioNode, chain: AudioNode[]) {
    this.stemChain = chain;
    if (chain.length === 0) {
      source.connect(this.gainNode);
      return;
    }
    let previous: AudioNode = source;
    chain.forEach((node) => {
      previous.connect(node);
      previous = node;
    });
    previous.connect(this.gainNode);
  }

  private buildStemChain(type: StemInfo["type"]): AudioNode[] {
    const createFilter = (
      filterType: BiquadFilterType,
      frequency: number,
      q = 0.8,
      gain = 0
    ): BiquadFilterNode => {
      const filter = this.context.createBiquadFilter();
      filter.type = filterType;
      filter.frequency.value = frequency;
      filter.Q.value = q;
      if (filterType === "peaking" || filterType === "lowshelf" || filterType === "highshelf") {
        filter.gain.value = gain;
      }
      return filter;
    };

    switch (type) {
      case "vocals": {
        const highPass = createFilter("highpass", 140, 0.7);
        const presence = createFilter("peaking", 3200, 1.4, 6);
        const air = createFilter("lowpass", 7800, 0.9);
        return [highPass, presence, air];
      }
      case "leads": {
        const body = createFilter("highpass", 260, 0.9);
        const focus = createFilter("peaking", 1800, 1.1, 7);
        const shimmer = createFilter("lowpass", 5200, 0.85);
        return [body, focus, shimmer];
      }
      case "percussion": {
        const snap = createFilter("highpass", 2000, 0.7);
        const sparkle = createFilter("peaking", 5200, 1.2, 5);
        return [snap, sparkle];
      }
      case "kicks": {
        const thump = createFilter("lowshelf", 110, 0.7, 8);
        const tighten = createFilter("lowpass", 220, 0.9);
        return [thump, tighten];
      }
      case "bass": {
        const lowCut = createFilter("highpass", 35, 0.8);
        const warmth = createFilter("lowpass", 320, 0.85);
        return [lowCut, warmth];
      }
      default:
        return [];
    }
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
