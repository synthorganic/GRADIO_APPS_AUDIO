import type { Measure, SampleClip, StemInfo, TimelineChannel } from "../types";

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
  private activeSources: AudioBufferSourceNode[] = [];
  private startTime = 0;
  private startOffset = 0;
  private scheduledMeasures: Measure[] = [];
  private onMeasure?: (measure: Measure) => void;
  private timelineOffset = 0;
  private rafId: number | null = null;
  private stemChain: AudioNode[] = [];
  private playbackDuration = 0;
  private shouldEmitTimelineEvents = true;
  private channelMix = new Map<string, { volume: number; pan: number }>();

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
    options: { timelineOffset?: number; emitTimelineEvents?: boolean } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop(false);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    this.connectThroughChain(source, []);
    this.activeSources = [source];
    this.startTime = this.context.currentTime;
    this.startOffset = 0;
    this.scheduledMeasures = measures ?? [];
    this.timelineOffset = options.timelineOffset ?? 0;
    this.shouldEmitTimelineEvents = options.emitTimelineEvents ?? true;
    const { startOffset, duration } = getPlaybackOffsets(target);
    if (duration !== undefined) {
      source.start(0, startOffset, duration);
      this.playbackDuration = duration;
    } else {
      source.start(0, startOffset);
      this.playbackDuration = buffer.duration - startOffset;
    }
    source.onended = () => {
      this.handleSourceEnded(source);
    };
    this.dispatchPlayEvent();
    this.startWatch();
  }

  async playSegment(
    target: PlaybackTarget,
    segmentStart: number,
    segmentDuration: number,
    options: { timelineOffset?: number; emitTimelineEvents?: boolean } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop(false);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    this.connectThroughChain(source, []);
    const { startOffset } = getPlaybackOffsets(target);
    const start = startOffset + segmentStart;
    source.start(0, start, segmentDuration);
    this.activeSources = [source];
    this.startTime = this.context.currentTime;
    this.startOffset = start;
    this.scheduledMeasures = [];
    this.timelineOffset = options.timelineOffset ?? 0;
    this.shouldEmitTimelineEvents = options.emitTimelineEvents ?? true;
    this.playbackDuration = segmentDuration;
    source.onended = () => {
      this.handleSourceEnded(source);
    };
    this.dispatchPlayEvent();
    this.startWatch();
  }

  async playStem(
    sample: SampleClip,
    stem: StemInfo,
    measures?: Measure[],
    options: { timelineOffset?: number; emitTimelineEvents?: boolean } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(sample);
    if (!buffer) return;

    this.stop(false);
    const source = this.context.createBufferSource();
    source.buffer = buffer;

    const chain = this.buildStemChain(stem.type);
    this.connectThroughChain(source, chain);

    const { startOffset, duration } = getPlaybackOffsets(stem);
    this.activeSources = [source];
    this.startTime = this.context.currentTime;
    this.startOffset = startOffset ?? 0;
    this.scheduledMeasures = measures ?? [];
    this.timelineOffset = options.timelineOffset ?? 0;
    this.shouldEmitTimelineEvents = options.emitTimelineEvents ?? true;

    if (duration !== undefined) {
      source.start(0, startOffset ?? 0, duration);
      this.playbackDuration = duration;
    } else {
      source.start(0, startOffset ?? 0);
      this.playbackDuration = buffer.duration - (startOffset ?? 0);
    }

    source.onended = () => {
      this.handleSourceEnded(source);
    };

    this.dispatchPlayEvent();
    this.startWatch();
  }

  async playTimeline(clips: SampleClip[]) {
    const playable = clips.filter((clip) => clip.isInTimeline !== false);
    if (playable.length === 0) return;

    await this.context.resume();
    this.stop(false);
    this.shouldEmitTimelineEvents = true;
    this.timelineOffset = 0;
    this.scheduledMeasures = [];

    const buffers = await Promise.all(playable.map((clip) => this.decodeSample(clip)));
    const now = this.context.currentTime + 0.05;
    this.startTime = now;
    this.startOffset = 0;

    let longest = 0;
    const sources: AudioBufferSourceNode[] = [];

    playable.forEach((clip, index) => {
      const buffer = buffers[index];
      if (!buffer) return;
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      const mix = clip.channelId ? this.channelMix.get(clip.channelId) : undefined;
      if (mix) {
        const gain = this.context.createGain();
        gain.gain.value = mix.volume;
        const panner = this.context.createStereoPanner();
        panner.pan.value = mix.pan;
        source.connect(gain);
        gain.connect(panner);
        panner.connect(this.gainNode);
      } else {
        source.connect(this.gainNode);
      }
      const { startOffset, duration } = getPlaybackOffsets(clip);
      const when = now + clip.position;
      if (duration !== undefined) {
        source.start(when, startOffset, duration);
        longest = Math.max(longest, clip.position + duration);
      } else {
        const clipDuration = clip.duration ?? clip.length ?? buffer.duration - startOffset;
        source.start(when, startOffset);
        longest = Math.max(longest, clip.position + clipDuration);
      }
      source.onended = () => this.handleSourceEnded(source);
      sources.push(source);
    });

    this.playbackDuration = longest;
    this.activeSources = sources;
    this.dispatchPlayEvent();
    this.startWatch();
  }

  stop(emitEvent = true) {
    if (this.activeSources.length > 0) {
      this.activeSources.forEach((source) => {
        try {
          source.stop();
        } catch (error) {
          // Source may already be stopped.
        }
        source.disconnect();
      });
      this.activeSources = [];
    }
    this.finalizePlayback(emitEvent);
  }

  setVolume(value: number) {
    this.gainNode.gain.setTargetAtTime(value, this.context.currentTime, 0.05);
  }

  setChannelMix(channelId: string, mix: { volume: number; pan: number }) {
    this.channelMix.set(channelId, mix);
  }

  syncChannelMix(channels: TimelineChannel[]) {
    this.channelMix.clear();
    channels.forEach((channel) => {
      this.channelMix.set(channel.id, {
        volume: channel.volume ?? 0.85,
        pan: channel.pan ?? 0,
      });
    });
  }

  isPlaying() {
    return this.activeSources.length > 0;
  }

  getPlaybackPosition() {
    if (this.activeSources.length === 0) return null;
    const elapsed = this.context.currentTime - this.startTime + this.startOffset;
    return Math.max(0, Math.min(elapsed, this.playbackDuration));
  }

  private startWatch() {
    if (!this.shouldEmitTimelineEvents) return;
    if (this.activeSources.length === 0) return;
    const check = () => {
      if (this.activeSources.length === 0) return;
      const elapsed = this.context.currentTime - this.startTime + this.startOffset;
      if (this.scheduledMeasures.length > 0) {
        const measure = this.scheduledMeasures.find((item) => elapsed >= item.start && elapsed < item.end);
        if (measure) {
          this.onMeasure?.(measure);
        }
      }
      this.dispatchTickEvent(elapsed);
      if (this.activeSources.length > 0) {
        this.rafId = requestAnimationFrame(check);
      }
    };
    this.rafId = requestAnimationFrame(check);
  }

  private handleSourceEnded(source: AudioBufferSourceNode) {
    this.activeSources = this.activeSources.filter((item) => item !== source);
    if (this.activeSources.length === 0) {
      this.finalizePlayback();
    }
  }

  private finalizePlayback(emitEvent = true) {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.cleanupStemChain();
    if (emitEvent && this.shouldEmitTimelineEvents) {
      this.dispatchStopEvent();
    }
    this.startTime = 0;
    this.startOffset = 0;
    this.scheduledMeasures = [];
    this.timelineOffset = 0;
    this.playbackDuration = 0;
    this.shouldEmitTimelineEvents = true;
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
    if (typeof window === "undefined" || !this.shouldEmitTimelineEvents) return;
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
    if (typeof window === "undefined" || !this.shouldEmitTimelineEvents) return;
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
