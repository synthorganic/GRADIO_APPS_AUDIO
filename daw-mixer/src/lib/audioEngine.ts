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

interface PlaybackOptions {
  timelineOffset?: number;
  emitTimelineEvents?: boolean;
  loop?: boolean;
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
  private bufferCache = new Map<string, Promise<AudioBuffer | null>>();

  constructor(options: AudioEngineOptions = {}) {
    this.gainNode.connect(this.context.destination);
    this.onMeasure = options.onMeasure;
  }

  private getCacheKey(target: PlaybackTarget) {
    if (isSampleClip(target)) {
      if (target.originSampleId) {
        return `sample:${target.originSampleId}`;
      }
      if (target.url) {
        return `url:${target.url}`;
      }
      if (target.file) {
        const { name, size, lastModified } = target.file;
        return `file:${name}:${size}:${lastModified}`;
      }
      return `sample:${target.id}`;
    }
    if (target.sourceStemId) {
      return `stem:${target.sourceStemId}`;
    }
    if (target.id) {
      return `stem:${target.id}`;
    }
    return null;
  }

  private async fetchAndDecode(target: PlaybackTarget): Promise<AudioBuffer | null> {
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

  async decodeSample(target: PlaybackTarget): Promise<AudioBuffer | null> {
    const cacheKey = this.getCacheKey(target);
    if (!cacheKey) {
      return this.fetchAndDecode(target);
    }
    const existing = this.bufferCache.get(cacheKey);
    if (existing) {
      return existing;
    }
    const pending = this.fetchAndDecode(target).then((buffer) => {
      if (!buffer) {
        this.bufferCache.delete(cacheKey);
      }
      return buffer;
    });
    this.bufferCache.set(cacheKey, pending);
    return pending;
  }

  async play(target: PlaybackTarget, measures?: Measure[], options: PlaybackOptions = {}) {
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
    const baseStart = startOffset ?? 0;
    const playbackDuration = duration ?? buffer.duration - baseStart;
    const shouldLoop = options.loop ?? false;
    source.loop = shouldLoop;
    if (shouldLoop) {
      source.loopStart = baseStart;
      source.loopEnd = baseStart + playbackDuration;
      source.start(0, baseStart);
      this.playbackDuration = playbackDuration;
    } else if (duration !== undefined) {
      source.start(0, baseStart, duration);
      this.playbackDuration = duration;
    } else {
      source.start(0, baseStart);
      this.playbackDuration = playbackDuration;
    }
    source.onended = () => {
      this.handleSourceEnded(source);
    };
    this.dispatchPlayEvent();
    this.startWatch();
  }

  async triggerOneShot(
    target: PlaybackTarget,
    semitoneOffset = 0,
    options: { duration?: number } = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    this.connectThroughChain(source, []);
    const playbackRate = 2 ** (semitoneOffset / 12);
    source.playbackRate.value = Number.isFinite(playbackRate) ? playbackRate : 1;

    const { startOffset, duration } = getPlaybackOffsets(target);
    const start = startOffset ?? 0;
    const playDuration = options.duration ?? duration ?? buffer.duration - start;
    source.start(0, start, playDuration);
    source.onended = () => {
      source.disconnect();
    };
  }

  async playSegment(
    target: PlaybackTarget,
    segmentStart: number,
    segmentDuration: number,
    options: PlaybackOptions = {}
  ) {
    await this.context.resume();
    const buffer = await this.decodeSample(target);
    if (!buffer) return;

    this.stop(false);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    this.connectThroughChain(source, []);
    const { startOffset } = getPlaybackOffsets(target);
    const start = (startOffset ?? 0) + segmentStart;
    const shouldLoop = options.loop ?? false;
    source.loop = shouldLoop;
    if (shouldLoop) {
      source.loopStart = start;
      source.loopEnd = start + segmentDuration;
      source.start(0, start);
    } else {
      source.start(0, start, segmentDuration);
    }
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
    options: PlaybackOptions = {}
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
    const baseStart = startOffset ?? 0;
    this.startOffset = baseStart;
    this.scheduledMeasures = measures ?? [];
    this.timelineOffset = options.timelineOffset ?? 0;
    this.shouldEmitTimelineEvents = options.emitTimelineEvents ?? true;

    const playbackDuration = duration ?? buffer.duration - baseStart;
    const shouldLoop = options.loop ?? false;
    source.loop = shouldLoop;
    if (shouldLoop) {
      source.loopStart = baseStart;
      source.loopEnd = baseStart + playbackDuration;
      source.start(0, baseStart);
      this.playbackDuration = playbackDuration;
    } else if (duration !== undefined) {
      source.start(0, baseStart, duration);
      this.playbackDuration = duration;
    } else {
      source.start(0, baseStart);
      this.playbackDuration = playbackDuration;
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

  async getWaveformPeaks(sample: SampleClip, resolution = 240): Promise<Float32Array | null> {
    const buffer = await this.decodeSample(sample);
    if (!buffer) return null;

    const { startOffset, duration } = getPlaybackOffsets(sample);
    const baseOffset = startOffset ?? 0;
    const clipDuration =
      duration ?? sample.duration ?? sample.length ?? Math.max(buffer.duration - baseOffset, 0);

    const startFrame = Math.max(0, Math.floor(baseOffset * buffer.sampleRate));
    const totalFrames = Math.floor(clipDuration * buffer.sampleRate);
    const endFrame = Math.min(buffer.length, startFrame + totalFrames);
    if (endFrame <= startFrame || totalFrames <= 0) {
      return new Float32Array();
    }

    const bucketCount = Math.max(1, Math.min(resolution, endFrame - startFrame));
    const peaks = new Float32Array(bucketCount);
    const framesPerBucket = (endFrame - startFrame) / bucketCount;

    for (let bucket = 0; bucket < bucketCount; bucket += 1) {
      const bucketStart = Math.floor(startFrame + bucket * framesPerBucket);
      const bucketEnd = Math.max(bucketStart + 1, Math.floor(startFrame + (bucket + 1) * framesPerBucket));
      let max = 0;
      for (let frame = bucketStart; frame < bucketEnd; frame += 1) {
        let amplitude = 0;
        for (let channelIndex = 0; channelIndex < buffer.numberOfChannels; channelIndex += 1) {
          const data = buffer.getChannelData(channelIndex);
          amplitude += Math.abs(data[frame] ?? 0);
        }
        amplitude /= buffer.numberOfChannels || 1;
        if (amplitude > max) {
          max = amplitude;
        }
      }
      peaks[bucket] = max;
    }

    return peaks;
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
        const highPass = createFilter("highpass", 170, 0.85);
        const presence = createFilter("peaking", 3200, 1.2, 4.5);
        const deEss = createFilter("peaking", 7800, 3.2, -6);
        const airTrim = createFilter("lowpass", 13500, 0.7);
        return [highPass, presence, deEss, airTrim];
      }
      case "leads": {
        const body = createFilter("highpass", 340, 0.95);
        const focus = createFilter("peaking", 2100, 1.05, 6.5);
        const sheen = createFilter("peaking", 5200, 1.1, 3);
        const air = createFilter("lowpass", 9000, 0.8);
        return [body, focus, sheen, air];
      }
      case "percussion": {
        const snap = createFilter("highpass", 2000, 0.7);
        const midAttenuation = createFilter("peaking", 950, 1.4, -8);
        const sparkle = createFilter("peaking", 5200, 1.2, 5);
        return [snap, midAttenuation, sparkle];
      }
      case "kicks": {
        const subTrim = createFilter("highpass", 45, 0.8);
        const punch = createFilter("peaking", 120, 1.05, 5.5);
        const tighten = createFilter("lowpass", 260, 0.85);
        const airDamp = createFilter("highshelf", 1200, 0.9, -6);
        return [subTrim, punch, tighten, airDamp];
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
