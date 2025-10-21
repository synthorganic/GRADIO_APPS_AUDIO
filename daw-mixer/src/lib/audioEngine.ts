import type {
  AutomationChannel,
  MasteringSettings,
  Measure,
  SampleClip,
  StemInfo,
  TimelineChannel,
  TrackEffects,
} from "../types";

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

interface TransportState {
  isPlaying: boolean;
  bpm: number;
  timelineOffset: number;
  position: number | null;
  playbackDuration: number;
}

export class AudioEngine {
  private context = new AudioContext();
  private gainNode = this.context.createGain();
  private masterOut = this.context.createGain();
  // Mastering chain nodes
  private master = {
    splitter: this.context.createChannelSplitter(2),
    merger: this.context.createChannelMerger(2),
    // Mid/Side matrix
    midLeft: this.context.createGain(),
    midRight: this.context.createGain(),
    sideLeft: this.context.createGain(),
    sideRight: this.context.createGain(),
    sideScale: this.context.createGain(),
    leftOut: this.context.createGain(),
    rightOut: this.context.createGain(),
    // Processing
    glue: this.context.createDynamicsCompressor(),
    tiltLow: this.context.createBiquadFilter(),
    tiltHigh: this.context.createBiquadFilter(),
    tapeWet: this.context.createGain(),
    tapeDry: this.context.createGain(),
    tape: this.context.createWaveShaper(),
    limiter: this.context.createDynamicsCompressor(),
  };
  private masterAnalyser = this.context.createAnalyser();
  private bpm = 120;
  private automation: AutomationChannel[] = [];
  private activeSources: AudioBufferSourceNode[] = [];
  private startTime = 0;
  private startOffset = 0;
  private scheduledMeasures: Measure[] = [];
  private onMeasure?: (measure: Measure) => void;
  private timelineOffset = 0;
  private rafId: number | null = null;
  private stemChain: Array<{ inlet: AudioNode; outlet: AudioNode }> = [];
  private stemCleanups: Array<() => void> = [];
  private playbackDuration = 0;
  private shouldEmitTimelineEvents = true;
  private channelMix = new Map<string, { volume: number; pan: number }>();
  private channelBuses = new Map<
    string,
    {
      inlet: GainNode;
      panner: StereoPannerNode;
      analyser: AnalyserNode;
    }
  >();
  private bufferCache = new Map<string, Promise<AudioBuffer | null>>();

  constructor(options: AudioEngineOptions = {}) {
    this.initMasterChain();
    this.gainNode.connect(this.master.splitter);
    // Master chain to destination
    this.masterOut.connect(this.context.destination);
    this.onMeasure = options.onMeasure;
  }

  private initMasterChain() {
    // Configure fixed nodes
    // Mid (L+R)/2
    this.master.midLeft.gain.value = 0.5;
    this.master.midRight.gain.value = 0.5;
    // Side (L-R)/2
    this.master.sideLeft.gain.value = 0.5;
    this.master.sideRight.gain.value = -0.5;
    this.master.sideScale.gain.value = 0; // 0 = mono, 1 = original width

    // Tilt filters
    this.master.tiltLow.type = "lowshelf";
    this.master.tiltLow.frequency.value = 220;
    this.master.tiltHigh.type = "highshelf";
    this.master.tiltHigh.frequency.value = 3500;

    // Tape dry/wet init
    this.master.tapeDry.gain.value = 1;
    this.master.tapeWet.gain.value = 0; // default off
    this.master.tape.curve = this.makeSaturationCurve(0);

    // Limiter defaults (ceiling around -0.3 dB)
    this.master.limiter.threshold.value = -0.3;
    this.master.limiter.knee.value = 0;
    this.master.limiter.ratio.value = 20;
    this.master.limiter.attack.value = 0.003;
    this.master.limiter.release.value = 0.050;

    // Glue compressor defaults (subtle)
    this.master.glue.threshold.value = -18;
    this.master.glue.knee.value = 6;
    this.master.glue.ratio.value = 2.5;
    this.master.glue.attack.value = 0.020;
    this.master.glue.release.value = 0.150;

    // Wire M/S widen matrix
    // Split -> mid/side sums -> reconstruct -> processing -> masterOut
    const s = this.master;
    // Mid sum
    s.splitter.connect(s.midLeft, 0);
    s.splitter.connect(s.midRight, 1);
    s.midLeft.connect(s.leftOut);
    s.midRight.connect(s.leftOut);
    s.midLeft.connect(s.rightOut);
    s.midRight.connect(s.rightOut);
    // Side sum and scaling
    s.splitter.connect(s.sideLeft, 0);
    s.splitter.connect(s.sideRight, 1);
    s.sideLeft.connect(s.sideScale);
    s.sideRight.connect(s.sideScale);
    // Reconstruct L = mid + side, R = mid - side
    s.sideScale.connect(s.leftOut);
    const sideInvert = this.context.createGain();
    sideInvert.gain.value = -1;
    s.sideScale.connect(sideInvert);
    sideInvert.connect(s.rightOut);

    // Merge back to stereo
    s.leftOut.connect(s.merger, 0, 0);
    s.rightOut.connect(s.merger, 0, 1);

    // Post processing chain: merger -> glue -> tilt -> tape dry/wet -> limiter -> masterAnalyser -> masterOut
    s.merger.connect(s.glue);
    s.glue.connect(s.tiltLow);
    s.tiltLow.connect(s.tiltHigh);
    // Split to dry/wet tape
    s.tiltHigh.connect(s.tapeDry);
    s.tiltHigh.connect(s.tape);
    s.tape.connect(s.tapeWet);
    const tapeSum = this.context.createGain();
    s.tapeDry.connect(tapeSum);
    s.tapeWet.connect(tapeSum);
    tapeSum.connect(s.limiter);
    this.masterAnalyser.fftSize = 2048;
    this.masterAnalyser.smoothingTimeConstant = 0.8;
    s.limiter.connect(this.masterAnalyser);
    this.masterAnalyser.connect(this.masterOut);
  }

  private makeSaturationCurve(amount: number) {
    // amount: 0..1 -> gentle to stronger
    const k = Math.max(0, Math.min(1, amount)) * 12; // drive
    const n = 512;
    const curve = new Float32Array(n);
    for (let i = 0; i < n; i += 1) {
      const x = (i / (n - 1)) * 2 - 1;
      curve[i] = Math.tanh(k * x) / Math.tanh(k || 1);
    }
    return curve;
  }

  setTempo(bpm: number) {
    const next = Number.isFinite(bpm) ? bpm : 120;
    this.bpm = Math.max(30, Math.min(240, next));
  }

  setMastering(settings: MasteringSettings) {
    const now = this.context.currentTime;
    // Width (0..1)
    this.master.sideScale.gain.setTargetAtTime(
      Math.max(0, Math.min(1, settings.widenStereo ?? 0)),
      now,
      0.05,
    );
    // Glue: map 0..1 to threshold/ratio/attack
    const glue = Math.max(0, Math.min(1, settings.glueCompression ?? 0));
    const thr = -30 + glue * 20; // -30..-10 dB
    const ratio = 1.2 + glue * 3; // ~1.2..4.2
    const attack = 0.005 + glue * 0.040; // 5..45 ms
    this.master.glue.threshold.setTargetAtTime(thr, now, 0.1);
    this.master.glue.ratio.setTargetAtTime(ratio, now, 0.1);
    this.master.glue.attack.setTargetAtTime(attack, now, 0.1);

    // Spectral tilt: lowshelf -, highshelf + in dB
    const tilt = Math.max(0, Math.min(1, settings.spectralTilt ?? 0));
    const dB = (tilt - 0.5) * 9; // -4.5..+4.5
    this.master.tiltLow.gain.setTargetAtTime(-dB, now, 0.05);
    this.master.tiltHigh.gain.setTargetAtTime(dB, now, 0.05);

    // Limiter ceiling in dB (-2..0)
    const ceiling = settings.limiterCeiling ?? -0.3;
    this.master.limiter.threshold.setTargetAtTime(ceiling, now, 0.05);

    // Tape saturation mix and curve
    const tape = Math.max(0, Math.min(1, settings.tapeSaturation ?? 0));
    this.master.tapeWet.gain.setTargetAtTime(tape, now, 0.1);
    this.master.tapeDry.gain.setTargetAtTime(1 - tape * 0.6, now, 0.1);
    this.master.tape.curve = this.makeSaturationCurve(tape);
  }

  setAutomationChannels(channels: AutomationChannel[]) {
    this.automation = channels;
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
    const chain = isSampleClip(target) ? this.buildSamplePlaybackChain(target) : [];
    const dest = isSampleClip(target) && target.channelId && this.channelBuses.get(target.channelId)
      ? this.channelBuses.get(target.channelId)!.inlet
      : this.gainNode;
    this.connectThroughChain(source, chain, dest);
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
    this.scheduleAutomation();
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
    const chain = this.buildSamplePlaybackChain(target);
    const dest = isSampleClip(target) && target.channelId && this.channelBuses.get(target.channelId)
      ? this.channelBuses.get(target.channelId)!.inlet
      : this.gainNode;
    this.connectThroughChain(source, chain, dest);
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
    const chain = this.buildSamplePlaybackChain(target);
    const dest = isSampleClip(target) && target.channelId && this.channelBuses.get(target.channelId)
      ? this.channelBuses.get(target.channelId)!.inlet
      : this.gainNode;
    this.connectThroughChain(source, chain, dest);
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
    this.scheduleAutomation();
  }

  async playStem(
    sample: SampleClip,
    stem: StemInfo,
    measures?: Measure[],
    options: PlaybackOptions = {}
  ) {
    await this.context.resume();
    // Prefer decoded stem asset if provided; fallback to original sample
    const buffer = await this.decodeSample((stem.url ? (stem as unknown as PlaybackTarget) : sample));
    if (!buffer) return;

    this.stop(false);
    const source = this.context.createBufferSource();
    source.buffer = buffer;

    const chain = this.buildStemChain(stem.type);
    const dest = sample.channelId && this.channelBuses.get(sample.channelId)
      ? this.channelBuses.get(sample.channelId)!.inlet
      : this.gainNode;
    this.connectThroughChain(source, chain, dest);

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
      const chain = this.buildSamplePlaybackChain(clip);
      const dest = clip.channelId && this.channelBuses.get(clip.channelId)
        ? this.channelBuses.get(clip.channelId)!.inlet
        : this.gainNode;
      this.connectThroughChain(source, chain, dest);
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
    this.scheduleAutomation();
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
    let overallMax = 0;
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
      if (max > overallMax) {
        overallMax = max;
      }
    }

    if (overallMax > 0) {
      for (let bucket = 0; bucket < peaks.length; bucket += 1) {
        peaks[bucket] = Math.min(1, peaks[bucket] / overallMax);
      }
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
    const keepIds = new Set<string>();
    this.channelMix.clear();
    channels.forEach((channel) => {
      keepIds.add(channel.id);
      const volume = channel.volume ?? 0.85;
      const pan = channel.pan ?? 0;
      this.channelMix.set(channel.id, { volume, pan });
      let bus = this.channelBuses.get(channel.id);
      if (!bus) {
        const inlet = this.context.createGain();
        const panner = this.context.createStereoPanner();
        const analyser = this.context.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.8;
        inlet.connect(panner);
        panner.connect(analyser);
        analyser.connect(this.gainNode);
        bus = { inlet, panner, analyser };
        this.channelBuses.set(channel.id, bus);
      }
      bus.inlet.gain.value = volume;
      bus.panner.pan.value = pan;
    });
    // Cleanup removed
    for (const id of Array.from(this.channelBuses.keys())) {
      if (!keepIds.has(id)) {
        const bus = this.channelBuses.get(id)!;
        try { bus.inlet.disconnect(); } catch {}
        try { bus.panner.disconnect(); } catch {}
        try { bus.analyser.disconnect(); } catch {}
        this.channelBuses.delete(id);
      }
    }
  }

  isPlaying() {
    return this.activeSources.length > 0;
  }

  getPlaybackPosition() {
    if (this.activeSources.length === 0) return null;
    const elapsed = this.context.currentTime - this.startTime + this.startOffset;
    return Math.max(0, Math.min(elapsed, this.playbackDuration));
  }

  getTransportState(): TransportState {
    return {
      isPlaying: this.isPlaying(),
      bpm: this.bpm,
      timelineOffset: this.timelineOffset,
      position: this.getPlaybackPosition(),
      playbackDuration: this.playbackDuration,
    };
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
    if (this.stemCleanups.length > 0) {
      this.stemCleanups.forEach((fn) => {
        try {
          fn();
        } catch {
          // ignore
        }
      });
    }
    this.stemChain = [];
    this.stemCleanups = [];
  }

  private connectThroughChain(
    source: AudioNode,
    chain: Array<{ inlet: AudioNode; outlet: AudioNode }>,
    destination: AudioNode = this.gainNode,
  ) {
    this.stemChain = chain;
    if (chain.length === 0) {
      source.connect(destination);
      return;
    }
    let previous: AudioNode = source;
    chain.forEach((stage) => {
      previous.connect(stage.inlet);
      previous = stage.outlet;
    });
    previous.connect(destination);
  }

  private buildStemChain(type: StemInfo["type"]): Array<{ inlet: AudioNode; outlet: AudioNode }> {
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
        return [highPass, presence, deEss, airTrim].map((n) => ({ inlet: n, outlet: n }));
      }
      case "leads": {
        const body = createFilter("highpass", 340, 0.95);
        const focus = createFilter("peaking", 2100, 1.05, 6.5);
        const sheen = createFilter("peaking", 5200, 1.1, 3);
        const air = createFilter("lowpass", 9000, 0.8);
        return [body, focus, sheen, air].map((n) => ({ inlet: n, outlet: n }));
      }
      case "percussion": {
        const snap = createFilter("highpass", 2000, 0.7);
        const midAttenuation = createFilter("peaking", 950, 1.4, -8);
        const sparkle = createFilter("peaking", 5200, 1.2, 5);
        return [snap, midAttenuation, sparkle].map((n) => ({ inlet: n, outlet: n }));
      }
      case "kicks": {
        const subTrim = createFilter("highpass", 45, 0.8);
        const punch = createFilter("peaking", 120, 1.05, 5.5);
        const tighten = createFilter("lowpass", 260, 0.85);
        const airDamp = createFilter("highshelf", 1200, 0.9, -6);
        return [subTrim, punch, tighten, airDamp].map((n) => ({ inlet: n, outlet: n }));
      }
      case "bass": {
        const lowCut = createFilter("highpass", 35, 0.8);
        const warmth = createFilter("lowpass", 320, 0.85);
        return [lowCut, warmth].map((n) => ({ inlet: n, outlet: n }));
      }
      default:
        return [];
    }
  }

  private buildSamplePlaybackChain(sample: SampleClip): Array<{ inlet: AudioNode; outlet: AudioNode }> {
    const chain: Array<{ inlet: AudioNode; outlet: AudioNode }> = [];
    // Track effects
    if (sample.effects) {
      const fxChain = this.buildTrackEffectsChain(sample.effects);
      chain.push(...fxChain);
    }
    return chain;
  }

  getMasterLevels(): { rms: number; peak: number } {
    const analyser = this.masterAnalyser;
    const bins = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(bins);
    let sumSq = 0;
    let peak = 0;
    for (let i = 0; i < bins.length; i += 1) {
      const v = bins[i] ?? 0;
      sumSq += v * v;
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    const rms = Math.sqrt(sumSq / (bins.length || 1));
    return { rms, peak };
  }

  getChannelLevels(): Map<string, { rms: number; peak: number }> {
    const result = new Map<string, { rms: number; peak: number }>();
    this.channelBuses.forEach((bus, id) => {
      const bins = new Float32Array(bus.analyser.fftSize);
      bus.analyser.getFloatTimeDomainData(bins);
      let sumSq = 0;
      let peak = 0;
      for (let i = 0; i < bins.length; i += 1) {
        const v = bins[i] ?? 0;
        sumSq += v * v;
        const a = Math.abs(v);
        if (a > peak) peak = a;
      }
      result.set(id, { rms: Math.sqrt(sumSq / (bins.length || 1)), peak });
    });
    return result;
  }

  getMasterAnalyser(): AnalyserNode {
    return this.masterAnalyser;
  }

  private buildTrackEffectsChain(
    effects: TrackEffects,
  ): Array<{ inlet: AudioNode; outlet: AudioNode }> {
    const stages: Array<{ inlet: AudioNode; outlet: AudioNode }> = [];
    const makeDryWetStage = (wetNode: AudioNode, mix: number) => {
      const inlet = this.context.createGain();
      const dryGain = this.context.createGain();
      const wetGain = this.context.createGain();
      const sum = this.context.createGain();
      dryGain.gain.value = Math.max(0, Math.min(1, 1 - mix));
      wetGain.gain.value = Math.max(0, Math.min(1, mix));
      inlet.connect(dryGain);
      inlet.connect(wetNode);
      wetNode.connect(wetGain);
      dryGain.connect(sum);
      wetGain.connect(sum);
      return { inlet, outlet: sum } as const;
    };

    // Hi-pass
    if (effects.hiPass?.enabled) {
      const filter = this.context.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = effects.hiPass.cutoff;
      filter.Q.value = Math.max(0.0001, effects.hiPass.resonance);
      stages.push(makeDryWetStage(filter, effects.hiPass.mix));
    }
    // Low-pass
    if (effects.lowPass?.enabled) {
      const filter = this.context.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = effects.lowPass.cutoff;
      filter.Q.value = Math.max(0.0001, effects.lowPass.resonance);
      stages.push(makeDryWetStage(filter, effects.lowPass.mix));
    }
    // Chorus (simple modulated delay)
    if (effects.chorus?.enabled) {
      const delay = this.context.createDelay(0.05);
      delay.delayTime.value = 0.012; // base 12ms
      const lfo = this.context.createOscillator();
      const lfoDepth = this.context.createGain();
      lfo.type = "sine";
      lfo.frequency.value = Math.max(0.05, effects.chorus.rate);
      lfoDepth.gain.value = Math.max(0.0005, Math.min(0.01, effects.chorus.depth * 0.01));
      lfo.connect(lfoDepth);
      lfoDepth.connect(delay.delayTime);
      lfo.start();
      this.stemCleanups.push(() => {
        try {
          lfo.stop();
        } catch {}
      });
      stages.push(makeDryWetStage(delay, effects.chorus.mix));
    }
    // Delay (tempo-synced)
    if (effects.delay?.enabled) {
      const delay = this.context.createDelay(2);
      const feedback = this.context.createGain();
      const wetGain = this.context.createGain();
      // Compute time from bpm
      const beat = 60 / this.bpm;
      const map: Record<'1/8' | '1/4' | '1/2' | '1', number> = {
        "1/8": beat / 2,
        "1/4": beat,
        "1/2": beat * 2,
        "1": beat * 4,
      };
      delay.delayTime.value = map[effects.delay.time] ?? beat;
      feedback.gain.value = Math.max(0, Math.min(0.95, effects.delay.feedback));
      wetGain.gain.value = Math.max(0, Math.min(1, effects.delay.mix));
      // Subgraph
      const inlet = this.context.createGain();
      inlet.connect(delay);
      delay.connect(feedback);
      feedback.connect(delay);
      delay.connect(wetGain);
      const sum = this.context.createGain();
      inlet.connect(sum); // dry
      wetGain.connect(sum);
      stages.push({ inlet, outlet: sum });
    }
    // Reverb (generated IR)
    if (effects.reverb?.enabled) {
      const convolver = this.context.createConvolver();
      const length = Math.floor(this.context.sampleRate * (0.5 + effects.reverb.size * 2.0));
      const impulse = this.context.createBuffer(2, length, this.context.sampleRate);
      for (let c = 0; c < 2; c += 1) {
        const data = impulse.getChannelData(c);
        for (let i = 0; i < length; i += 1) {
          // simple noise decay
          const t = i / length;
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2 + effects.reverb.decay * 6);
        }
      }
      convolver.buffer = impulse;
      stages.push(makeDryWetStage(convolver, effects.reverb.mix));
    }
    // Bitcrush (amplitude quantization)
    if (effects.bitcrush?.enabled) {
      const shaper = this.context.createWaveShaper();
      const steps = Math.max(2, Math.min(65536, 2 ** Math.round(effects.bitcrush.depth)));
      const n = 512;
      const curve = new Float32Array(n);
      for (let i = 0; i < n; i += 1) {
        const x = (i / (n - 1)) * 2 - 1;
        const q = Math.round(((x + 1) / 2) * (steps - 1)) / (steps - 1);
        curve[i] = q * 2 - 1;
      }
      shaper.curve = curve;
      stages.push(makeDryWetStage(shaper, effects.bitcrush.mix));
    }
    // Rhythmic gate and slicer intentionally omitted for first pass wiring

    return stages;
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

  private scheduleAutomation() {
    if (this.automation.length === 0) return;
    const t0 = this.startTime;
    const offset = this.timelineOffset;
    const tEnd = t0 + this.playbackDuration;
    const clampTime = (when: number) => Math.max(t0, Math.min(tEnd, when));

    const setWiden = (v: number, when: number) => {
      this.master.sideScale.gain.setValueAtTime(Math.max(0, Math.min(1, v)), when);
    };
    const setTilt = (v: number, when: number) => {
      const dB = (Math.max(0, Math.min(1, v)) - 0.5) * 9;
      this.master.tiltLow.gain.setValueAtTime(-dB, when);
      this.master.tiltHigh.gain.setValueAtTime(dB, when);
    };
    const setLimiter = (v: number, when: number) => {
      // Expect -2..0; if normalized 0..1, map to -2..0
      const thr = v <= 0 ? v : -2 + v * 2;
      this.master.limiter.threshold.setValueAtTime(thr, when);
    };
    const setGlue = (v: number, when: number) => {
      const glue = Math.max(0, Math.min(1, v));
      const thr = -30 + glue * 20;
      const ratio = 1.2 + glue * 3;
      const attack = 0.005 + glue * 0.040;
      this.master.glue.threshold.setValueAtTime(thr, when);
      this.master.glue.ratio.setValueAtTime(ratio, when);
      this.master.glue.attack.setValueAtTime(attack, when);
    };
    const setBpm = (v: number, when: number) => {
      // BPM changes mid-playback affect tempo-synced effects only next chunks; apply immediately
      if (when <= this.context.currentTime + 0.01) {
        this.setTempo(v);
      } else {
        // Fallback: schedule by timeout
        const delayMs = (when - this.context.currentTime) * 1000;
        setTimeout(() => this.setTempo(v), Math.max(0, Math.floor(delayMs)));
      }
    };

    this.automation.forEach((channel) => {
      const id = channel.parameterId;
      channel.points.forEach((pt) => {
        const when = clampTime(t0 + Math.max(0, pt.time - offset));
        if (id === "master-widenStereo") setWiden(pt.value, when);
        if (id === "master-spectralTilt") setTilt(pt.value, when);
        if (id === "master-limiterCeiling") setLimiter(pt.value, when);
        if (id === "master-glueCompression") setGlue(pt.value, when);
        if (id === "master-bpm") setBpm(pt.value, when);
      });
    });
  }

  // --- Offline rendering (mixdown) ---
  async renderTimelineMix(
    clips: SampleClip[],
    mastering: MasteringSettings,
    channels: TimelineChannel[],
    options: { sampleRate?: number; duration?: number } = {},
  ): Promise<Blob> {
    const sr = Math.max(8000, Math.min(192000, Math.floor(options.sampleRate ?? 48000)));
    let duration = options.duration ?? 0;
    if (!options.duration) {
      clips.forEach((clip) => {
        const end = (clip.position ?? 0) + (clip.duration ?? clip.length ?? 0);
        duration = Math.max(duration, end);
      });
      duration = Math.max(duration, 0.1);
    }
    const length = Math.ceil(duration * sr);
    const ctx = new OfflineAudioContext(2, length, sr);

    const master = this.buildOfflineMasterChain(ctx, mastering);

    const mixMap = new Map<string, { volume: number; pan: number }>();
    channels.forEach((ch) => {
      mixMap.set(ch.id, { volume: ch.volume ?? 0.85, pan: ch.pan ?? 0 });
    });

    const copyToOffline = (src: AudioBuffer): AudioBuffer => {
      const buf = ctx.createBuffer(src.numberOfChannels, src.length, src.sampleRate);
      for (let c = 0; c < src.numberOfChannels; c += 1) {
        buf.getChannelData(c).set(src.getChannelData(c));
      }
      return buf;
    };

    const buffers = await Promise.all(
      clips.map(async (clip) => {
        const buf = await this.decodeSample(clip);
        return buf ? copyToOffline(buf) : null;
      }),
    );

    clips.forEach((clip, index) => {
      const buffer = buffers[index];
      if (!buffer) return;
      const source = ctx.createBufferSource();
      source.buffer = buffer;

      // Per-clip FX
      let tail: AudioNode = source;
      const fx = this.buildTrackEffectsChainWithContext(ctx, (clip.effects as TrackEffects) || ({} as TrackEffects));
      fx.forEach((stage) => {
        tail.connect(stage.inlet);
        tail = stage.outlet;
      });

      // Channel mix
      const mix = clip.channelId ? mixMap.get(clip.channelId) : undefined;
      if (mix) {
        const g = ctx.createGain();
        g.gain.value = mix.volume;
        const p = ctx.createStereoPanner();
        p.pan.value = mix.pan;
        tail.connect(g);
        g.connect(p);
        tail = p;
      }

      tail.connect(master.input);

      const startOffset = clip.startOffset ?? 0;
      const startTime = Math.max(0, clip.position ?? 0);
      const playDur = clip.duration ?? clip.length ?? buffer.duration - startOffset;
      if (Number.isFinite(playDur)) {
        source.start(startTime, startOffset, playDur);
      } else {
        source.start(startTime, startOffset);
      }
    });

    master.output.connect(ctx.destination);
    const rendered = await ctx.startRendering();
    return this.encodeWav(rendered, 16);
  }

  private buildOfflineMasterChain(
    ctx: BaseAudioContext,
    settings: MasteringSettings,
  ): { input: AudioNode; output: AudioNode } {
    const splitter = ctx.createChannelSplitter(2);
    const merger = ctx.createChannelMerger(2);
    const midLeft = ctx.createGain();
    const midRight = ctx.createGain();
    const sideLeft = ctx.createGain();
    const sideRight = ctx.createGain();
    const sideScale = ctx.createGain();
    const leftOut = ctx.createGain();
    const rightOut = ctx.createGain();

    midLeft.gain.value = 0.5;
    midRight.gain.value = 0.5;
    sideLeft.gain.value = 0.5;
    sideRight.gain.value = -0.5;
    sideScale.gain.value = Math.max(0, Math.min(1, settings.widenStereo ?? 0));

    splitter.connect(midLeft, 0);
    splitter.connect(midRight, 1);
    midLeft.connect(leftOut);
    midRight.connect(leftOut);
    midLeft.connect(rightOut);
    midRight.connect(rightOut);
    splitter.connect(sideLeft, 0);
    splitter.connect(sideRight, 1);
    sideLeft.connect(sideScale);
    sideRight.connect(sideScale);
    sideScale.connect(leftOut);
    const sideInvert = ctx.createGain();
    sideInvert.gain.value = -1;
    sideScale.connect(sideInvert);
    sideInvert.connect(rightOut);
    leftOut.connect(merger, 0, 0);
    rightOut.connect(merger, 0, 1);

    const glue = ctx.createDynamicsCompressor();
    const tiltLow = ctx.createBiquadFilter();
    const tiltHigh = ctx.createBiquadFilter();
    const tapeWet = ctx.createGain();
    const tapeDry = ctx.createGain();
    const tape = ctx.createWaveShaper();
    const limiter = ctx.createDynamicsCompressor();

    const glueAmt = Math.max(0, Math.min(1, settings.glueCompression ?? 0));
    glue.threshold.value = -30 + glueAmt * 20;
    glue.ratio.value = 1.2 + glueAmt * 3;
    glue.attack.value = 0.005 + glueAmt * 0.040;
    glue.knee.value = 6;
    glue.release.value = 0.150;

    tiltLow.type = "lowshelf";
    tiltLow.frequency.value = 220;
    tiltHigh.type = "highshelf";
    tiltHigh.frequency.value = 3500;
    const tilt = Math.max(0, Math.min(1, settings.spectralTilt ?? 0));
    const dB = (tilt - 0.5) * 9;
    tiltLow.gain.value = -dB;
    tiltHigh.gain.value = dB;

    const tapeAmt = Math.max(0, Math.min(1, settings.tapeSaturation ?? 0));
    tapeWet.gain.value = tapeAmt;
    tapeDry.gain.value = 1 - tapeAmt * 0.6;
    tape.curve = this.makeSaturationCurve(tapeAmt);

    limiter.threshold.value = settings.limiterCeiling ?? -0.3;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.050;

    merger.connect(glue);
    glue.connect(tiltLow);
    tiltLow.connect(tiltHigh);
    const sum = ctx.createGain();
    tiltHigh.connect(tapeDry);
    tiltHigh.connect(tape);
    tape.connect(tapeWet);
    tapeDry.connect(sum);
    tapeWet.connect(sum);
    sum.connect(limiter);

    return { input: splitter, output: limiter };
  }

  private buildTrackEffectsChainWithContext(
    ctx: BaseAudioContext,
    effects: TrackEffects,
  ): Array<{ inlet: AudioNode; outlet: AudioNode }> {
    const stages: Array<{ inlet: AudioNode; outlet: AudioNode }> = [];
    const makeDryWetStage = (wetNode: AudioNode, mix: number) => {
      const inlet = ctx.createGain();
      const dryGain = ctx.createGain();
      const wetGain = ctx.createGain();
      const sum = ctx.createGain();
      dryGain.gain.value = Math.max(0, Math.min(1, 1 - mix));
      wetGain.gain.value = Math.max(0, Math.min(1, mix));
      inlet.connect(dryGain);
      inlet.connect(wetNode);
      wetNode.connect(wetGain);
      dryGain.connect(sum);
      wetGain.connect(sum);
      return { inlet, outlet: sum } as const;
    };

    if (effects?.hiPass?.enabled) {
      const filter = ctx.createBiquadFilter();
      filter.type = "highpass";
      filter.frequency.value = effects.hiPass.cutoff;
      filter.Q.value = Math.max(0.0001, effects.hiPass.resonance);
      stages.push(makeDryWetStage(filter, effects.hiPass.mix));
    }
    if (effects?.lowPass?.enabled) {
      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.value = effects.lowPass.cutoff;
      filter.Q.value = Math.max(0.0001, effects.lowPass.resonance);
      stages.push(makeDryWetStage(filter, effects.lowPass.mix));
    }
    if (effects?.chorus?.enabled) {
      const delay = ctx.createDelay(0.05);
      delay.delayTime.value = 0.012;
      const lfo = ctx.createOscillator();
      const lfoDepth = ctx.createGain();
      lfo.type = "sine";
      lfo.frequency.value = Math.max(0.05, effects.chorus.rate);
      lfoDepth.gain.value = Math.max(0.0005, Math.min(0.01, effects.chorus.depth * 0.01));
      lfo.connect(lfoDepth);
      lfoDepth.connect(delay.delayTime);
      try { lfo.start(); } catch {}
      stages.push(makeDryWetStage(delay, effects.chorus.mix));
    }
    if (effects?.delay?.enabled) {
      const delay = ctx.createDelay(2);
      const feedback = ctx.createGain();
      const wetGain = ctx.createGain();
      const beat = 60 / this.bpm;
      const map: Record<'1/8' | '1/4' | '1/2' | '1', number> = {
        "1/8": beat / 2,
        "1/4": beat,
        "1/2": beat * 2,
        "1": beat * 4,
      };
      delay.delayTime.value = map[effects.delay.time] ?? beat;
      feedback.gain.value = Math.max(0, Math.min(0.95, effects.delay.feedback));
      wetGain.gain.value = Math.max(0, Math.min(1, effects.delay.mix));
      const inlet = ctx.createGain();
      inlet.connect(delay);
      delay.connect(feedback);
      feedback.connect(delay);
      delay.connect(wetGain);
      const sum = ctx.createGain();
      inlet.connect(sum);
      wetGain.connect(sum);
      stages.push({ inlet, outlet: sum });
    }
    if (effects?.reverb?.enabled) {
      const convolver = ctx.createConvolver();
      const length = Math.floor(ctx.sampleRate * (0.5 + effects.reverb.size * 2.0));
      const impulse = ctx.createBuffer(2, length, ctx.sampleRate);
      for (let c = 0; c < 2; c += 1) {
        const data = impulse.getChannelData(c);
        for (let i = 0; i < length; i += 1) {
          const t = i / length;
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2 + effects.reverb.decay * 6);
        }
      }
      convolver.buffer = impulse;
      stages.push(makeDryWetStage(convolver, effects.reverb.mix));
    }
    if (effects?.bitcrush?.enabled) {
      const shaper = ctx.createWaveShaper();
      const steps = Math.max(2, Math.min(65536, 2 ** Math.round(effects.bitcrush.depth)));
      const n = 512;
      const curve = new Float32Array(n);
      for (let i = 0; i < n; i += 1) {
        const x = (i / (n - 1)) * 2 - 1;
        const q = Math.round(((x + 1) / 2) * (steps - 1)) / (steps - 1);
        curve[i] = q * 2 - 1;
      }
      shaper.curve = curve;
      stages.push(makeDryWetStage(shaper, effects.bitcrush.mix));
    }
    return stages;
  }

  private encodeWav(buffer: AudioBuffer, bitDepth = 16): Blob {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const numFrames = buffer.length;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataBytes = numFrames * blockAlign;
    const headerBytes = 44;
    const totalBytes = headerBytes + dataBytes;
    const arrayBuffer = new ArrayBuffer(totalBytes);
    const view = new DataView(arrayBuffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i += 1) view.setUint8(offset + i, str.charCodeAt(i));
    };
    let offset = 0;
    writeString(offset, "RIFF"); offset += 4;
    view.setUint32(offset, totalBytes - 8, true); offset += 4;
    writeString(offset, "WAVE"); offset += 4;
    writeString(offset, "fmt "); offset += 4;
    view.setUint32(offset, 16, true); offset += 4;
    view.setUint16(offset, 1, true); offset += 2;
    view.setUint16(offset, numChannels, true); offset += 2;
    view.setUint32(offset, sampleRate, true); offset += 4;
    view.setUint32(offset, byteRate, true); offset += 4;
    view.setUint16(offset, blockAlign, true); offset += 2;
    view.setUint16(offset, bitDepth, true); offset += 2;
    writeString(offset, "data"); offset += 4;
    view.setUint32(offset, dataBytes, true); offset += 4;

    const channels: Float32Array[] = [];
    for (let c = 0; c < numChannels; c += 1) channels.push(buffer.getChannelData(c));
    let idx = 0;
    if (bitDepth === 16) {
      for (let i = 0; i < numFrames; i += 1) {
        for (let c = 0; c < numChannels; c += 1) {
          const sample = Math.max(-1, Math.min(1, channels[c][i] ?? 0));
          const s = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
          view.setInt16(offset + idx, s as number, true);
          idx += 2;
        }
      }
    } else {
      for (let i = 0; i < numFrames; i += 1) {
        for (let c = 0; c < numChannels; c += 1) {
          view.setFloat32(offset + idx, channels[c][i] ?? 0, true);
          idx += 4;
        }
      }
    }
    return new Blob([view.buffer], { type: "audio/wav" });
  }
}

export const audioEngine = new AudioEngine();
