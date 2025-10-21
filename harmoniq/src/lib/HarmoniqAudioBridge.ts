import type {
  DeckAudioSource,
  DeckFxId,
  DeckFxParams,
  DeckId,
  DeckPlaybackDiagnostics,
  EqBandId,
  StemType,
} from "../types";

const FX_ORDER: DeckFxId[] = ["rhythmicGate", "stutter", "glitch", "crush", "phaser", "reverb"];

interface EffectStage {
  input: GainNode;
  output: AudioNode;
  update(params: DeckFxParams): void;
  dispose(): void;
}

interface StemBand {
  filter: BiquadFilterNode;
  gain: GainNode;
}

interface DeckNodes {
  input: GainNode;
  eq: Record<EqBandId, BiquadFilterNode>;
  stems: {
    low: StemBand;
    mid: StemBand;
    high: StemBand;
    mix: GainNode;
  };
  captureTap: GainNode;
  captureMonitor: GainNode;
  level: GainNode;
  effects: Map<DeckFxId, EffectStage>;
  meterTap: GainNode;
  analyser: AnalyserNode;
  meterData: MeterArray;
}

type MeterArray = Float32Array<ArrayBuffer>;

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function numeric(value: number | string | undefined, fallback: number) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function clampNumber(value: number | string | undefined, min: number, max: number, fallback: number) {
  return clamp(numeric(value, fallback), min, max);
}

const RATE_TO_HZ: Record<string, number> = {
  "1/4": 1.5,
  "1/8": 3,
  "1/16": 6,
  "1/32": 8,
};

function rateToHz(value: number | string | undefined) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return clamp(value, 0.1, 16);
  }
  if (typeof value === "string") {
    const mapped = RATE_TO_HZ[value];
    if (mapped) {
      return mapped;
    }
  }
  return 3;
}

const DIVISION_TO_SECONDS: Record<string, number> = {
  "1": 2,
  "1/2": 1,
  "1/4": 0.5,
  "1/8": 0.25,
  "1/16": 0.125,
};

function divisionToSeconds(value: number | string | undefined) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return clamp(value, 0.02, 2);
  }
  if (typeof value === "string") {
    const mapped = DIVISION_TO_SECONDS[value];
    if (mapped) {
      return mapped;
    }
  }
  return 0.25;
}

function createReverbStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wet = context.createGain();
  const convolver = context.createConvolver();
  const sum = context.createGain();
  input.connect(dry);
  input.connect(convolver);
  convolver.connect(wet);
  dry.connect(sum);
  wet.connect(sum);

  let lastSize: number | null = null;
  let lastDecay: number | null = null;

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.6);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.08);
    wet.gain.setTargetAtTime(mix, context.currentTime, 0.08);
    const size = clampNumber(params.size, 0, 1, 0.6);
    const decay = clampNumber(params.decay, 0, 1, 0.7);
    if (size !== lastSize || decay !== lastDecay || !convolver.buffer) {
      const length = Math.max(1, Math.floor(context.sampleRate * (0.3 + size * 2.4)));
      const impulse = context.createBuffer(2, length, context.sampleRate);
      for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
        const data = impulse.getChannelData(channel);
        for (let i = 0; i < length; i += 1) {
          const t = i / length;
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2 + decay * 6);
        }
      }
      convolver.buffer = impulse;
      lastSize = size;
      lastDecay = decay;
    }
  };

  const dispose = () => {
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wet.disconnect();
    } catch {}
    try {
      convolver.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
    convolver.buffer = null;
  };

  return { input, output: sum, update, dispose };
}

function createGateStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wetIn = context.createGain();
  const gateGain = context.createGain();
  const wetOut = context.createGain();
  const sum = context.createGain();

  input.connect(dry);
  input.connect(wetIn);
  wetIn.connect(gateGain);
  gateGain.connect(wetOut);
  dry.connect(sum);
  wetOut.connect(sum);

  const offset = context.createConstantSource();
  offset.offset.value = 0.5;
  const depth = context.createGain();
  depth.gain.value = 0.5;
  const lfo = context.createOscillator();
  lfo.type = "square";
  lfo.frequency.value = 3;
  lfo.connect(depth);
  depth.connect(gateGain.gain);
  offset.connect(gateGain.gain);
  offset.start();
  lfo.start();

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.8);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.05);
    wetOut.gain.setTargetAtTime(mix, context.currentTime, 0.05);
    const rate = rateToHz(params.rate);
    lfo.frequency.setTargetAtTime(rate, context.currentTime, 0.05);
  };

  const dispose = () => {
    try {
      lfo.stop();
    } catch {}
    try {
      offset.stop();
    } catch {}
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wetIn.disconnect();
    } catch {}
    try {
      gateGain.disconnect();
    } catch {}
    try {
      wetOut.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
  };

  return { input, output: sum, update, dispose };
}

function createStutterStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wetIn = context.createGain();
  const delay = context.createDelay(1);
  const feedback = context.createGain();
  const wetOut = context.createGain();
  const sum = context.createGain();

  input.connect(dry);
  input.connect(wetIn);
  wetIn.connect(delay);
  delay.connect(feedback);
  feedback.connect(delay);
  delay.connect(wetOut);
  dry.connect(sum);
  wetOut.connect(sum);

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.6);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.05);
    wetOut.gain.setTargetAtTime(mix, context.currentTime, 0.05);
    const base = divisionToSeconds(params.division);
    const jitter = clampNumber(params.jitter, 0, 1, 0.15);
    const jittered = clamp(base * (1 + (Math.random() - 0.5) * jitter * 0.6), 0.02, 0.6);
    delay.delayTime.setTargetAtTime(jittered, context.currentTime, 0.05);
    const feedbackAmount = clamp(0.2 + mix * 0.45, 0, 0.92);
    feedback.gain.setTargetAtTime(feedbackAmount, context.currentTime, 0.1);
  };

  const dispose = () => {
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wetIn.disconnect();
    } catch {}
    try {
      delay.disconnect();
    } catch {}
    try {
      feedback.disconnect();
    } catch {}
    try {
      wetOut.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
  };

  return { input, output: sum, update, dispose };
}

function createGlitchStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wetIn = context.createGain();
  const shaper = context.createWaveShaper();
  const wetOut = context.createGain();
  const sum = context.createGain();

  input.connect(dry);
  input.connect(wetIn);
  wetIn.connect(shaper);
  shaper.connect(wetOut);
  dry.connect(sum);
  wetOut.connect(sum);

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.7);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.05);
    wetOut.gain.setTargetAtTime(mix, context.currentTime, 0.05);
    const density = clampNumber(params.density, 0, 1, 0.5);
    const scatter = clampNumber(params.scatter, 0, 1, 0.35);
    const steps = Math.max(2, Math.round(4 + density * 24));
    const curve = new Float32Array(256);
    for (let i = 0; i < curve.length; i += 1) {
      const x = (i / (curve.length - 1)) * 2 - 1;
      const step = Math.round(((x + 1) / 2) * (steps - 1));
      const base = (step / (steps - 1)) * 2 - 1;
      const jitter = (Math.random() * 2 - 1) * scatter * 0.25;
      curve[i] = clamp(base + jitter, -1, 1);
    }
    shaper.curve = curve;
  };

  const dispose = () => {
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wetIn.disconnect();
    } catch {}
    try {
      shaper.disconnect();
    } catch {}
    try {
      wetOut.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
  };

  return { input, output: sum, update, dispose };
}

function createCrushStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wetIn = context.createGain();
  const shaper = context.createWaveShaper();
  const wetOut = context.createGain();
  const sum = context.createGain();

  input.connect(dry);
  input.connect(wetIn);
  wetIn.connect(shaper);
  shaper.connect(wetOut);
  dry.connect(sum);
  wetOut.connect(sum);

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.75);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.05);
    wetOut.gain.setTargetAtTime(mix, context.currentTime, 0.05);
    const depth = clampNumber(params.depth, 2, 12, 6);
    const downsample = clampNumber(params.downsample, 0, 1, 0.3);
    const steps = Math.max(2, Math.round(2 ** clamp(depth, 1, 8)));
    const curve = new Float32Array(256);
    for (let i = 0; i < curve.length; i += 1) {
      const x = (i / (curve.length - 1)) * 2 - 1;
      const step = Math.round(((x + 1) / 2) * (steps - 1));
      const quantized = (step / (steps - 1)) * 2 - 1;
      const jitter = (Math.random() * 2 - 1) * downsample * 0.1;
      curve[i] = clamp(quantized + jitter, -1, 1);
    }
    shaper.curve = curve;
  };

  const dispose = () => {
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wetIn.disconnect();
    } catch {}
    try {
      shaper.disconnect();
    } catch {}
    try {
      wetOut.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
  };

  return { input, output: sum, update, dispose };
}

function createPhaserStage(context: AudioContext): EffectStage {
  const input = context.createGain();
  const dry = context.createGain();
  const wetIn = context.createGain();
  const stageA = context.createBiquadFilter();
  stageA.type = "allpass";
  const stageB = context.createBiquadFilter();
  stageB.type = "allpass";
  const wetOut = context.createGain();
  const sum = context.createGain();

  input.connect(dry);
  input.connect(wetIn);
  wetIn.connect(stageA);
  stageA.connect(stageB);
  stageB.connect(wetOut);
  dry.connect(sum);
  wetOut.connect(sum);

  const lfo = context.createOscillator();
  lfo.type = "sine";
  lfo.frequency.value = 0.35;
  const depthA = context.createGain();
  const depthB = context.createGain();
  depthA.gain.value = 160;
  depthB.gain.value = 120;
  lfo.connect(depthA);
  lfo.connect(depthB);
  depthA.connect(stageA.frequency);
  depthB.connect(stageB.frequency);
  stageA.frequency.value = 420;
  stageB.frequency.value = 620;
  lfo.start();

  const update = (params: DeckFxParams) => {
    const mix = clampNumber(params.mix, 0, 1, 0.65);
    dry.gain.setTargetAtTime(1 - mix, context.currentTime, 0.05);
    wetOut.gain.setTargetAtTime(mix, context.currentTime, 0.05);
    const rate = clampNumber(params.rate, 0.05, 5, 0.35);
    lfo.frequency.setTargetAtTime(rate, context.currentTime, 0.05);
    const depth = clampNumber(params.depth, 0, 1, 0.6);
    const base = 320 + depth * 480;
    stageA.frequency.setTargetAtTime(base, context.currentTime, 0.1);
    stageB.frequency.setTargetAtTime(base * 1.3, context.currentTime, 0.1);
    depthA.gain.setTargetAtTime(base * 0.55, context.currentTime, 0.1);
    depthB.gain.setTargetAtTime(base * 0.4, context.currentTime, 0.1);
  };

  const dispose = () => {
    try {
      lfo.stop();
    } catch {}
    try {
      input.disconnect();
    } catch {}
    try {
      dry.disconnect();
    } catch {}
    try {
      wetIn.disconnect();
    } catch {}
    try {
      stageA.disconnect();
    } catch {}
    try {
      stageB.disconnect();
    } catch {}
    try {
      wetOut.disconnect();
    } catch {}
    try {
      sum.disconnect();
    } catch {}
  };

  return { input, output: sum, update, dispose };
}

function createEffectStage(effectId: DeckFxId, context: AudioContext): EffectStage {
  switch (effectId) {
    case "reverb":
      return createReverbStage(context);
    case "rhythmicGate":
      return createGateStage(context);
    case "stutter":
      return createStutterStage(context);
    case "glitch":
      return createGlitchStage(context);
    case "crush":
      return createCrushStage(context);
    case "phaser":
      return createPhaserStage(context);
    default: {
      const passthrough = context.createGain();
      return {
        input: passthrough,
        output: passthrough,
        update: () => {},
        dispose: () => {
          try {
            passthrough.disconnect();
          } catch {}
        },
      };
    }
  }
}

interface DeckPlaybackInternal {
  buffer: AudioBuffer | null;
  source: AudioBufferSourceNode | null;
  position: number;
  startedAt: number | null;
  isPlaying: boolean;
  durationSeconds: number | null;
  error: string | null;
  lastSource?: DeckAudioSource;
}

type DiagnosticsListener = (snapshot: DeckPlaybackDiagnostics) => void;

interface LoopCaptureSession {
  deckId: DeckId;
  processor: ScriptProcessorNode;
  monitor: GainNode;
  buffers: Float32Array[][];
  durationSeconds: number;
  resolve: (result: { buffer: AudioBuffer; durationSeconds: number }) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout> | null;
}

interface HarmoniqAudioBridgeOptions {
  context?: AudioContext;
  logger?: (...args: unknown[]) => void;
  debug?: boolean;
  meterIntervalMs?: number;
}

export class HarmoniqAudioBridge {
  private context: AudioContext | null = null;

  private master: GainNode | null = null;

  private decks = new Map<DeckId, DeckNodes>();

  private deckGains = new Map<DeckId, number>();

  private effectStates = new Map<DeckId, Map<DeckFxId, boolean>>();

  private eqStates = new Map<DeckId, Record<EqBandId, boolean>>();

  private stemStates = new Map<DeckId, StemType | null>();

  private masterTrim = 0.9;

  private smoothing = 0.08;

  private readonly injectedContext: AudioContext | null;

  private readonly ownsContext: boolean;

  private readonly logger: (...args: unknown[]) => void;

  private readonly debugEnabled: boolean;

  private readonly meterIntervalMs: number;

  private meterInterval: ReturnType<typeof setInterval> | null = null;

  private playback = new Map<DeckId, DeckPlaybackInternal>();

  private diagnosticsListeners = new Set<DiagnosticsListener>();

  private loopCaptures = new Map<DeckId, LoopCaptureSession>();

  constructor(options: HarmoniqAudioBridgeOptions = {}) {
    this.injectedContext = options.context ?? null;
    this.ownsContext = !this.injectedContext;
    this.logger = options.logger ?? ((...args: unknown[]) => {
      if (typeof console !== "undefined" && typeof console.debug === "function") {
        console.debug(...args);
      }
    });
    const globalDebugFlag =
      typeof window !== "undefined" && typeof (window as { __HARMONIQ_DEBUG__?: boolean }).__HARMONIQ_DEBUG__ === "boolean"
        ? (window as { __HARMONIQ_DEBUG__?: boolean }).__HARMONIQ_DEBUG__
        : undefined;
    this.debugEnabled = options.debug ?? (globalDebugFlag ?? true);
    this.meterIntervalMs = options.meterIntervalMs ?? 120;
    if (this.injectedContext) {
      this.context = this.injectedContext;
      this.ensureMaster(this.injectedContext);
    }
  }

  private debug(message: string, ...args: unknown[]) {
    if (!this.debugEnabled) {
      return;
    }
    this.logger(`[HarmoniqAudioBridge] ${message}`, ...args);
  }

  private getContext(): AudioContext | null {
    if (this.injectedContext) {
      return this.injectedContext;
    }
    if (this.context) {
      return this.context;
    }
    if (typeof window === "undefined") {
      return null;
    }
    const ctor = (window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext) as
      | typeof AudioContext
      | undefined;
    if (!ctor) {
      return null;
    }
    const context = new ctor();
    this.context = context;
    this.ensureMaster(context);
    this.debug("created new AudioContext instance");
    return context;
  }

  private ensureMaster(context: AudioContext) {
    if (this.master) {
      return this.master;
    }
    const master = context.createGain();
    master.gain.value = this.masterTrim;
    master.connect(context.destination);
    this.master = master;
    return master;
  }

  private ensureDeck(context: AudioContext, deckId: DeckId): DeckNodes {
    let deck = this.decks.get(deckId);
    if (deck) {
      return deck;
    }
    const input = context.createGain();
    const eqLows = context.createBiquadFilter();
    eqLows.type = "highpass";
    eqLows.frequency.value = 24;
    eqLows.Q.value = 0.707;
    const eqMids = context.createBiquadFilter();
    eqMids.type = "peaking";
    eqMids.frequency.value = 1200;
    eqMids.Q.value = 0.9;
    eqMids.gain.value = 0;
    const eqHighs = context.createBiquadFilter();
    eqHighs.type = "lowpass";
    eqHighs.frequency.value = 18000;
    eqHighs.Q.value = 0.6;
    const stemLowFilter = context.createBiquadFilter();
    stemLowFilter.type = "lowpass";
    stemLowFilter.frequency.value = 220;
    stemLowFilter.Q.value = 0.9;
    const stemLowGain = context.createGain();
    stemLowGain.gain.value = 1;
    const stemMidFilter = context.createBiquadFilter();
    stemMidFilter.type = "bandpass";
    stemMidFilter.frequency.value = 1200;
    stemMidFilter.Q.value = 1.2;
    const stemMidGain = context.createGain();
    stemMidGain.gain.value = 1;
    const stemHighFilter = context.createBiquadFilter();
    stemHighFilter.type = "highpass";
    stemHighFilter.frequency.value = 3200;
    stemHighFilter.Q.value = 0.8;
    const stemHighGain = context.createGain();
    stemHighGain.gain.value = 1;
    const stemMix = context.createGain();
    stemMix.gain.value = 1;
    const level = context.createGain();
    level.gain.value = 0;
    const master = this.ensureMaster(context);
    const captureTap = context.createGain();
    captureTap.gain.value = 1;
    const captureMonitor = context.createGain();
    captureMonitor.gain.value = 0;
    const meterTap = context.createGain();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.85;
    meterTap.connect(level);
    meterTap.connect(analyser);
    level.connect(master);
    captureMonitor.connect(master);
    const meterData = new Float32Array(analyser.fftSize) as MeterArray;
    deck = {
      input,
      eq: { lows: eqLows, mids: eqMids, highs: eqHighs },
      stems: {
        low: { filter: stemLowFilter, gain: stemLowGain },
        mid: { filter: stemMidFilter, gain: stemMidGain },
        high: { filter: stemHighFilter, gain: stemHighGain },
        mix: stemMix,
      },
      captureTap,
      captureMonitor,
      level,
      effects: new Map(),
      meterTap,
      analyser,
      meterData,
    };
    this.decks.set(deckId, deck);
    this.effectStates.set(deckId, new Map());
    this.eqStates.set(deckId, { highs: false, mids: false, lows: false });
    this.stemStates.set(deckId, null);
    this.rebuildDeckChain(deckId);
    return deck;
  }

  setDeckBlend(deckId: DeckId, gain: number) {
    const context = this.getContext();
    if (!context) return;
    const deck = this.ensureDeck(context, deckId);
    const clamped = clamp(gain, 0, 1);
    const previous = this.deckGains.get(deckId);
    if (previous === clamped) {
      return;
    }
    deck.level.gain.setTargetAtTime(clamped, context.currentTime, this.smoothing);
    this.deckGains.set(deckId, clamped);
  }

  setMasterTrim(value: number) {
    const context = this.getContext();
    if (!context) return;
    const clamped = clamp(value, 0.2, 2);
    this.masterTrim = clamped;
    const master = this.ensureMaster(context);
    master.gain.setTargetAtTime(clamped, context.currentTime, 0.1);
  }

  setEffectState(deckId: DeckId, effectId: DeckFxId, enabled: boolean, params: DeckFxParams = {}) {
    const context = this.getContext();
    if (!context) return;
    const deck = this.ensureDeck(context, deckId);
    const state = this.effectStates.get(deckId) ?? new Map<DeckFxId, boolean>();
    const previous = state.get(effectId);

    if (!enabled) {
      if (previous) {
        const existing = deck.effects.get(effectId);
        if (existing) {
          existing.dispose();
          deck.effects.delete(effectId);
        }
        state.set(effectId, false);
        this.effectStates.set(deckId, state);
        this.rebuildDeckChain(deckId);
      }
      return;
    }

    let stage = deck.effects.get(effectId);
    if (!stage) {
      stage = createEffectStage(effectId, context);
      deck.effects.set(effectId, stage);
    }
    stage.update(params);
    if (!previous) {
      state.set(effectId, true);
      this.effectStates.set(deckId, state);
      this.rebuildDeckChain(deckId);
    }
  }

  setEqCut(deckId: DeckId, band: EqBandId, enabled: boolean) {
    const context = this.getContext();
    if (!context) return;
    const deck = this.ensureDeck(context, deckId);
    const states = this.eqStates.get(deckId) ?? { highs: false, mids: false, lows: false };
    if (states[band] === enabled) {
      return;
    }
    const now = context.currentTime;
    switch (band) {
      case "lows":
        deck.eq.lows.frequency.setTargetAtTime(enabled ? 180 : 24, now, 0.08);
        deck.eq.lows.Q.setTargetAtTime(enabled ? 0.95 : 0.707, now, 0.08);
        break;
      case "mids":
        deck.eq.mids.gain.setTargetAtTime(enabled ? -9 : 0, now, 0.06);
        deck.eq.mids.frequency.setTargetAtTime(enabled ? 1100 : 1200, now, 0.1);
        deck.eq.mids.Q.setTargetAtTime(enabled ? 1.35 : 0.9, now, 0.1);
        break;
      case "highs":
        deck.eq.highs.frequency.setTargetAtTime(enabled ? 5400 : 18000, now, 0.1);
        deck.eq.highs.Q.setTargetAtTime(enabled ? 0.9 : 0.6, now, 0.1);
        break;
      default:
        break;
    }
    states[band] = enabled;
    this.eqStates.set(deckId, states);
  }

  setStemProfile(deckId: DeckId, stem: StemType | null) {
    const context = this.getContext();
    if (!context) return;
    const deck = this.ensureDeck(context, deckId);
    const current = this.stemStates.get(deckId) ?? null;
    if (current === stem) {
      return;
    }
    const now = context.currentTime;
    const profile = stem
      ? stem === "drums"
        ? { low: 1.15, mid: 0.35, high: 0.2, lowFreq: 280, midFreq: 980, highFreq: 3600 }
        : stem === "synths"
        ? { low: 0.35, mid: 1.1, high: 0.55, lowFreq: 200, midFreq: 1500, highFreq: 4200 }
        : { low: 0.25, mid: 0.5, high: 1.1, lowFreq: 220, midFreq: 1700, highFreq: 5200 }
      : { low: 1, mid: 1, high: 1, lowFreq: 220, midFreq: 1200, highFreq: 3200 };
    deck.stems.low.gain.gain.setTargetAtTime(profile.low, now, 0.08);
    deck.stems.mid.gain.gain.setTargetAtTime(profile.mid, now, 0.08);
    deck.stems.high.gain.gain.setTargetAtTime(profile.high, now, 0.08);
    deck.stems.low.filter.frequency.setTargetAtTime(profile.lowFreq, now, 0.1);
    deck.stems.mid.filter.frequency.setTargetAtTime(profile.midFreq, now, 0.1);
    deck.stems.high.filter.frequency.setTargetAtTime(profile.highFreq, now, 0.1);
    this.stemStates.set(deckId, stem ?? null);
  }

  startLoopCapture(deckId: DeckId, durationSeconds: number): Promise<{ buffer: AudioBuffer; durationSeconds: number }> {
    const context = this.getContext();
    if (!context) {
      return Promise.reject(new Error("Audio engine unavailable"));
    }
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      return Promise.reject(new Error("Capture duration must be positive"));
    }
    this.ensureDeck(context, deckId);
    const existing = this.loopCaptures.get(deckId);
    if (existing) {
      this.debug(`cancelling existing loop capture for deck ${deckId}`);
      this.cancelLoopCapture(deckId);
    }
    return new Promise((resolve, reject) => {
      try {
        const deck = this.ensureDeck(context, deckId);
        const processor = context.createScriptProcessor(4096, 2, 2);
        const buffers: Float32Array[][] = [[], []];
        processor.onaudioprocess = (event) => {
          const input = event.inputBuffer;
          const channelCount = Math.min(2, input.numberOfChannels);
          for (let channel = 0; channel < channelCount; channel += 1) {
            const chunk = new Float32Array(input.length);
            input.copyFromChannel(chunk, channel);
            buffers[channel].push(chunk);
          }
          for (let channel = channelCount; channel < buffers.length; channel += 1) {
            if (!buffers[channel]) {
              buffers[channel] = [];
            }
          }
        };
        deck.captureTap.connect(processor);
        processor.connect(deck.captureMonitor);
        const session: LoopCaptureSession = {
          deckId,
          processor,
          monitor: deck.captureMonitor,
          buffers,
          durationSeconds,
          resolve,
          reject,
          timer: null,
        };
        const delay = Math.max(10, durationSeconds * 1000);
        session.timer = setTimeout(() => {
          void this.finalizeLoopCapture(deckId).catch((error) => {
            session.reject(error instanceof Error ? error : new Error(String(error)));
          });
        }, delay);
        this.loopCaptures.set(deckId, session);
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  cancelLoopCapture(deckId: DeckId) {
    void this.finalizeLoopCapture(deckId, true).catch((error) => {
      this.debug(`error while cancelling loop capture for deck ${deckId}`, error);
    });
  }

  private async finalizeLoopCapture(deckId: DeckId, cancelled = false): Promise<void> {
    const session = this.loopCaptures.get(deckId);
    if (!session) {
      return;
    }
    if (session.timer) {
      clearTimeout(session.timer);
      session.timer = null;
    }
    this.loopCaptures.delete(deckId);
    try {
      session.processor.onaudioprocess = null;
    } catch {}
    try {
      session.processor.disconnect();
    } catch {}
    const deck = this.decks.get(deckId);
    if (deck) {
      try {
        deck.captureTap.disconnect(session.processor);
      } catch {}
    }
    if (cancelled) {
      session.reject(new Error("Loop capture cancelled"));
      return;
    }
    try {
      const context = this.getContext();
      if (!context) {
        throw new Error("Audio engine unavailable");
      }
      const channelCount = Math.max(1, session.buffers.length);
      const totalSamplesPerChannel = session.buffers.map((chunks) =>
        chunks.reduce((sum, chunk) => sum + chunk.length, 0),
      );
      const expectedFrames = Math.max(1, Math.floor(session.durationSeconds * context.sampleRate));
      const availableSamples = totalSamplesPerChannel.reduce(
        (min, value) => Math.min(min, value || 0),
        Number.POSITIVE_INFINITY,
      );
      const usableSamples = Number.isFinite(availableSamples)
        ? Math.max(1, availableSamples)
        : expectedFrames;
      const frameCount = Math.max(1, Math.min(expectedFrames, usableSamples));
      const result = context.createBuffer(channelCount, frameCount, context.sampleRate);
      for (let channel = 0; channel < channelCount; channel += 1) {
        const channelData = result.getChannelData(channel);
        const chunks = session.buffers[channel] ?? [];
        let offset = 0;
        for (const chunk of chunks) {
          const remaining = frameCount - offset;
          if (remaining <= 0) {
            break;
          }
          const toCopy = Math.min(remaining, chunk.length);
          channelData.set(chunk.subarray(0, toCopy), offset);
          offset += toCopy;
        }
      }
      session.resolve({ buffer: result, durationSeconds: frameCount / context.sampleRate });
    } catch (error) {
      session.reject(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private rebuildDeckChain(deckId: DeckId) {
    const deck = this.decks.get(deckId);
    if (!deck) return;
    try {
      deck.input.disconnect();
    } catch {}
    try {
      deck.eq.lows.disconnect();
    } catch {}
    try {
      deck.eq.mids.disconnect();
    } catch {}
    try {
      deck.eq.highs.disconnect();
    } catch {}
    try {
      deck.stems.low.filter.disconnect();
    } catch {}
    try {
      deck.stems.mid.filter.disconnect();
    } catch {}
    try {
      deck.stems.high.filter.disconnect();
    } catch {}
    try {
      deck.stems.low.gain.disconnect();
    } catch {}
    try {
      deck.stems.mid.gain.disconnect();
    } catch {}
    try {
      deck.stems.high.gain.disconnect();
    } catch {}
    try {
      deck.stems.mix.disconnect();
    } catch {}
    try {
      deck.captureTap.disconnect();
    } catch {}
    deck.effects.forEach((stage) => {
      try {
        stage.input.disconnect();
      } catch {}
      try {
        stage.output.disconnect();
      } catch {}
    });
    deck.input.connect(deck.eq.lows);
    deck.eq.lows.connect(deck.eq.mids);
    deck.eq.mids.connect(deck.eq.highs);
    deck.eq.highs.connect(deck.stems.low.filter);
    deck.eq.highs.connect(deck.stems.mid.filter);
    deck.eq.highs.connect(deck.stems.high.filter);
    deck.stems.low.filter.connect(deck.stems.low.gain);
    deck.stems.mid.filter.connect(deck.stems.mid.gain);
    deck.stems.high.filter.connect(deck.stems.high.gain);
    deck.stems.low.gain.connect(deck.stems.mix);
    deck.stems.mid.gain.connect(deck.stems.mix);
    deck.stems.high.gain.connect(deck.stems.mix);
    let previous: AudioNode = deck.stems.mix;
    for (const effectId of FX_ORDER) {
      const stage = deck.effects.get(effectId);
      if (!stage) continue;
      previous.connect(stage.input);
      previous = stage.output;
    }
    previous.connect(deck.captureTap);
    deck.captureTap.connect(deck.captureMonitor);
    deck.captureTap.connect(deck.meterTap);
    const capture = this.loopCaptures.get(deckId);
    if (capture) {
      try {
        deck.captureTap.connect(capture.processor);
      } catch {}
    }
  }

  private ensurePlayback(deckId: DeckId): DeckPlaybackInternal {
    let playback = this.playback.get(deckId);
    if (!playback) {
      playback = {
        buffer: null,
        source: null,
        position: 0,
        startedAt: null,
        isPlaying: false,
        durationSeconds: null,
        error: null,
      };
      this.playback.set(deckId, playback);
    }
    return playback;
  }

  private getActiveContext(): AudioContext | null {
    return this.context ?? this.injectedContext;
  }

  private computeCurrentTime(playback: DeckPlaybackInternal, context: AudioContext | null): number {
    const duration = playback.durationSeconds ?? playback.buffer?.duration ?? null;
    const clampedPosition = clamp(playback.position, 0, duration ?? Number.MAX_SAFE_INTEGER);
    if (!playback.isPlaying || playback.startedAt === null || !context) {
      return duration ? clamp(clampedPosition, 0, duration) : clampedPosition;
    }
    const elapsed = Math.max(0, context.currentTime - playback.startedAt);
    const current = clampedPosition + elapsed;
    return duration ? clamp(current, 0, duration) : current;
  }

  private stopPlayback(deckId: DeckId, playback: DeckPlaybackInternal, preservePosition = false) {
    const context = this.getActiveContext();
    if (!preservePosition) {
      const current = this.computeCurrentTime(playback, context);
      const duration = playback.durationSeconds ?? playback.buffer?.duration ?? null;
      playback.position = duration ? clamp(current, 0, duration) : current;
    }
    const source = playback.source;
    if (source) {
      try {
        source.onended = null;
      } catch {}
      try {
        source.stop();
      } catch {}
      try {
        source.disconnect();
      } catch {}
    }
    playback.source = null;
    playback.isPlaying = false;
    playback.startedAt = null;
    this.debug(`stopped deck ${deckId}`);
  }

  private async resolveArrayBuffer(source: DeckAudioSource): Promise<ArrayBuffer> {
    if (source.arrayBuffer) {
      return source.arrayBuffer.slice(0);
    }
    if (source.file) {
      return source.file.arrayBuffer();
    }
    if (source.objectUrl) {
      const response = await fetch(source.objectUrl);
      if (!response.ok) {
        throw new Error(`Failed to load audio (${response.status})`);
      }
      return response.arrayBuffer();
    }
    throw new Error("No audio source provided for deck");
  }

  private async decodeSource(context: AudioContext, source: DeckAudioSource): Promise<AudioBuffer> {
    const data = await this.resolveArrayBuffer(source);
    const bufferCopy = data.slice(0);
    return context.decodeAudioData(bufferCopy);
  }

  private emitDiagnostics(deckId: DeckId) {
    if (!this.diagnosticsListeners.size) {
      return;
    }
    const playback = this.playback.get(deckId);
    const deck = this.decks.get(deckId);
    const context = this.getActiveContext();
    if (!playback || !deck) {
      return;
    }
    let vu = 0;
    if (deck.analyser) {
      try {
        deck.analyser.getFloatTimeDomainData(deck.meterData);
        let sum = 0;
        for (let i = 0; i < deck.meterData.length; i += 1) {
          const sample = deck.meterData[i];
          sum += sample * sample;
        }
        const rms = Math.sqrt(sum / deck.meterData.length);
        vu = clamp(rms * 1.4, 0, 1);
      } catch {
        vu = 0;
      }
    }
    const snapshot: DeckPlaybackDiagnostics = {
      deckId,
      isPlaying: playback.isPlaying,
      currentTimeSeconds: this.computeCurrentTime(playback, context),
      durationSeconds: playback.durationSeconds,
      vu,
      error: playback.error,
    };
    this.diagnosticsListeners.forEach((listener) => listener(snapshot));
  }

  private ensureMetering() {
    if (this.meterInterval || !this.diagnosticsListeners.size) {
      return;
    }
    this.meterInterval = setInterval(() => this.updateAllDiagnostics(), this.meterIntervalMs);
  }

  private updateAllDiagnostics() {
    if (!this.diagnosticsListeners.size) {
      return;
    }
    this.playback.forEach((_, deckId) => this.emitDiagnostics(deckId));
  }

  private teardownMetering() {
    if (this.meterInterval) {
      clearInterval(this.meterInterval);
      this.meterInterval = null;
    }
  }

  async analyzeSource(source: DeckAudioSource): Promise<{ durationSeconds: number | null }> {
    const context = this.getContext();
    if (!context) {
      throw new Error("Audio engine unavailable");
    }
    const buffer = await this.decodeSource(context, source);
    this.debug(`analyzed source ${source.id ?? "unknown"}`, { duration: buffer.duration });
    return { durationSeconds: Number.isFinite(buffer.duration) ? buffer.duration : null };
  }

  async loadDeckAudio(deckId: DeckId, source: DeckAudioSource): Promise<{ durationSeconds: number | null }> {
    const context = this.getContext();
    if (!context) {
      throw new Error("Audio engine unavailable");
    }
    this.ensureDeck(context, deckId);
    const playback = this.ensurePlayback(deckId);
    try {
      const buffer = await this.decodeSource(context, source);
      this.stopPlayback(deckId, playback);
      playback.buffer = buffer;
      playback.position = 0;
      playback.startedAt = null;
      playback.isPlaying = false;
      playback.durationSeconds = Number.isFinite(buffer.duration) ? buffer.duration : null;
      playback.error = null;
      playback.lastSource = source;
      this.debug(`loaded deck ${deckId}`, {
        duration: buffer.duration,
        channels: buffer.numberOfChannels,
        sampleRate: buffer.sampleRate,
      });
      this.emitDiagnostics(deckId);
      return { durationSeconds: playback.durationSeconds };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      playback.error = message;
      this.emitDiagnostics(deckId);
      throw error;
    }
  }

  async playDeck(deckId: DeckId) {
    const context = this.getContext();
    if (!context) {
      throw new Error("Audio engine unavailable");
    }
    const deck = this.ensureDeck(context, deckId);
    const playback = this.ensurePlayback(deckId);
    if (!playback.buffer) {
      const message = "No audio loaded";
      playback.error = message;
      this.emitDiagnostics(deckId);
      throw new Error(message);
    }
    try {
      this.stopPlayback(deckId, playback, true);
      const offset = clamp(
        playback.position,
        0,
        playback.durationSeconds ?? playback.buffer.duration,
      );
      const source = context.createBufferSource();
      source.buffer = playback.buffer;
      source.connect(deck.input);
      source.onended = () => {
        const current = this.playback.get(deckId);
        if (!current || current.source !== source) {
          return;
        }
        current.isPlaying = false;
        current.startedAt = null;
        current.source = null;
        current.position = 0;
        this.emitDiagnostics(deckId);
      };
      playback.source = source;
      playback.startedAt = context.currentTime;
      playback.isPlaying = true;
      playback.error = null;
      this.debug(`play deck ${deckId}`, { offset });
      source.start(0, offset);
      this.emitDiagnostics(deckId);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      playback.error = message;
      this.emitDiagnostics(deckId);
      throw error;
    }
  }

  async stopDeck(deckId: DeckId) {
    const playback = this.ensurePlayback(deckId);
    if (!playback.source && !playback.isPlaying) {
      playback.error = null;
      return;
    }
    this.stopPlayback(deckId, playback);
    playback.error = null;
    this.emitDiagnostics(deckId);
  }

  async seekDeck(deckId: DeckId, seconds: number) {
    const context = this.getContext();
    if (!context) {
      throw new Error("Audio engine unavailable");
    }
    const playback = this.ensurePlayback(deckId);
    if (!playback.buffer) {
      const message = "No audio loaded";
      playback.error = message;
      this.emitDiagnostics(deckId);
      throw new Error(message);
    }
    const duration = playback.durationSeconds ?? playback.buffer.duration;
    const target = clamp(seconds, 0, duration);
    const wasPlaying = playback.isPlaying;
    this.stopPlayback(deckId, playback, true);
    playback.position = target;
    playback.startedAt = null;
    playback.error = null;
    if (wasPlaying) {
      await this.playDeck(deckId);
    } else {
      this.emitDiagnostics(deckId);
    }
  }

  async reloadDeck(deckId: DeckId, source?: DeckAudioSource) {
    const playback = this.ensurePlayback(deckId);
    const nextSource = source ?? playback.lastSource;
    if (!nextSource) {
      const message = "No source available to reload";
      playback.error = message;
      this.emitDiagnostics(deckId);
      throw new Error(message);
    }
    const wasPlaying = playback.isPlaying;
    const resumePosition = playback.position;
    const result = await this.loadDeckAudio(deckId, nextSource);
    const refreshed = this.ensurePlayback(deckId);
    const duration = refreshed.durationSeconds ?? refreshed.buffer?.duration ?? null;
    refreshed.position = duration ? clamp(resumePosition, 0, duration) : resumePosition;
    refreshed.startedAt = null;
    refreshed.error = null;
    if (wasPlaying) {
      await this.playDeck(deckId);
    } else {
      this.emitDiagnostics(deckId);
    }
    return result;
  }

  subscribeDiagnostics(listener: DiagnosticsListener): () => void {
    this.diagnosticsListeners.add(listener);
    this.ensureMetering();
    this.playback.forEach((_, deckId) => this.emitDiagnostics(deckId));
    return () => {
      this.diagnosticsListeners.delete(listener);
      if (!this.diagnosticsListeners.size) {
        this.teardownMetering();
      }
    };
  }

  dispose() {
    this.teardownMetering();
    this.loopCaptures.forEach((_, id) => {
      this.cancelLoopCapture(id);
    });
    this.loopCaptures.clear();
    this.playback.forEach((playback, deckId) => {
      this.stopPlayback(deckId, playback);
    });
    this.playback.clear();
    this.diagnosticsListeners.clear();
    this.decks.forEach((deck) => {
      deck.effects.forEach((stage) => {
        stage.dispose();
      });
      try {
        deck.input.disconnect();
      } catch {}
      try {
        deck.eq.lows.disconnect();
      } catch {}
      try {
        deck.eq.mids.disconnect();
      } catch {}
      try {
        deck.eq.highs.disconnect();
      } catch {}
      try {
        deck.stems.low.filter.disconnect();
      } catch {}
      try {
        deck.stems.mid.filter.disconnect();
      } catch {}
      try {
        deck.stems.high.filter.disconnect();
      } catch {}
      try {
        deck.stems.low.gain.disconnect();
      } catch {}
      try {
        deck.stems.mid.gain.disconnect();
      } catch {}
      try {
        deck.stems.high.gain.disconnect();
      } catch {}
      try {
        deck.stems.mix.disconnect();
      } catch {}
      try {
        deck.level.disconnect();
      } catch {}
      try {
        deck.captureTap.disconnect();
      } catch {}
      try {
        deck.captureMonitor.disconnect();
      } catch {}
      try {
        deck.meterTap.disconnect();
      } catch {}
    });
    this.decks.clear();
    this.deckGains.clear();
    this.effectStates.clear();
    this.eqStates.clear();
    this.stemStates.clear();
    if (this.master) {
      try {
        this.master.disconnect();
      } catch {}
      this.master = null;
    }
    if (this.context && this.ownsContext) {
      const ctx = this.context;
      this.context = null;
      void ctx.close().catch(() => undefined);
    }
  }
}
