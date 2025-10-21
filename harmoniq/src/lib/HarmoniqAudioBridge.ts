import type {
  DeckAudioSource,
  DeckFxId,
  DeckFxParams,
  DeckId,
  DeckPlaybackDiagnostics,
  StemType,
} from "../types";
import { STEM_TYPES } from "../types";

const FX_ORDER: DeckFxId[] = ["rhythmicGate", "stutter", "glitch", "crush", "phaser", "reverb"];

interface EffectStage {
  input: GainNode;
  output: AudioNode;
  update(params: DeckFxParams): void;
  dispose(): void;
}

interface DeckNodes {
  input: GainNode;
  stemMix: GainNode;
  stemStages: Map<StemType, StemStage>;
  level: GainNode;
  effects: Map<DeckFxId, EffectStage>;
  meterTap: GainNode;
  analyser: AnalyserNode;
  meterData: MeterArray;
}

type MeterArray = Float32Array<ArrayBuffer>;

interface StemStage {
  input: GainNode;
  gate: GainNode;
  output: GainNode;
  dispose(): void;
}

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

function createStemStage(context: AudioContext, stem: StemType): StemStage {
  const input = context.createGain();
  const gate = context.createGain();
  gate.gain.value = 1;
  input.connect(gate);
  let tail: AudioNode = gate;
  const filters: BiquadFilterNode[] = [];

  const appendFilter = (configure: (filter: BiquadFilterNode) => void) => {
    const filter = context.createBiquadFilter();
    configure(filter);
    tail.connect(filter);
    tail = filter;
    filters.push(filter);
  };

  switch (stem) {
    case "vocals":
      appendFilter((filter) => {
        filter.type = "highpass";
        filter.frequency.value = 180;
        filter.Q.value = 0.9;
      });
      appendFilter((filter) => {
        filter.type = "bandpass";
        filter.frequency.value = 2200;
        filter.Q.value = 1.15;
      });
      break;
    case "drums":
      appendFilter((filter) => {
        filter.type = "highpass";
        filter.frequency.value = 120;
        filter.Q.value = 0.8;
      });
      appendFilter((filter) => {
        filter.type = "peaking";
        filter.frequency.value = 2800;
        filter.Q.value = 0.7;
        filter.gain.value = 5.5;
      });
      break;
    case "synths":
    default:
      appendFilter((filter) => {
        filter.type = "lowpass";
        filter.frequency.value = 4200;
        filter.Q.value = 0.75;
      });
      appendFilter((filter) => {
        filter.type = "peaking";
        filter.frequency.value = 900;
        filter.Q.value = 0.9;
        filter.gain.value = 3.5;
      });
      break;
  }

  const output = context.createGain();
  output.gain.value = 0.92;
  tail.connect(output);

  const dispose = () => {
    try {
      input.disconnect();
    } catch {}
    try {
      gate.disconnect();
    } catch {}
    filters.forEach((filter) => {
      try {
        filter.disconnect();
      } catch {}
    });
    try {
      output.disconnect();
    } catch {}
  };

  return { input, gate, output, dispose };
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

  private stemFocus = new Map<DeckId, StemType | null>();

  private masterTrim = 0.9;

  private timestretch = 1;

  private smoothing = 0.08;

  private readonly injectedContext: AudioContext | null;

  private readonly ownsContext: boolean;

  private readonly logger: (...args: unknown[]) => void;

  private readonly debugEnabled: boolean;

  private readonly meterIntervalMs: number;

  private meterInterval: ReturnType<typeof setInterval> | null = null;

  private playback = new Map<DeckId, DeckPlaybackInternal>();

  private diagnosticsListeners = new Set<DiagnosticsListener>();

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
    const stemMix = context.createGain();
    stemMix.gain.value = 0.78;
    const stemStages = new Map<StemType, StemStage>();
    STEM_TYPES.forEach((stem) => {
      const stage = createStemStage(context, stem);
      stemStages.set(stem, stage);
      input.connect(stage.input);
      stage.output.connect(stemMix);
    });
    const level = context.createGain();
    level.gain.value = 0;
    const master = this.ensureMaster(context);
    const meterTap = context.createGain();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.85;
    meterTap.connect(level);
    meterTap.connect(analyser);
    level.connect(master);
    const meterData = new Float32Array(analyser.fftSize) as MeterArray;
    deck = { input, stemMix, stemStages, level, effects: new Map(), meterTap, analyser, meterData };
    this.decks.set(deckId, deck);
    this.effectStates.set(deckId, new Map());
    this.rebuildDeckChain(deckId);
    this.applyStemFocus(deckId, deck, context);
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

  setTimestretch(value: number) {
    const clamped = clamp(value, 0.25, 4);
    const previous = this.timestretch;
    this.timestretch = clamped;
    const context = this.getContext();
    if (!context || previous === clamped) {
      return;
    }
    this.playback.forEach((playback) => {
      if (!playback.isPlaying || playback.startedAt === null) {
        return;
      }
      const duration = playback.durationSeconds ?? playback.buffer?.duration ?? null;
      const elapsed = Math.max(0, context.currentTime - playback.startedAt);
      const advanced = playback.position + elapsed * previous;
      playback.position = duration ? clamp(advanced, 0, duration) : advanced;
      playback.startedAt = context.currentTime;
    });
    this.playback.forEach((playback) => {
      const source = playback.source;
      if (!source) {
        return;
      }
      try {
        if (source.playbackRate && typeof source.playbackRate.setTargetAtTime === "function") {
          source.playbackRate.setTargetAtTime(clamped, context.currentTime, 0.05);
        } else if (source.playbackRate) {
          source.playbackRate.value = clamped;
        }
      } catch {}
    });
    this.updateAllDiagnostics();
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

  private rebuildDeckChain(deckId: DeckId) {
    const deck = this.decks.get(deckId);
    if (!deck) return;
    try {
      deck.stemMix.disconnect();
    } catch {}
    deck.effects.forEach((stage) => {
      try {
        stage.input.disconnect();
      } catch {}
      try {
        stage.output.disconnect();
      } catch {}
    });
    let previous: AudioNode = deck.stemMix;
    for (const effectId of FX_ORDER) {
      const stage = deck.effects.get(effectId);
      if (!stage) continue;
      previous.connect(stage.input);
      previous = stage.output;
    }
    previous.connect(deck.meterTap);
    const context = this.getActiveContext();
    if (context) {
      this.applyStemFocus(deckId, deck, context);
    }
  }

  private applyStemFocus(deckId: DeckId, deck: DeckNodes, context: AudioContext) {
    const focus = this.stemFocus.get(deckId) ?? null;
    deck.stemStages.forEach((stage, type) => {
      const target = focus ? (type === focus ? 1 : 0.18) : 1;
      stage.gate.gain.setTargetAtTime(target, context.currentTime, this.smoothing);
    });
  }

  setDeckStemFocus(deckId: DeckId, stem: StemType | null) {
    const context = this.getContext();
    if (!context) return;
    const deck = this.ensureDeck(context, deckId);
    const normalized = stem ?? null;
    const previous = this.stemFocus.get(deckId) ?? null;
    if (previous === normalized) {
      return;
    }
    this.stemFocus.set(deckId, normalized);
    this.applyStemFocus(deckId, deck, context);
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
    const rate = Math.max(this.timestretch, 0.01);
    const current = clampedPosition + elapsed * rate;
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
      if (source.playbackRate) {
        source.playbackRate.value = this.timestretch;
        try {
          if (typeof source.playbackRate.setValueAtTime === "function") {
            source.playbackRate.setValueAtTime(this.timestretch, context.currentTime);
          }
        } catch {}
      }
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
    this.playback.forEach((playback, deckId) => {
      this.stopPlayback(deckId, playback);
    });
    this.playback.clear();
    this.diagnosticsListeners.clear();
    this.decks.forEach((deck) => {
      deck.effects.forEach((stage) => {
        stage.dispose();
      });
      deck.stemStages.forEach((stage) => {
        stage.dispose();
      });
      try {
        deck.input.disconnect();
      } catch {}
      try {
        deck.stemMix.disconnect();
      } catch {}
      try {
        deck.level.disconnect();
      } catch {}
      try {
        deck.meterTap.disconnect();
      } catch {}
    });
    this.decks.clear();
    this.deckGains.clear();
    this.effectStates.clear();
    this.stemFocus.clear();
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
