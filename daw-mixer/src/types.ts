export type StemType =
  | "full"
  | "vocals"
  | "leads"
  | "percussion"
  | "kicks"
  | "bass";

export interface StemInfo {
  id: string;
  name: string;
  type: StemType;
  color: string;
  url?: string;
  bpm?: number;
  key?: string;
  waveform?: Float32Array;
  sourceStemId?: string;
  startOffset?: number;
  duration?: number;
}

export interface Measure {
  id: string;
  start: number;
  end: number;
  beatCount: number;
  isDownbeat: boolean;
  detectedPitch?: string;
  tunedPitch?: string;
  energy?: number;
  beats?: Beat[];
}

export interface GlitchEffectSettings {
  enabled: boolean;
  mix: number;
  density: number;
  scatter: number;
}

export interface GateEffectSettings {
  enabled: boolean;
  mix: number;
  rate: "1/4" | "1/8" | "1/16" | "1/32";
  swing: number;
}

export interface SlicerEffectSettings {
  enabled: boolean;
  mix: number;
  division: "1/2" | "1/4" | "1/8" | "1/16";
  jitter: number;
}

export interface BitcrushEffectSettings {
  enabled: boolean;
  mix: number;
  depth: number;
  downsample: number;
}

export interface ReverbEffectSettings {
  enabled: boolean;
  mix: number;
  size: number;
  decay: number;
}

export interface DelayEffectSettings {
  enabled: boolean;
  mix: number;
  time: "1/8" | "1/4" | "1/2" | "1";
  feedback: number;
}

export interface ChorusEffectSettings {
  enabled: boolean;
  mix: number;
  rate: number;
  depth: number;
}

export interface FilterEffectSettings {
  enabled: boolean;
  mix: number;
  cutoff: number;
  resonance: number;
}

export interface VstRoutingSettings {
  enabled: boolean;
  pluginName: string;
  notes?: string;
}

export interface TrackEffects {
  glitch: GlitchEffectSettings;
  rhythmicGate: GateEffectSettings;
  slicer: SlicerEffectSettings;
  bitcrush: BitcrushEffectSettings;
  reverb: ReverbEffectSettings;
  delay: DelayEffectSettings;
  chorus: ChorusEffectSettings;
  hiPass: FilterEffectSettings;
  lowPass: FilterEffectSettings;
  vst: VstRoutingSettings;
}

export interface SampleClip {
  id: string;
  name: string;
  file?: File;
  url?: string;
  bpm?: number;
  key?: string;
  measures: Measure[];
  stems: StemInfo[];
  position: number;
  length: number;
  isLooping: boolean;
  startOffset?: number;
  duration?: number;
  originSampleId?: string;
  isFragment?: boolean;
  variantLabel?: string;
  retuneMap?: string[];
  rekeyedAt?: string;
  effects: TrackEffects;
  isInTimeline?: boolean;
}

export interface Beat {
  id: string;
  start: number;
  end: number;
  index: number;
  stems: StemInfo[];
}

export interface Project {
  id: string;
  name: string;
  masterBpm: number;
  samples: SampleClip[];
  mastering: MasteringSettings;
}

export interface MasteringSettings {
  widenStereo: number;
  glueCompression: number;
  spectralTilt: number;
  limiterCeiling: number;
  tapeSaturation: number;
}
