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
  extractionModel?: string;
  processingNotes?: string;
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

export type ChannelType = "audio" | "automation" | "midi";

export interface AutomationPoint {
  id: string;
  time: number;
  value: number;
}

export interface MidiNote {
  id: string;
  blockId: string;
  start: number;
  length: number;
  pitch: number;
  velocity: number;
  sampleId?: string;
}

export interface MidiBlock {
  id: string;
  start: number;
  length: number;
  notes: MidiNote[];
  rootNote: number;
  baseMidi?: number;
  sampleId?: string;
  harmonicSource?: string;
}

export interface TimelineChannelBase {
  id: string;
  name: string;
  type: ChannelType;
  color: string;
  isFxEnabled: boolean;
  volume: number;
  pan: number;
}

export interface AudioChannel extends TimelineChannelBase {
  type: "audio";
}

export interface AutomationChannel extends TimelineChannelBase {
  type: "automation";
  parameterId: string;
  parameterLabel: string;
  points: AutomationPoint[];
}

export interface MidiChannel extends TimelineChannelBase {
  type: "midi";
  instrument: string;
  blocks: MidiBlock[];
  blockSizeMeasures: number;
}

export type TimelineChannel = AudioChannel | AutomationChannel | MidiChannel;

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
  waveform?: Float32Array;
  isInTimeline?: boolean;
  channelId?: string;
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
  scale: string;
  samples: SampleClip[];
  channels: TimelineChannel[];
  mastering: MasteringSettings;
}

export interface MasteringSettings {
  widenStereo: number;
  glueCompression: number;
  spectralTilt: number;
  limiterCeiling: number;
  tapeSaturation: number;
}
