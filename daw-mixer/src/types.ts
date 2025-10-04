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
