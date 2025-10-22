export type DeckId = "A" | "B" | "C" | "D";

export type LoopSlotStatus = "idle" | "queued" | "recording" | "playing";

export interface LoopSlot {
  id: string;
  label: string;
  status: LoopSlotStatus;
  length: "bar" | "half";
}

export type StemStatus = "standby" | "active" | "muted";

export type DeckFxId = "reverb" | "rhythmicGate" | "stutter" | "glitch" | "crush" | "phaser";

export type DeckFxParams = Record<string, number | string>;

export interface DeckPerformance {
  id: DeckId;
  loopName: string;
  waveform: Float32Array;
  filter: number;
  resonance: number;
  zoom: number;
  fxActive: Record<DeckFxId, boolean>;
  isFocused: boolean;
  level: number;
  bpm?: number;
  tonalKey?: string;
  mood?: string;
  eqCuts: Record<EqBandId, boolean>;
  activeStem: StemType | null;
  queuedStem: StemType | null;
  stemStatus: "main" | "queued" | "stem";
  stems?: DeckStem[];
  source?: string;
  scale?: string;
  durationSeconds?: number | null;
  currentTimeSeconds?: number;
  isPlaying?: boolean;
  playbackError?: string | null;
  vu?: number;
  objectUrl?: string;
  trackId?: string;
  file?: File | null;
}

export interface CrossfadeState {
  x: number;
  y: number;
}

export type StemType = "vocals" | "drums" | "synths";

export const STEM_TYPES: readonly StemType[] = ["vocals", "drums", "synths"];

export interface DeckStem {
  id: string;
  label: string;
  status: StemStatus;
  type: StemType;
  sourceStemId?: string | null;
}

export interface DeckPlaybackDiagnostics {
  deckId: DeckId;
  isPlaying: boolean;
  currentTimeSeconds: number;
  durationSeconds: number | null;
  vu: number;
  error: string | null;
}

export interface DeckAudioSource {
  id: string;
  objectUrl: string;
  file?: File;
  arrayBuffer?: ArrayBuffer;
  name?: string;
}
