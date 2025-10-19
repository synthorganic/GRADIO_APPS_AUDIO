export type DeckId = "A" | "B" | "C" | "D";

export type LoopSlotStatus = "idle" | "queued" | "recording" | "playing";

export interface LoopSlot {
  id: string;
  label: string;
  status: LoopSlotStatus;
  length: "bar" | "half";
}

export type StemStatus = "standby" | "active" | "muted";

export interface DeckStem {
  id: string;
  label: string;
  status: StemStatus;
}

export interface DeckPerformance {
  id: DeckId;
  loopName: string;
  waveform: Float32Array;
  filter: number;
  resonance: number;
  zoom: number;
  fxStack: string[];
  isFocused: boolean;
  level: number;
  bpm?: number;
  tonalKey?: string;
  mood?: string;
  eqCuts: Record<"highs" | "mids" | "lows", boolean>;
  activeStem: StemType | null;
  queuedStem: StemType | null;
  stemStatus: "main" | "queued" | "stem";
  stems?: DeckStem[];
  source?: string;
  scale?: string;
}

export interface CrossfadeState {
  x: number;
  y: number;
}

export type StemType = "vocals" | "drums" | "synths";
