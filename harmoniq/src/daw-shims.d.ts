declare module "@daw/theme" {
  export const theme: {
    background: string;
    surface: string;
    surfaceRaised: string;
    surfaceOverlay: string;
    border: string;
    text: string;
    textMuted: string;
    accentBeam: string[];
    button: {
      base: string;
      hover: string;
      active: string;
      primary: string;
      primaryText: string;
      disabled: string;
      outline: string;
    };
    shadow: string;
    divider: string;
    cardGlow: string;
  };
  export type Theme = typeof theme;
}

declare module "@daw/components/layout/styles" {
  import type { CSSProperties } from "react";
  export const cardSurfaceStyle: CSSProperties;
  export const toolbarButtonStyle: CSSProperties;
  export const toolbarButtonDisabledStyle: CSSProperties;
  export const gridRows: (...values: Array<string | number>) => string;
}

declare module "@daw/shared/WaveformPreview" {
  import type { FC } from "react";
  export interface WaveformPreviewProps {
    waveform: Float32Array;
    fillColor?: string;
    strokeColor?: string;
  }
  export interface WaveformPathResult {
    commands: string;
    viewWidth: number;
  }
  export const WaveformPreview: FC<WaveformPreviewProps>;
}

declare module "@daw/lib/audioEngine" {
  export const audioEngine: {
    setTempo(bpm: number): void;
  };
}

type HarmoniqDeckId = "A" | "B" | "C" | "D";

type HarmoniqLoopSlotLength = "bar" | "half";

type HarmoniqLoopSlotStatus = "idle" | "queued" | "recording" | "playing";

interface HarmoniqLoopArmingEntry {
  deckId: HarmoniqDeckId;
  slotId: string;
  length: HarmoniqLoopSlotLength;
  startTime: number;
  stopTime: number;
  state: "waiting" | "recording";
}

declare const transportState: import("react").MutableRefObject<{
  position: number;
  hasTick: boolean;
}>;

declare const loopArmings: import("react").MutableRefObject<Map<string, HarmoniqLoopArmingEntry>>;

declare function processLoopArmings(position: number): void;
declare function resetLoopArmings(cancelPending?: boolean): void;
declare function scheduleLoopArming(
  deckId: HarmoniqDeckId,
  slotId: string,
  length: HarmoniqLoopSlotLength,
): void;
declare function clearLoopTimersForSlot(key: string): void;
declare function registerLoopTimer(key: string, delay: number, callback: () => void): void;
declare function updateLoopSlotStatus(
  deckId: HarmoniqDeckId,
  slotId: string,
  status: HarmoniqLoopSlotStatus,
): void;
