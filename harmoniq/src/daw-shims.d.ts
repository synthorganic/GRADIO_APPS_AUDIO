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
