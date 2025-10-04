import type { TrackEffects } from "../types";

export function createDefaultTrackEffects(): TrackEffects {
  return {
    glitch: { enabled: false, mix: 0.5, density: 0.4, scatter: 0.3 },
    rhythmicGate: { enabled: false, mix: 0.5, rate: "1/8", swing: 0.1 },
    slicer: { enabled: false, mix: 0.5, division: "1/4", jitter: 0.15 },
    bitcrush: { enabled: false, mix: 0.45, depth: 12, downsample: 4 },
    reverb: { enabled: false, mix: 0.35, size: 0.6, decay: 0.55 },
    delay: { enabled: false, mix: 0.35, time: "1/4", feedback: 0.45 },
    chorus: { enabled: false, mix: 0.4, rate: 0.35, depth: 0.5 },
    hiPass: { enabled: false, mix: 1, cutoff: 120, resonance: 0.3 },
    lowPass: { enabled: false, mix: 1, cutoff: 16000, resonance: 0.25 },
    vst: { enabled: false, pluginName: "", notes: "" }
  };
}

export function cloneTrackEffects(effects?: TrackEffects): TrackEffects {
  const source = effects ?? createDefaultTrackEffects();
  return JSON.parse(JSON.stringify(source)) as TrackEffects;
}
