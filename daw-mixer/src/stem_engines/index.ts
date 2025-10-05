import { nanoid } from "nanoid";
import type { StemInfo } from "../types";
import type { StemEngineDefinition, StemEngineId, StemHeuristicSettings } from "./types";

export type { StemEngineDefinition, StemEngineId, StemHeuristicSettings, StemProcessingOptions } from "./types";

export const DEFAULT_ENGINE_ID: StemEngineId = "uvrmdxnet";

export const DEFAULT_HEURISTICS: StemHeuristicSettings = {
  percussion: {
    attenuation: true,
    tonalCut: true,
  },
  vocals: {
    attenuationCut: true,
    profileMatch: true,
    frequencyShift: true,
  },
};

const RVCV_ENGINE: StemEngineDefinition = {
  id: "rvcv",
  name: "RVCV Extractor",
  description:
    "Realtime Voice Conversion tuned for expressive midrange isolation and transient-aware rhythm carving.",
  models: {
    full: "RVCV Full-Band Re-synth",
    vocals: "RVCV Vocal Imprint",
    leads: "RVCV Melody Split",
    percussion: "RVCV Transient Gate",
    kicks: "RVCV Low-End Sculpt",
    bass: "RVCV Harmonic Anchor",
  },
  heuristics: {
    percussion: {
      attenuation: "Attenuation tracking to clamp resonant tails",
      tonalCut: "Cutting tonal heuristics to keep drums dry",
    },
    vocals: {
      attenuationCut: "Attenuated ambience removal for clear phrasing",
      profileMatch: "Profile matching against vocal formant ranges",
      frequencyShift: "Frequency-shift pattern alignment for vibrato",
    },
  },
};

const UVR_ENGINE: StemEngineDefinition = {
  id: "uvrmdxnet",
  name: "UVR MDX-NET Suite",
  description:
    "Ultimate Vocal Remover stack with multi-model blending for detailed separation and stem enhancement.",
  models: {
    full: "UVR Full Reference",
    vocals: "UVR-MDX-NET Karaoke v2",
    leads: "UVR-MDX-NET Main",
    percussion: "UVR-MDX-NET Percussion v3",
    kicks: "UVR-MDX-NET Low-End",
    bass: "UVR-MDX-NET Bass Tight",
  },
  heuristics: {
    percussion: {
      attenuation: "Attenuation scanning with transient emphasis",
      tonalCut: "Tonal cutback heuristics to drop cymbal wash",
    },
    vocals: {
      attenuationCut: "Cutting attenuated bleed for tighter comps",
      profileMatch: "Pulling formants that match vocal profiles",
      frequencyShift: "Frequency-shift pattern guidance for doubles",
    },
  },
};

export const STEM_ENGINES: StemEngineDefinition[] = [RVCV_ENGINE, UVR_ENGINE];

export function getStemEngineDefinition(id: StemEngineId): StemEngineDefinition {
  return STEM_ENGINES.find((engine) => engine.id === id) ?? UVR_ENGINE;
}

export function cloneHeuristics(settings: StemHeuristicSettings): StemHeuristicSettings {
  return {
    percussion: { ...settings.percussion },
    vocals: { ...settings.vocals },
  };
}

export function describeHeuristics(
  engine: StemEngineDefinition,
  heuristics: StemHeuristicSettings,
): {
  percussion: string[];
  vocals: string[];
} {
  const percussion: string[] = [];
  if (heuristics.percussion.attenuation) {
    percussion.push(engine.heuristics.percussion.attenuation);
  }
  if (heuristics.percussion.tonalCut) {
    percussion.push(engine.heuristics.percussion.tonalCut);
  }

  const vocals: string[] = [];
  if (heuristics.vocals.attenuationCut) {
    vocals.push(engine.heuristics.vocals.attenuationCut);
  }
  if (heuristics.vocals.profileMatch) {
    vocals.push(engine.heuristics.vocals.profileMatch);
  }
  if (heuristics.vocals.frequencyShift) {
    vocals.push(engine.heuristics.vocals.frequencyShift);
  }

  return { percussion, vocals };
}

export function applyEngineToStem(
  stem: StemInfo,
  engine: StemEngineDefinition,
  heuristics: StemHeuristicSettings,
): StemInfo {
  const model = engine.models[stem.type];
  const summary = describeHeuristics(engine, heuristics);
  const baseNotes = stem.processingNotes ? [stem.processingNotes] : [];
  let processingNotes: string | undefined;
  if (stem.type === "percussion") {
    const combined = [...summary.percussion, ...baseNotes];
    processingNotes = combined.length > 0 ? combined.join(" • ") : undefined;
  } else if (stem.type === "vocals") {
    const combined = [...summary.vocals, ...baseNotes];
    processingNotes = combined.length > 0 ? combined.join(" • ") : undefined;
  } else {
    processingNotes = baseNotes.length > 0 ? baseNotes.join(" • ") : undefined;
  }

  return {
    ...stem,
    id: stem.id || nanoid(),
    extractionModel: model ?? stem.extractionModel,
    processingNotes,
  };
}
