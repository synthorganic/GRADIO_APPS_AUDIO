import type { StemInfo } from "../types";

export type StemEngineId = "rvcv" | "uvrmdxnet";

export interface StemHeuristicSettings {
  percussion: {
    attenuation: boolean;
    tonalCut: boolean;
  };
  vocals: {
    attenuationCut: boolean;
    profileMatch: boolean;
    frequencyShift: boolean;
  };
}

export interface StemEngineDefinition {
  id: StemEngineId;
  name: string;
  description: string;
  models: Partial<Record<StemInfo["type"], string>>;
  heuristics: {
    percussion: {
      attenuation: string;
      tonalCut: string;
    };
    vocals: {
      attenuationCut: string;
      profileMatch: string;
      frequencyShift: string;
    };
  };
}

export interface StemProcessingOptions {
  engine: StemEngineId;
  heuristics: StemHeuristicSettings;
}
