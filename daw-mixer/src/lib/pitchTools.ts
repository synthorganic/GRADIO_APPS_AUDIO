import type { SampleClip } from "../types";
import { CAMEL0T_KEYS } from "./demucsClient";

function wrapIndex(index: number): number {
  const length = CAMEL0T_KEYS.length;
  return ((index % length) + length) % length;
}

export function retuneSampleMeasures(sample: SampleClip, targetKey?: string) {
  const fallbackKey = targetKey ?? sample.key ?? "8A";
  const rootIndex = CAMEL0T_KEYS.indexOf(fallbackKey);
  const resolvedRoot = rootIndex === -1 ? 0 : rootIndex;

  const orbit = [
    CAMEL0T_KEYS[resolvedRoot],
    CAMEL0T_KEYS[wrapIndex(resolvedRoot + 7)],
    CAMEL0T_KEYS[wrapIndex(resolvedRoot - 5)],
    CAMEL0T_KEYS[wrapIndex(resolvedRoot + 2)]
  ];

  const measures = sample.measures.map((measure, index) => {
    const tunedPitch = orbit[index % orbit.length];
    return {
      ...measure,
      tunedPitch
    };
  });

  const retuneMap = measures.map((measure, index) => {
    const detected = measure.detectedPitch ?? "?";
    const tuned = measure.tunedPitch ?? "?";
    return `M${index + 1}: ${detected} → ${tuned}`;
  });

  return { measures, retuneMap, tunedKey: CAMEL0T_KEYS[resolvedRoot] };
}
