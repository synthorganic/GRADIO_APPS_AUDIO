import { useCallback, useState } from "react";
import { attachDemucsToSample, runDemucs } from "../lib/demucsClient";
import type { SampleClip } from "../types";
import { DEFAULT_ENGINE_ID, DEFAULT_HEURISTICS, type StemProcessingOptions } from "../stem_engines";

export function useDemucsProcessing(
  onComplete?: (sample: SampleClip) => void,
  options?: StemProcessingOptions,
) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processSample = useCallback(
    async (sample: SampleClip): Promise<SampleClip | null> => {
      if (isProcessing) return null;
      setIsProcessing(true);
      setError(null);
      try {
        const runtimeOptions = options ?? {
          engine: DEFAULT_ENGINE_ID,
          heuristics: DEFAULT_HEURISTICS,
        };
        const result = await runDemucs(sample.file, runtimeOptions);
        const updated = attachDemucsToSample(sample, result);
        onComplete?.(updated);
        return updated;
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        return null;
      } finally {
        setIsProcessing(false);
      }
    },
    [isProcessing, onComplete, options]
  );

  return { isProcessing, error, processSample };
}
