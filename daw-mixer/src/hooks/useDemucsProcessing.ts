import { useCallback, useState } from "react";
import { attachDemucsToSample, runDemucs } from "../lib/demucsClient";
import type { SampleClip } from "../types";

export function useDemucsProcessing(onComplete?: (sample: SampleClip) => void) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const processSample = useCallback(
    async (sample: SampleClip): Promise<SampleClip | null> => {
      if (isProcessing) return null;
      setIsProcessing(true);
      setError(null);
      try {
        const result = await runDemucs(sample.file);
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
    [isProcessing, onComplete]
  );

  return { isProcessing, error, processSample };
}
