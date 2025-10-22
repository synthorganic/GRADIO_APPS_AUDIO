import { useEffect, useMemo, useRef, useState } from "react";
import { theme } from "@daw/theme";
import { DeckMatrix } from "./components/DeckMatrix";
import type {
  CrossfadeState,
  DeckAudioSource,
  DeckFxId,
  DeckFxParams,
  DeckId,
  DeckPerformance,
  DeckPlaybackDiagnostics,
  DeckStem,
  LoopSlot,
  EqBandId,
  StemType,
  StemStatus,
} from "./types";
import { STEM_TYPES } from "./types";
import { HarmonicWheelSelector } from "./components/HarmonicWheelSelector";
import { FxRackPanel, type FxModuleConfig } from "./components/FxRackPanel";
import { createWaveform, createWaveformFromAudioBuffer } from "./shared/waveforms";
import { useLoopLibrary } from "./state/LoopLibraryStore";
import { TrackSelectionModal } from "./components/TrackSelectionModal";
import {
  TrackUploadPanel,
  type AnalyzedStem,
  type AnalyzedTrackSummary,
} from "./components/TrackUploadPanel";
import { TrackLibraryList } from "./components/TrackLibraryList";
import { HarmoniqAudioBridge } from "./lib/HarmoniqAudioBridge";
import { encodeAudioBufferToWav } from "./lib/audioEncoding";

const CAMELOT_ORDER = [
  "1A",
  "2A",
  "3A",
  "4A",
  "5A",
  "6A",
  "7A",
  "8A",
  "9A",
  "10A",
  "11A",
  "12A",
  "1B",
  "2B",
  "3B",
  "4B",
  "5B",
  "6B",
  "7B",
  "8B",
  "9B",
  "10B",
  "11B",
  "12B",
];

const CAMELOT_RANK = new Map(CAMELOT_ORDER.map((key, index) => [key, index]));

const FX_IDS: DeckFxId[] = ["reverb", "rhythmicGate", "stutter", "glitch", "crush", "phaser"];

const FX_DEFAULT_PARAMS: Record<DeckFxId, DeckFxParams> = {
  reverb: { mix: 0.65, size: 0.72, decay: 0.66 },
  rhythmicGate: { mix: 0.85, rate: "1/8", swing: 0.15 },
  stutter: { mix: 0.62, division: "1/8", jitter: 0.25 },
  glitch: { mix: 0.7, density: 0.5, scatter: 0.4 },
  crush: { mix: 0.78, depth: 6, downsample: 0.35 },
  phaser: { mix: 0.68, rate: 0.35, depth: 0.55 },
};

const DEFAULT_FX_STATE: Record<DeckFxId, boolean> = {
  reverb: false,
  rhythmicGate: false,
  stutter: false,
  glitch: false,
  crush: false,
  phaser: false,
};

function createFxState(overrides: Partial<Record<DeckFxId, boolean>> = {}) {
  return { ...DEFAULT_FX_STATE, ...overrides };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

const STEM_LABEL_DEFAULTS: Record<StemType, string> = {
  vocals: "Vocals",
  drums: "Drums",
  synths: "Synths",
};

const STEM_KEYWORDS: Record<StemType, string[]> = {
  vocals: ["vocal", "vox", "singer", "lyric", "lead vocal", "voice", "acapella"],
  drums: ["drum", "percussion", "perc", "beat", "kick", "snare", "hi-hat", "rhythm"],
  synths: ["synth", "melody", "lead", "keys", "pad", "bass", "chord", "instrument"],
};

function scoreStemCandidate(stem: AnalyzedStem, target: StemType) {
  const haystack = `${stem.id} ${stem.label}`.toLowerCase();
  const keywords = STEM_KEYWORDS[target];
  return keywords.reduce((score, keyword, index) => {
    if (!keyword) return score;
    return haystack.includes(keyword) ? score + (keywords.length - index) : score;
  }, 0);
}

function assignDeckStems(trackId: string, stems: AnalyzedStem[]): DeckStem[] {
  const pool = stems.map((stem, index) => ({ stem, index }));
  const used = new Set<number>();
  const results: DeckStem[] = [];

  const pickBest = (type: StemType): DeckStem | null => {
    let bestIndex = -1;
    let bestScore = 0;
    pool.forEach(({ stem, index }) => {
      if (used.has(index)) return;
      const score = scoreStemCandidate(stem, type);
      if (score > bestScore || (score > 0 && score === bestScore && bestIndex === -1)) {
        bestScore = score;
        bestIndex = index;
      }
    });
    if (bestIndex === -1) {
      return null;
    }
    used.add(bestIndex);
    const match = pool[bestIndex].stem;
    return {
      id: `${trackId}-${type}`,
      label: match.label,
      status: "standby",
      type,
      sourceStemId: match.id,
    };
  };

  const pullFallback = (): { stem: AnalyzedStem; index: number } | null => {
    for (const entry of pool) {
      if (!used.has(entry.index)) {
        used.add(entry.index);
        return entry;
      }
    }
    return null;
  };

  STEM_TYPES.forEach((type) => {
    const best = pickBest(type);
    if (best) {
      results.push(best);
      return;
    }
    const fallback = pullFallback();
    if (fallback) {
      results.push({
        id: `${trackId}-${type}`,
        label: fallback.stem.label,
        status: "standby",
        type,
        sourceStemId: fallback.stem.id,
      });
      return;
    }
    results.push({
      id: `${trackId}-${type}`,
      label: STEM_LABEL_DEFAULTS[type],
      status: "standby",
      type,
      sourceStemId: null,
    });
  });

  return results;
}

function computeCrossfadeWeights(crossfade: CrossfadeState): Record<DeckId, number> {
  const left = 1 - crossfade.x;
  const right = crossfade.x;
  const top = 1 - crossfade.y;
  const bottom = crossfade.y;
  return {
    A: Number((left * top).toFixed(4)),
    B: Number((right * top).toFixed(4)),
    C: Number((left * bottom).toFixed(4)),
    D: Number((right * bottom).toFixed(4)),
  };
}

function sortLoopsByCamelot<T extends { key: string }>(loops: T[]): T[] {
  return [...loops].sort((a, b) => {
    const rankA = CAMELOT_RANK.get(a.key) ?? Number.MAX_SAFE_INTEGER;
    const rankB = CAMELOT_RANK.get(b.key) ?? Number.MAX_SAFE_INTEGER;
    if (rankA === rankB) {
      return a.key.localeCompare(b.key);
    }
    return rankA - rankB;
  });
}

const INITIAL_DECKS: DeckPerformance[] = [
  {
    id: "A",
    loopName: "Neon Skyline",
    waveform: createWaveform(0.8),
    filter: 0.42,
    resonance: 0.4,
    zoom: 1,
    fxActive: createFxState({ reverb: true, stutter: true, phaser: true }),
    isFocused: true,
    level: 0.68,
    bpm: 124,
    tonalKey: "8A",
    mood: "Glasswave",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
    durationSeconds: 252,
    currentTimeSeconds: 0,
    isPlaying: false,
    playbackError: null,
    vu: 0,
  },
  {
    id: "B",
    loopName: "Chromatic Drift",
    waveform: createWaveform(1.3),
    filter: 0.53,
    resonance: 0.5,
    zoom: 1,
    fxActive: createFxState({ phaser: true, crush: true }),
    isFocused: false,
    level: 0.72,
    bpm: 128,
    tonalKey: "2B",
    mood: "Retrograde",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
    durationSeconds: 236,
    currentTimeSeconds: 0,
    isPlaying: false,
    playbackError: null,
    vu: 0,
  },
  {
    id: "C",
    loopName: "Vapor Trails",
    waveform: createWaveform(0.35),
    filter: 0.31,
    resonance: 0.46,
    zoom: 1.1,
    fxActive: createFxState({ phaser: true, rhythmicGate: true }),
    isFocused: false,
    level: 0.44,
    bpm: 118,
    tonalKey: "11B",
    mood: "Azure",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
    durationSeconds: 214,
    currentTimeSeconds: 0,
    isPlaying: false,
    playbackError: null,
    vu: 0,
  },
  {
    id: "D",
    loopName: "Night Echo",
    waveform: createWaveform(1.62),
    filter: 0.37,
    resonance: 0.42,
    zoom: 0.92,
    fxActive: createFxState({ reverb: true, crush: true }),
    isFocused: false,
    level: 0.58,
    bpm: 122,
    tonalKey: "5A",
    mood: "Hypnotic",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
    durationSeconds: 206,
    currentTimeSeconds: 0,
    isPlaying: false,
    playbackError: null,
    vu: 0,
  },
];

function createLoopSlots(deckId: DeckId): LoopSlot[] {
  const labels = ["Clip 1", "Clip 2", "Clip 3", "Clip 4", "Clip 5", "Clip 6"];
  return labels.map((label, index) => ({
    id: `${deckId.toLowerCase()}-clip-${index + 1}`,
    label,
    status: "idle",
    length: index % 2 === 0 ? "bar" : "half",
  }));
}

const INITIAL_LOOP_SLOTS: Record<DeckId, LoopSlot[]> = {
  A: createLoopSlots("A"),
  B: createLoopSlots("B"),
  C: createLoopSlots("C"),
  D: createLoopSlots("D"),
};

const MASTER_BASE_BPM = 128;

const FX_RACK_PRESETS: Record<"left" | "right", FxModuleConfig[]> = {
  left: [
    { id: "drive", label: "Drive", amount: 0.62, enabled: true, accent: "rgba(255, 148, 241, 0.85)" },
    { id: "chorus", label: "Chorus", amount: 0.46, enabled: true, accent: "rgba(132, 94, 255, 0.85)" },
    { id: "noise", label: "Noise Gate", amount: 0.28, enabled: false, accent: "rgba(120, 203, 220, 0.85)" },
    { id: "bit", label: "Bit Crush", amount: 0.4, enabled: true, accent: "rgba(255, 112, 173, 0.85)" },
  ],
  right: [
    { id: "reverb", label: "Reverb", amount: 0.74, enabled: true, accent: "rgba(126, 244, 255, 0.85)" },
    { id: "delay", label: "Tape Echo", amount: 0.58, enabled: true, accent: "rgba(255, 211, 137, 0.85)" },
    { id: "phaser", label: "Phaser", amount: 0.33, enabled: true, accent: "rgba(132, 94, 255, 0.85)" },
    { id: "duck", label: "Duck", amount: 0.52, enabled: false, accent: "rgba(255, 148, 241, 0.85)" },
  ],
};

export default function App() {
  const { loops: storedLoops, registerLoopLoad, addLoop } = useLoopLibrary();

  const [decks, setDecks] = useState<DeckPerformance[]>(INITIAL_DECKS);
  const [crossfade, setCrossfade] = useState<CrossfadeState>({ x: 0.45, y: 0.35 });
  const [selectedKey, setSelectedKey] = useState("8A");
  const [loopSlots, setLoopSlots] = useState<Record<DeckId, LoopSlot[]>>(INITIAL_LOOP_SLOTS);
  const [masterTimestretch, setMasterTimestretch] = useState(1);
  const [masterTrim, setMasterTrim] = useState(0.9);
  const [selectorKey, setSelectorKey] = useState<string | null>(null);
  const [libraryTracks, setLibraryTracks] = useState<AnalyzedTrackSummary[]>([]);
  const masterBpm = useMemo(() => MASTER_BASE_BPM * masterTimestretch, [masterTimestretch]);
  const loopTimers = useRef<Map<string, number>>(new Map());
  const captureObjectUrls = useRef<Map<DeckId, string>>(new Map());
  const audioBridge = useMemo(() => {
    if (typeof window === "undefined") {
      return null;
    }
    return new HarmoniqAudioBridge();
  }, []);

  useEffect(() => {
    return () => {
      audioBridge?.dispose();
    };
  }, [audioBridge]);

  const handleTracksAnalyzed = (tracks: AnalyzedTrackSummary[]) => {
    if (!tracks.length) return;
    void (async () => {
      const enriched = await Promise.all(
        tracks.map(async (track) => {
          if (!audioBridge) {
            return {
              ...track,
              durationSeconds: track.durationSeconds ?? null,
              analysisError: track.analysisError ?? "Audio engine unavailable",
            };
          }
          try {
            const analysis = await audioBridge.analyzeSource({
              id: track.id,
              objectUrl: track.objectUrl,
              file: track.file,
              name: track.name,
            });
            return {
              ...track,
              durationSeconds: analysis.durationSeconds ?? track.durationSeconds ?? null,
              analysisError: null,
            };
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            return {
              ...track,
              durationSeconds: null,
              analysisError: `Unable to decode audio: ${message}`,
            };
          }
        }),
      );
      setLibraryTracks((prev) => {
        const merged = new Map<string, AnalyzedTrackSummary>();
        prev.forEach((item) => {
          merged.set(item.origin, item);
        });
        enriched.forEach((item) => {
          merged.set(item.origin, item);
        });
        return Array.from(merged.values()).sort((a, b) => a.name.localeCompare(b.name));
      });
    })();
  };

  const computeSeedFromName = (name: string) => {
    const normalized = name.toLowerCase();
    let hash = 0;
    for (let index = 0; index < normalized.length; index += 1) {
      hash = (hash * 33 + normalized.charCodeAt(index)) & 0xffffffff;
    }
    return Math.abs(hash % 2500) / 1000 + 0.3;
  };

  const handleFocusDeck = (deckId: DeckId) => {
    setDecks((prev) => prev.map((deck) => ({ ...deck, isFocused: deck.id === deckId })));
    const targets: Record<DeckId, CrossfadeState> = {
      A: { x: 0, y: 0 },
      B: { x: 1, y: 0 },
      C: { x: 0, y: 1 },
      D: { x: 1, y: 1 },
    };
    setCrossfade(targets[deckId]);
  };

  const handleLoadTrackToDeck = (deckId: DeckId, track: AnalyzedTrackSummary) => {
    if (track.analysisError) {
      setDecks((prev) =>
        prev.map((deck) =>
          deck.id === deckId ? { ...deck, playbackError: track.analysisError, isPlaying: false } : deck,
        ),
      );
      return;
    }
    const waveformSeed = computeSeedFromName(track.name);
    const assignedStems = assignDeckStems(track.id, track.stems);
    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === deckId
          ? {
              ...deck,
              loopName: track.name,
              waveform: createWaveform(waveformSeed),
              bpm: track.bpm,
              tonalKey: track.scale,
              scale: track.scale,
              source: track.origin,
              stems: assignedStems,
              durationSeconds: track.durationSeconds ?? deck.durationSeconds ?? null,
              currentTimeSeconds: 0,
              isPlaying: false,
              playbackError: null,
              objectUrl: track.objectUrl,
              trackId: track.id,
              file: track.file,
              vu: 0,
            }
          : deck,
      ),
    );
    setLoopSlots((prev) => ({
      ...prev,
      [deckId]: prev[deckId]?.map((slot) => ({ ...slot, status: "idle" })) ?? createLoopSlots(deckId),
    }));
    handleFocusDeck(deckId);
    if (audioBridge) {
      audioBridge.setDeckStemFocus(deckId, null);
    }
    if (audioBridge) {
      const source: DeckAudioSource = {
        id: track.id,
        objectUrl: track.objectUrl,
        file: track.file,
        name: track.name,
      };
      audioBridge
        .loadDeckAudio(deckId, source)
        .then((result) => {
          setDecks((prev) =>
            prev.map((deck) =>
              deck.id === deckId
                ? {
                    ...deck,
                    durationSeconds:
                      result.durationSeconds ?? track.durationSeconds ?? deck.durationSeconds ?? null,
                    playbackError: null,
                    objectUrl: source.objectUrl,
                    trackId: source.id,
                    file: source.file ?? deck.file ?? null,
                  }
                : deck,
            ),
          );
        })
        .catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          setDecks((prev) =>
            prev.map((deck) =>
              deck.id === deckId
                ? {
                    ...deck,
                    playbackError: `Playback failed: ${message}`,
                    isPlaying: false,
                  }
                : deck,
            ),
          );
        });
    }
  };

  const updatePlaybackError = (deckId: DeckId, message: string | null) => {
    setDecks((prev) =>
      prev.map((deck) => (deck.id === deckId ? { ...deck, playbackError: message, isPlaying: message ? false : deck.isPlaying } : deck)),
    );
  };

  const handleRetryPlayback = async (deckId: DeckId) => {
    if (!audioBridge) return;
    const deck = decks.find((item) => item.id === deckId);
    if (!deck || !deck.objectUrl || !deck.trackId) {
      return;
    }
    try {
      await audioBridge.reloadDeck(deckId, {
        id: deck.trackId,
        objectUrl: deck.objectUrl,
        file: deck.file ?? undefined,
        name: deck.loopName,
      });
      updatePlaybackError(deckId, null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updatePlaybackError(deckId, message);
    }
  };

  const handleTogglePlayback = async (deckId: DeckId) => {
    if (!audioBridge) return;
    const deck = decks.find((item) => item.id === deckId);
    if (!deck) return;
    if (deck.playbackError) {
      await handleRetryPlayback(deckId);
      return;
    }
    try {
      if (deck.isPlaying) {
        await audioBridge.stopDeck(deckId);
      } else {
        await audioBridge.playDeck(deckId);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updatePlaybackError(deckId, message);
    }
  };

  const handleSeekDeck = async (deckId: DeckId, ratio: number) => {
    if (!audioBridge) return;
    const deck = decks.find((item) => item.id === deckId);
    if (!deck || !deck.durationSeconds || Number.isNaN(deck.durationSeconds)) {
      return;
    }
    const clampedRatio = clamp(ratio, 0, 1);
    const targetSeconds = deck.durationSeconds * clampedRatio;
    try {
      await audioBridge.seekDeck(deckId, targetSeconds);
      setDecks((prev) =>
        prev.map((item) =>
          item.id === deckId
            ? {
                ...item,
                currentTimeSeconds: targetSeconds,
              }
            : item,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      updatePlaybackError(deckId, message);
    }
  };

  useEffect(() => {
    if (!audioBridge) return;
    const weights = computeCrossfadeWeights(crossfade);
    decks.forEach((deck) => {
      const deckWeight = weights[deck.id] ?? 0;
      const blend = clamp(deck.level * deckWeight, 0, 1);
      audioBridge.setDeckBlend(deck.id, blend);
    });
  }, [audioBridge, crossfade, decks]);

  useEffect(() => {
    if (!audioBridge) return;
    audioBridge.setMasterTrim(masterTrim);
  }, [audioBridge, masterTrim]);

  useEffect(() => {
    if (!audioBridge) return;
    audioBridge.setTimestretch(masterTimestretch);
  }, [audioBridge, masterTimestretch]);

  useEffect(() => {
    if (!audioBridge) return;
    decks.forEach((deck) => {
      FX_IDS.forEach((effectId) => {
        const enabled = deck.fxActive[effectId] ?? false;
        audioBridge.setEffectState(deck.id, effectId, enabled, FX_DEFAULT_PARAMS[effectId]);
      });
    });
  }, [audioBridge, decks]);

  useEffect(() => {
    if (!audioBridge) return;
    decks.forEach((deck) => {
      (Object.keys(deck.eqCuts) as EqBandId[]).forEach((band) => {
        audioBridge.setEqCut(deck.id, band, deck.eqCuts[band]);
      });
      const targetStem = deck.stemStatus === "stem" && deck.activeStem ? deck.activeStem : null;
      audioBridge.setStemProfile(deck.id, targetStem);
    });
  }, [audioBridge, decks]);

  useEffect(() => {
    if (!audioBridge) return;
    return audioBridge.subscribeDiagnostics((snapshot: DeckPlaybackDiagnostics) => {
      setDecks((prev) =>
        prev.map((deck) =>
          deck.id === snapshot.deckId
            ? {
                ...deck,
                isPlaying: snapshot.isPlaying,
                currentTimeSeconds: snapshot.currentTimeSeconds,
                durationSeconds: snapshot.durationSeconds ?? deck.durationSeconds ?? null,
                playbackError: snapshot.error,
                vu: snapshot.vu,
                level: snapshot.vu,
              }
            : deck,
        ),
      );
    });
  }, [audioBridge]);

  useEffect(() => {
    return () => {
      loopTimers.current.forEach((timer) => window.clearTimeout(timer));
      loopTimers.current.clear();
      captureObjectUrls.current.forEach((url) => URL.revokeObjectURL(url));
      captureObjectUrls.current.clear();
    };
  }, []);

  const updateLoopSlotStatus = (deckId: DeckId, slotId: string, status: LoopSlot["status"]) => {
    setLoopSlots((prev) => ({
      ...prev,
      [deckId]: prev[deckId].map((slot) => (slot.id === slotId ? { ...slot, status } : slot)),
    }));
  };

  const clearLoopTimersForSlot = (baseKey: string) => {
    loopTimers.current.forEach((timer, key) => {
      if (key.startsWith(baseKey)) {
        window.clearTimeout(timer);
        loopTimers.current.delete(key);
      }
    });
  };

  const registerLoopTimer = (key: string, delay: number, callback: () => void) => {
    const id = window.setTimeout(() => {
      loopTimers.current.delete(key);
      callback();
    }, delay);
    loopTimers.current.set(key, id);
  };

  const handleToggleLoopSlot = (deckId: DeckId, slotId: string) => {
    const slot = loopSlots[deckId]?.find((item) => item.id === slotId);
    if (!slot) return;
    const baseKey = `${deckId}-${slotId}`;
    const deck = decks.find((item) => item.id === deckId);
    if (!deck) return;
    if (slot.status !== "idle") {
      clearLoopTimersForSlot(baseKey);
      updateLoopSlotStatus(deckId, slotId, "idle");
      if (audioBridge) {
        audioBridge.cancelLoopCapture(deckId);
      }
      return;
    }

    clearLoopTimersForSlot(baseKey);
    updateLoopSlotStatus(deckId, slotId, "queued");
    const beatsPerBar = 4;
    const bpm = deck.bpm ?? masterTempo;
    const secondsPerBeat = 60 / Math.max(1, bpm);
    const captureLengthSeconds = secondsPerBeat * (slot.length === "bar" ? beatsPerBar : beatsPerBar / 2);
    const preparation = Math.max(120, Math.round(secondsPerBeat * 1000));

    registerLoopTimer(`${baseKey}:record`, preparation, () => {
      updateLoopSlotStatus(deckId, slotId, "recording");
      if (!audioBridge) {
        registerLoopTimer(`${baseKey}:play`, Math.round(captureLengthSeconds * 1000), () => {
          updateLoopSlotStatus(deckId, slotId, "playing");
        });
        return;
      }
      void (async () => {
        try {
          const { buffer, durationSeconds } = await audioBridge.startLoopCapture(
            deckId,
            captureLengthSeconds,
          );
          const waveform = createWaveformFromAudioBuffer(buffer);
          const wavData = encodeAudioBufferToWav(buffer);
          const blob = new Blob([wavData], { type: "audio/wav" });
          const sourceId = `capture-${deckId}-${Date.now()}`;
          const objectUrl = URL.createObjectURL(blob);
          const previousUrl = captureObjectUrls.current.get(deckId);
          if (previousUrl) {
            URL.revokeObjectURL(previousUrl);
          }
          captureObjectUrls.current.set(deckId, objectUrl);
          const name = deck.loopName ? `${deck.loopName} · ${slot.label}` : `${deckId} ${slot.label}`;
          await audioBridge.loadDeckAudio(deckId, {
            id: sourceId,
            arrayBuffer: wavData,
            objectUrl,
            name,
          });
          setDecks((prev) =>
            prev.map((item) =>
              item.id === deckId
                ? {
                    ...item,
                    loopName: name,
                    waveform,
                    durationSeconds,
                    currentTimeSeconds: 0,
                    playbackError: null,
                    objectUrl,
                    trackId: sourceId,
                    file: undefined,
                  }
                : item,
            ),
          );
          updateLoopSlotStatus(deckId, slotId, "playing");
          addLoop({
            name,
            bpm: Math.round(deck.bpm ?? masterTempo),
            key: deck.tonalKey ?? selectedKey,
            waveform,
            mood: deck.mood ?? "Captured Loop",
            folder: "Custom Imports",
            durationSeconds,
          });
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (message.includes("cancelled")) {
            return;
          }
          updateLoopSlotStatus(deckId, slotId, "idle");
          setDecks((prev) =>
            prev.map((item) =>
              item.id === deckId
                ? {
                    ...item,
                    playbackError: `Loop capture failed: ${message}`,
                    isPlaying: false,
                  }
                : item,
            ),
          );
        }
      })();
    });
  };

  const handleToggleFx = (deckId: DeckId, effectId: DeckFxId) => {
    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === deckId
          ? {
              ...deck,
              fxActive: { ...deck.fxActive, [effectId]: !deck.fxActive[effectId] },
            }
          : deck,
      ),
    );
    const deck = decks.find((item) => item.id === deckId);
    const nextEnabled = !(deck?.fxActive[effectId] ?? false);
    if (audioBridge) {
      audioBridge.setEffectState(deckId, effectId, nextEnabled, FX_DEFAULT_PARAMS[effectId]);
    }
  };

  const handleUpdateDeck = (deckId: DeckId, patch: Partial<DeckPerformance>) => {
    setDecks((prev) => prev.map((deck) => (deck.id === deckId ? { ...deck, ...patch } : deck)));
  };

  const handleNudge = (deckId: DeckId, direction: "forward" | "back") => {
    const delta = direction === "forward" ? 0.06 : -0.06;
    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === deckId
          ? { ...deck, level: clamp(Number((deck.level + delta).toFixed(2)), 0.1, 1) }
          : deck,
      ),
    );
  };

  const handleLoadLoop = (loopId: string) => {
    const loop = storedLoops.find((item) => item.id === loopId);
    if (!loop) return;
    const focusedDeck = decks.find((deck) => deck.isFocused) ?? decks[0];
    if (!focusedDeck) return;

    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === focusedDeck.id
          ? {
              ...deck,
              loopName: loop.name,
              waveform: loop.waveform,
              bpm: loop.bpm,
              tonalKey: loop.key,
              mood: loop.mood,
              filter: clamp(deck.filter + 0.05, 0, 1),
              resonance: clamp(deck.resonance + 0.03, 0, 1),
              zoom: 1,
              activeStem: null,
              queuedStem: null,
              stemStatus: "main",
            }
          : deck,
      ),
    );
    registerLoopLoad(loop.id, focusedDeck.id);
    setSelectorKey(null);
  };

  const handleToggleEq = (deckId: DeckId, band: EqBandId) => {
    const deck = decks.find((item) => item.id === deckId);
    const nextEnabled = deck ? !deck.eqCuts[band] : true;
    setDecks((prev) =>
      prev.map((item) =>
        item.id === deckId
          ? {
              ...item,
              eqCuts: { ...item.eqCuts, [band]: nextEnabled },
            }
          : item,
      ),
    );
    if (audioBridge) {
      audioBridge.setEqCut(deckId, band, nextEnabled);
    }
  };

  const handleTriggerStem = (deckId: DeckId, stem: StemType) => {
    const deck = decks.find((item) => item.id === deckId);
    if (!deck) return;
    const isActive = deck.activeStem === stem && deck.stemStatus === "stem";
    const targetStem: StemType | null = isActive ? null : stem;
    const timerKey = `stem-${deckId}`;
    clearLoopTimersForSlot(timerKey);

    const baseBpm = deck.bpm ?? MASTER_BASE_BPM;
    const stretchedBpm = Math.max(baseBpm * masterTimestretch, 1);
    const measureMs = Math.max(Math.round((60_000 / stretchedBpm) * 4), 400);

    setDecks((prev) =>
      prev.map((item) =>
        item.id === deckId
          ? {
              ...item,
              stemStatus: "queued",
              queuedStem: targetStem,
              level: targetStem ? clamp(item.level * 0.4, 0.04, 0.4) : item.level,
            }
          : item,
      ),
    );

    registerLoopTimer(timerKey, measureMs, () => {
      setDecks((prev) =>
        prev.map((item) => {
          if (item.id !== deckId) return item;
          const updatedStems = item.stems
            ? item.stems.map((stem) => {
                if (!targetStem) {
                  return { ...stem, status: "standby" };
                }
                if (stem.type === targetStem) {
                  return { ...stem, status: "active" };
                }
                return { ...stem, status: "muted" };
              })
            : item.stems;
          const nextLevel = targetStem
            ? clamp(Math.max(item.level, 0.62), 0, 1)
            : clamp(Math.max(item.level, 0.45), 0, 1);
          const updatedStems = item.stems
            ? item.stems.map((entry) => {
                if (targetStem && entry.type === targetStem) {
                  return { ...entry, status: "active" as StemStatus };
                }
                if (targetStem) {
                  return { ...entry, status: "muted" as StemStatus };
                }
                return { ...entry, status: "standby" as StemStatus };
              })
            : item.stems;
          return {
            ...item,
            activeStem: targetStem,
            queuedStem: null,
            stemStatus: targetStem ? "stem" : "main",
            level: nextLevel,
            stems: updatedStems,
          };
        }),
      );
      if (audioBridge) {
        audioBridge.setDeckStemFocus(deckId, targetStem);
      }
    });
  };

  const handleOpenKeySelector = (key: string) => {
    setSelectedKey(key);
    setSelectorKey(key);
  };

  const handleCloseSelector = () => {
    setSelectorKey(null);
  };

  const sortedLoops = useMemo(() => sortLoopsByCamelot(storedLoops), [storedLoops]);

  const selectionLoops = useMemo(
    () =>
      selectorKey
        ? sortedLoops.filter((loop) => loop.key === selectorKey)
        : [],
    [selectorKey, sortedLoops],
  );

  return (
    <>
      <div className="harmoniq-shell">
        <div
          className="harmoniq-shell__frame"
          style={{
            color: theme.text,
            filter: "drop-shadow(0 24px 80px rgba(0, 0, 0, 0.5))",
          }}
        >
          <div className="harmoniq-shell__matrix">
            <DeckMatrix
              decks={decks}
              crossfade={crossfade}
              onCrossfadeChange={setCrossfade}
              onUpdateDeck={handleUpdateDeck}
              onNudge={handleNudge}
              onFocusDeck={handleFocusDeck}
              loopSlots={loopSlots}
              onToggleLoopSlot={handleToggleLoopSlot}
              masterTimestretch={masterTimestretch}
              masterBpm={masterBpm}
              masterTrim={masterTrim}
              onMasterTimestretchChange={setMasterTimestretch}
              onMasterTrimChange={setMasterTrim}
              onToggleEq={handleToggleEq}
              onTriggerStem={handleTriggerStem}
              onToggleFx={handleToggleFx}
              onTogglePlayback={handleTogglePlayback}
              onSeekRatio={handleSeekDeck}
              onRetryPlayback={handleRetryPlayback}
            />
          </div>
          <div className="harmoniq-shell__wheel">
            <div className="harmoniq-shell__wheel-layout">
              <FxRackPanel title="FX Bank Alpha" modules={FX_RACK_PRESETS.left} alignment="left" />
              <HarmonicWheelSelector
                value={selectedKey}
                onChange={setSelectedKey}
                onOpenKey={handleOpenKeySelector}
              />
              <FxRackPanel title="FX Bank Omega" modules={FX_RACK_PRESETS.right} alignment="right" />
            </div>
            <div className="harmoniq-shell__library">
              <div className="harmoniq-shell__library-content">
                <div className="harmoniq-shell__library-upper">
                  <TrackUploadPanel onTracksAnalyzed={handleTracksAnalyzed} />
                  <TrackLibraryList tracks={libraryTracks} onLoad={handleLoadTrackToDeck} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {selectorKey && (
        <TrackSelectionModal
          keyLabel={selectorKey}
          loops={selectionLoops}
          onClose={handleCloseSelector}
          onLoadLoop={handleLoadLoop}
        />
      )}
    </>
  );
}
