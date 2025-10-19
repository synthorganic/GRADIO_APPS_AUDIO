import { useEffect, useMemo, useRef, useState } from "react";
import { theme } from "@daw/theme";
import { DeckMatrix } from "./components/DeckMatrix";
import type { CrossfadeState, DeckId, DeckPerformance, LoopSlot, StemType } from "./types";
import { HarmonicWheelSelector } from "./components/HarmonicWheelSelector";
import { FxRackPanel, type FxModuleConfig } from "./components/FxRackPanel";
import { LibraryFolderSelector } from "./components/LibraryFolderSelector";
import { SavedLoopsLibrary } from "./components/SavedLoopsLibrary";
import { ActiveLoopsPanel } from "./components/ActiveLoopsPanel";
import {
  AutomationEnvelopeList,
  type AutomationEnvelopeSummary,
} from "./components/AutomationEnvelopeList";
import { createWaveform } from "./shared/waveforms";
import { useLoopLibrary } from "./state/LoopLibraryStore";
import { TrackSelectionModal } from "./components/TrackSelectionModal";

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

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

const INITIAL_DECKS: DeckPerformance[] = [
  {
    id: "A",
    loopName: "Neon Skyline",
    waveform: createWaveform(0.8),
    filter: 0.42,
    resonance: 0.4,
    zoom: 1,
    fxStack: ["Delay", "Filter", "Space"],
    isFocused: true,
    level: 0.68,
    bpm: 124,
    tonalKey: "8A",
    mood: "Glasswave",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
  },
  {
    id: "B",
    loopName: "Chromatic Drift",
    waveform: createWaveform(1.3),
    filter: 0.53,
    resonance: 0.5,
    zoom: 1,
    fxStack: ["Flange", "Drive"],
    isFocused: false,
    level: 0.72,
    bpm: 128,
    tonalKey: "2B",
    mood: "Retrograde",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
  },
  {
    id: "C",
    loopName: "Vapor Trails",
    waveform: createWaveform(0.35),
    filter: 0.31,
    resonance: 0.46,
    zoom: 1.1,
    fxStack: ["Phase", "Ducker"],
    isFocused: false,
    level: 0.44,
    bpm: 118,
    tonalKey: "11B",
    mood: "Azure",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
  },
  {
    id: "D",
    loopName: "Night Echo",
    waveform: createWaveform(1.62),
    filter: 0.37,
    resonance: 0.42,
    zoom: 0.92,
    fxStack: ["Reverb", "Crush"],
    isFocused: false,
    level: 0.58,
    bpm: 122,
    tonalKey: "5A",
    mood: "Hypnotic",
    eqCuts: { highs: false, mids: false, lows: false },
    activeStem: null,
    queuedStem: null,
    stemStatus: "main",
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
  const {
    loops: storedLoops,
    automationEnvelopes,
    folders,
    registerLoopLoad,
    exportToFile,
    importFromFile,
    lastPersistedAt,
    persistError,
  } = useLoopLibrary();

  const [decks, setDecks] = useState<DeckPerformance[]>(INITIAL_DECKS);
  const [crossfade, setCrossfade] = useState<CrossfadeState>({ x: 0.45, y: 0.35 });
  const [selectedKey, setSelectedKey] = useState("8A");
  const [loopSlots, setLoopSlots] = useState<Record<DeckId, LoopSlot[]>>(INITIAL_LOOP_SLOTS);
  const [masterTempo, setMasterTempo] = useState(128);
  const [masterPitch, setMasterPitch] = useState(0);
  const [selectedFolder, setSelectedFolder] = useState<string>(() => folders[0] ?? "All Sessions");
  const [mutedDecks, setMutedDecks] = useState<Record<DeckId, boolean>>({
    A: false,
    B: false,
    C: false,
    D: false,
  });
  const [selectorKey, setSelectorKey] = useState<string | null>(null);
  const loopTimers = useRef<Map<string, number>>(new Map());

  const handleTracksAnalyzed = (tracks: AnalyzedTrackSummary[]) => {
    if (!tracks.length) return;
    setLibraryTracks((prev) => {
      const merged = new Map<string, AnalyzedTrackSummary>();
      prev.forEach((item) => {
        merged.set(item.origin, item);
      });
      tracks.forEach((item) => {
        merged.set(item.origin, item);
      });
      return Array.from(merged.values()).sort((a, b) => a.name.localeCompare(b.name));
    });
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
    const waveformSeed = computeSeedFromName(track.name);
    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === deckId
          ? {
              ...deck,
              loopName: track.name,
              waveform: createWaveform(waveformSeed),
              bpm: track.bpm,
              scale: track.scale,
              source: track.origin,
              stems: track.stems.map((stem) => ({
                id: `${track.id}-${stem.id}`,
                label: stem.label,
                status: "standby",
              })),
            }
          : deck,
      ),
    );
    setLoopSlots((prev) => ({
      ...prev,
      [deckId]: prev[deckId]?.map((slot) => ({ ...slot, status: "idle" })) ?? createLoopSlots(deckId),
    }));
    handleFocusDeck(deckId);
  };

  useEffect(() => {
    if (folders.length === 0) return;
    setSelectedFolder((prev) => (folders.includes(prev) ? prev : folders[0]));
  }, [folders]);

  useEffect(() => {
    const timer = setInterval(() => {
      setDecks((prev) =>
        prev.map((deck) => {
          const isMuted = mutedDecks[deck.id] ?? false;
          const floor = isMuted ? 0.02 : 0.15;
          const ceiling = isMuted ? 0.12 : 0.98;
          const jitter = (Math.random() - 0.5) * (isMuted ? 0.04 : 0.12);
          const nextLevel = clamp(deck.level + jitter, floor, ceiling);
          return { ...deck, level: Number(nextLevel.toFixed(2)) };
        }),
      );
    }, 1300);
    return () => clearInterval(timer);
  }, [mutedDecks]);

  useEffect(() => {
    return () => {
      loopTimers.current.forEach((timer) => window.clearTimeout(timer));
      loopTimers.current.clear();
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
    if (slot.status !== "idle") {
      clearLoopTimersForSlot(baseKey);
      updateLoopSlotStatus(deckId, slotId, "idle");
      return;
    }

    clearLoopTimersForSlot(baseKey);
    updateLoopSlotStatus(deckId, slotId, "queued");
    const preparation = slot.length === "bar" ? 1000 : 520;

    registerLoopTimer(`${baseKey}:record`, preparation, () => {
      updateLoopSlotStatus(deckId, slotId, "recording");
      registerLoopTimer(`${baseKey}:play`, 1800, () => {
        updateLoopSlotStatus(deckId, slotId, "playing");
      });
    });
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

  const handleToggleMute = (loopId: string) => {
    const deckId = loopId as DeckId;
    setMutedDecks((prev) => {
      const next = { ...prev, [deckId]: !prev[deckId] };
      setDecks((current) =>
        current.map((deck) =>
          deck.id === deckId
            ? {
                ...deck,
                level: next[deckId] ? 0.05 : Math.max(deck.level, 0.35),
              }
            : deck,
        ),
      );
      return next;
    });
  };

  const handleLoadLoop = (loopId: string) => {
    const loop = storedLoops.find((item) => item.id === loopId);
    if (!loop) return;
    const focusedDeck = decks.find((deck) => deck.isFocused) ?? decks[0];
    if (!focusedDeck) return;

    setMutedDecks((prev) => ({ ...prev, [focusedDeck.id]: false }));
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

  const handleToggleEq = (deckId: DeckId, band: "highs" | "mids" | "lows") => {
    setDecks((prev) =>
      prev.map((deck) =>
        deck.id === deckId
          ? {
              ...deck,
              eqCuts: { ...deck.eqCuts, [band]: !deck.eqCuts[band] },
            }
          : deck,
      ),
    );
  };

  const handleTriggerStem = (deckId: DeckId, stem: StemType) => {
    const deck = decks.find((item) => item.id === deckId);
    if (!deck) return;
    const isActive = deck.activeStem === stem && deck.stemStatus === "stem";
    const targetStem: StemType | null = isActive ? null : stem;
    const timerKey = `stem-${deckId}`;
    clearLoopTimersForSlot(timerKey);

    const bpm = deck.bpm ?? masterTempo;
    const measureMs = Math.max(Math.round((60_000 / Math.max(bpm, 1)) * 4), 400);

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
          const nextLevel = targetStem
            ? clamp(Math.max(item.level, 0.62), 0, 1)
            : clamp(Math.max(item.level, 0.45), 0, 1);
          return {
            ...item,
            activeStem: targetStem,
            queuedStem: null,
            stemStatus: targetStem ? "stem" : "main",
            level: nextLevel,
          };
        }),
      );
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

  const filteredLoops = useMemo(() => {
    if (selectedFolder === "All Sessions") {
      return sortedLoops;
    }
    return sortedLoops.filter((loop) => loop.folder === selectedFolder);
  }, [sortedLoops, selectedFolder]);

  const savedLoopPreview = useMemo(
    () =>
      filteredLoops.map((loop) => ({
        id: loop.id,
        name: loop.name,
        bpm: loop.bpm,
        key: loop.key,
        waveform: loop.waveform,
        mood: loop.mood,
      })),
    [filteredLoops],
  );

  const selectionLoops = useMemo(
    () =>
      selectorKey
        ? sortedLoops.filter((loop) => loop.key === selectorKey)
        : [],
    [selectorKey, sortedLoops],
  );

  const activeLoops = useMemo(() =>
    decks.map((deck) => ({
      id: deck.id,
      name: deck.activeStem ? `${deck.loopName} (${deck.activeStem.toUpperCase()} Stem)` : deck.loopName,
      bpm: deck.bpm ?? masterTempo,
      key: deck.tonalKey ?? selectedKey,
      deck: deck.id,
      waveform: deck.waveform,
      energy: clamp(deck.level + 0.18, 0, 1),
      filter: deck.filter,
      level: deck.level,
      isMuted: mutedDecks[deck.id] ?? false,
    })),
  [decks, masterTempo, selectedKey, mutedDecks]);

  const envelopeSummaries = useMemo<AutomationEnvelopeSummary[]>(
    () =>
      automationEnvelopes.map((envelope) => ({
        ...envelope,
        linkedLoops: storedLoops.filter((loop) => loop.automationIds.includes(envelope.id)),
      })),
    [automationEnvelopes, storedLoops],
  );

  const handleImportLibrary = async (file: File) => {
    await importFromFile(file);
  };

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
            masterTempo={masterTempo}
            masterPitch={masterPitch}
            onMasterTempoChange={setMasterTempo}
            onMasterPitchChange={setMasterPitch}
            onToggleEq={handleToggleEq}
            onTriggerStem={handleTriggerStem}
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
              <LibraryFolderSelector folders={folders} value={selectedFolder} onChange={setSelectedFolder} />
              <div className="harmoniq-shell__library-columns">
                <ActiveLoopsPanel
                  loops={activeLoops}
                  onToggleMute={handleToggleMute}
                  onFocusDeck={handleFocusDeck}
                  highlightKey={selectedKey}
                />
                <SavedLoopsLibrary loops={savedLoopPreview} onLoadLoop={handleLoadLoop} />
                <AutomationEnvelopeList
                  envelopes={envelopeSummaries}
                  onExport={exportToFile}
                  onImport={handleImportLibrary}
                  lastPersistedAt={lastPersistedAt}
                  persistError={persistError}
                />
              </div>
            </div>
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
