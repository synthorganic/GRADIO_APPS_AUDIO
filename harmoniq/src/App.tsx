import { useEffect, useRef, useState } from "react";
import { theme } from "@daw/theme";
import {
  DeckMatrix,
  type DeckPerformance,
  type DeckId,
  type CrossfadeState,
  type LoopSlot,
} from "./components/DeckMatrix";
import { HarmonicWheelSelector } from "./components/HarmonicWheelSelector";
import { FxRackPanel, type FxModuleConfig } from "./components/FxRackPanel";
import { LibraryFolderSelector } from "./components/LibraryFolderSelector";

function createWaveform(seed: number, length = 256) {
  const buffer = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const t = index / length;
    const sine = Math.sin(2 * Math.PI * (seed + 1) * t);
    const warp = Math.sin(2 * Math.PI * (seed * 0.35 + 0.5) * t * 2);
    const value = Math.abs((sine + warp * 0.5) * (0.6 + 0.4 * Math.sin(t * Math.PI * seed)));
    buffer[index] = Math.min(1, Math.max(0, value));
  }
  return buffer;
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
  },
];

function createLoopSlots(deckId: DeckId): LoopSlot[] {
  const labels = [
    "Clip 1",
    "Clip 2",
    "Clip 3",
    "Clip 4",
    "Clip 5",
    "Clip 6",
  ];
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

const LIBRARY_FOLDERS = [
  "All Sessions",
  "Aurora Drift",
  "Midnight Transit",
  "Pulse Archive",
  "Custom Imports",
];

export default function App() {
  const [decks, setDecks] = useState<DeckPerformance[]>(INITIAL_DECKS);
  const [crossfade, setCrossfade] = useState<CrossfadeState>({ x: 0.45, y: 0.35 });
  const [selectedKey, setSelectedKey] = useState("8A");
  const [loopSlots, setLoopSlots] = useState<Record<DeckId, LoopSlot[]>>(INITIAL_LOOP_SLOTS);
  const [masterTempo, setMasterTempo] = useState(128);
  const [masterPitch, setMasterPitch] = useState(0);
  const [selectedFolder, setSelectedFolder] = useState<string>(LIBRARY_FOLDERS[0]);
  const loopTimers = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const timer = setInterval(() => {
      setDecks((prev) =>
        prev.map((deck) => {
          const jitter = (Math.random() - 0.5) * 0.12;
          const nextLevel = clamp(deck.level + jitter, 0.15, 0.98);
          return { ...deck, level: Number(nextLevel.toFixed(2)) };
        }),
      );
    }, 1300);
    return () => clearInterval(timer);
  }, []);

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

  return (
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
          />
        </div>
        <div className="harmoniq-shell__wheel">
          <div className="harmoniq-shell__wheel-layout">
            <FxRackPanel title="FX Bank Alpha" modules={FX_RACK_PRESETS.left} alignment="left" />
            <HarmonicWheelSelector value={selectedKey} onChange={setSelectedKey} />
            <FxRackPanel title="FX Bank Omega" modules={FX_RACK_PRESETS.right} alignment="right" />
          </div>
          <div className="harmoniq-shell__library">
            <LibraryFolderSelector
              folders={LIBRARY_FOLDERS}
              value={selectedFolder}
              onChange={setSelectedFolder}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
