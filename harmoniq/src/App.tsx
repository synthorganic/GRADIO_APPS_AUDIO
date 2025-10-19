import { useCallback, useMemo, useState, type CSSProperties } from "react";
import { theme } from "@daw-theme";
import type { FxPluginDescriptor } from "@daw-shared/FxRack";
import { deriveFxPluginType } from "@daw-shared/FxRack";
import { ActiveLoopsPanel, type ActiveLoopDescriptor } from "./components/ActiveLoopsPanel";
import { SavedLoopLibrary, type SavedLoopDescriptor } from "./components/SavedLoopLibrary";
import { DualDeckPanel } from "./components/DualDeckPanel";
import { CrossfaderCore } from "./components/CrossfaderCore";
import { HarmonicWheel } from "./components/HarmonicWheel";

interface DeckPluginsState {
  A: FxPluginDescriptor[];
  B: FxPluginDescriptor[];
}

export function App() {
  const baseLoops = useMemo<ActiveLoopDescriptor[]>(
    () => [
      {
        id: "loop-a",
        title: "Aerial Dawn",
        key: "8A",
        bpm: 124,
        length: "4 Bars",
        energy: 0.72,
        waveform: generateWaveform(320, 4.8, 0.2, 1.2),
        assignedDeck: "A",
      },
      {
        id: "loop-b",
        title: "Neon Run",
        key: "9B",
        bpm: 128,
        length: "8 Bars",
        energy: 0.61,
        waveform: generateWaveform(320, 6.2, 0.32, 2.4),
        assignedDeck: "B",
      },
      {
        id: "loop-c",
        title: "Horizon Glide",
        key: "7A",
        bpm: 120,
        length: "2 Bars",
        energy: 0.54,
        waveform: generateWaveform(320, 3.5, 0.26, 3.8),
        assignedDeck: null,
      },
      {
        id: "loop-d",
        title: "Night Bloom",
        key: "10A",
        bpm: 116,
        length: "1 Bar",
        energy: 0.42,
        waveform: generateWaveform(320, 2.2, 0.18, 4.6),
        assignedDeck: null,
      },
    ],
    [],
  );

  const savedLoops = useMemo<SavedLoopDescriptor[]>(
    () => [
      {
        id: "saved-1",
        title: "Luminous Drift",
        key: "8B",
        mood: "uplifting",
        tags: ["arps", "dawn", "open"],
        waveform: generateWaveform(220, 5.2, 0.28, 6.1),
      },
      {
        id: "saved-2",
        title: "Sable Echo",
        key: "9A",
        mood: "moody",
        tags: ["pad", "deep", "analog"],
        waveform: generateWaveform(220, 4.1, 0.24, 7.4),
      },
      {
        id: "saved-3",
        title: "Glass Horizon",
        key: "7B",
        mood: "driving",
        tags: ["plucks", "upbeat", "drive"],
        waveform: generateWaveform(220, 6.5, 0.31, 8.2),
      },
      {
        id: "saved-4",
        title: "Cobalt Veil",
        key: "6A",
        mood: "mystic",
        tags: ["texture", "fog", "mids"],
        waveform: generateWaveform(220, 5.9, 0.27, 9.1),
      },
    ],
    [],
  );

  const [loops, setLoops] = useState<ActiveLoopDescriptor[]>(baseLoops);
  const [crossfader, setCrossfader] = useState(0);
  const [filters, setFilters] = useState<Record<"A" | "B", number>>({ A: 0.38, B: 0.44 });
  const [focus, setFocus] = useState(0.92);
  const [deckPlugins, setDeckPlugins] = useState<DeckPluginsState>({ A: [], B: [] });
  const [selectedKey, setSelectedKey] = useState("8A");
  const [wheelRotation, setWheelRotation] = useState(0);
  const [wheelEmphasis, setWheelEmphasis] = useState(1);

  const silentWaveform = useMemo(() => new Float32Array([0, 0, 0, 0]), []);

  const handleAssign = useCallback((deck: "A" | "B", loopId: string) => {
    setLoops((current) => {
      const already = current.find((loop) => loop.id === loopId)?.assignedDeck === deck;
      return current.map((loop) => {
        if (loop.id === loopId) {
          return { ...loop, assignedDeck: already ? null : deck };
        }
        if (loop.assignedDeck === deck) {
          return { ...loop, assignedDeck: null };
        }
        return loop;
      });
    });
  }, []);

  const handleFilterChange = useCallback((deck: "A" | "B", value: number) => {
    setFilters((current) => ({ ...current, [deck]: value }));
  }, []);

  const handlePluginLoad = useCallback(
    (deck: "A" | "B") =>
      (file: File | null) => {
        const next: FxPluginDescriptor = {
          id: `${deck}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
          name: file?.name ?? `Quick FX ${deckPlugins[deck].length + 1}`,
          type: file ? deriveFxPluginType(file) : "effect",
          loadedAt: new Date(),
          fileName: file?.name,
        };
        setDeckPlugins((current) => ({
          ...current,
          [deck]: [...current[deck], next],
        }));
      },
    [deckPlugins],
  );

  const decks = useMemo(() => {
    const deckEntries: Array<"A" | "B"> = ["A", "B"];
    return deckEntries.map((deck) => {
      const accent = deck === "A" ? theme.accentBeam[0] : theme.accentBeam[4];
      const assignedLoop = loops.find((loop) => loop.assignedDeck === deck);
      return {
        id: deck,
        loopTitle: assignedLoop?.title ?? `Load Deck ${deck}`,
        key: assignedLoop?.key ?? "--",
        waveform: assignedLoop?.waveform ?? silentWaveform,
        color: accent,
        loudness: assignedLoop?.energy ?? 0.35,
        plugins: deckPlugins[deck],
        onLoadPlugin: handlePluginLoad(deck),
      };
    });
  }, [deckPlugins, handlePluginLoad, loops, silentWaveform]);

  const handleRecall = useCallback(
    (loopId: string) => {
      const loop = savedLoops.find((candidate) => candidate.id === loopId);
      if (!loop) return;
      setSelectedKey(loop.key);
      setWheelEmphasis(1.1);
      setTimeout(() => setWheelEmphasis(1), 320);
      setCrossfader(0);
    },
    [savedLoops],
  );

  const layoutStyle: CSSProperties = useMemo(
    () => ({
      background: `linear-gradient(160deg, ${theme.surfaceRaised} 0%, ${theme.surface} 80%)`,
      borderRadius: "28px",
      padding: "32px 36px",
      border: `1px solid ${theme.border}`,
      boxShadow: theme.shadow,
      color: theme.text,
      backdropFilter: "blur(18px)",
    }),
    [],
  );

  return (
    <div style={layoutStyle}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(280px, 1fr) minmax(360px, 1.2fr) minmax(280px, 1fr)",
          gap: "28px",
          alignItems: "start",
        }}
      >
        <ActiveLoopsPanel loops={loops} onAssign={handleAssign} />
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <DualDeckPanel decks={decks} />
          <CrossfaderCore
            value={crossfader}
            onChange={setCrossfader}
            filters={filters}
            onFilterChange={handleFilterChange}
            focus={focus}
            onFocusChange={setFocus}
          />
          <HarmonicWheel
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
            rotation={wheelRotation}
            onRotationChange={setWheelRotation}
            emphasis={wheelEmphasis}
            onEmphasisChange={setWheelEmphasis}
          />
        </div>
        <SavedLoopLibrary loops={savedLoops} onRecall={handleRecall} />
      </div>
    </div>
  );
}

function generateWaveform(length: number, harmonics: number, variance: number, seed: number) {
  const data = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const t = index / Math.max(1, length - 1);
    const harmonic = Math.sin((t * harmonics + seed) * Math.PI * 2);
    const envelope = 0.4 + 0.6 * Math.sin((t + seed) * Math.PI);
    const noise = Math.sin((index * 7.13 + seed * 17.1) % Math.PI) * variance * 0.5;
    const value = 0.5 + harmonic * 0.35 * envelope + noise;
    data[index] = clamp01(value);
  }
  return data;
}

function clamp01(value: number) {
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}
