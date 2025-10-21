import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useState,
} from "react";
import type { ReactNode } from "react";
import type { DeckId } from "../types";
import {
  createWaveform,
  deserializeWaveform,
  serializeWaveform,
} from "../shared/waveforms";

const STORAGE_KEY = "harmoniq.loop-library.v1";

const DEFAULT_FOLDERS = [
  "All Sessions",
  "Aurora Drift",
  "Midnight Transit",
  "Pulse Archive",
  "Custom Imports",
];

const CAMELOT_KEYS = [
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

const CAMELOT_KEY_SET = new Set(CAMELOT_KEYS);

function detectCamelotKeyFromWaveform(waveform: Float32Array): string {
  if (waveform.length === 0) {
    return CAMELOT_KEYS[0];
  }
  const stride = Math.max(1, Math.floor(waveform.length / 4096));
  let accumulator = 0;
  for (let index = 0; index < waveform.length; index += stride) {
    const sample = waveform[index];
    const weight = (index % 97) + 1;
    accumulator += Math.abs(sample) * weight;
  }
  const normalized = Math.abs(accumulator) * 997;
  const keyIndex = Number.isFinite(normalized)
    ? Math.floor(normalized) % CAMELOT_KEYS.length
    : 0;
  return CAMELOT_KEYS[keyIndex];
}

function ensureCamelotKey(waveform: Float32Array, fallback: string): string {
  if (CAMELOT_KEY_SET.has(fallback)) {
    return fallback;
  }
  return detectCamelotKeyFromWaveform(waveform);
}

export interface AutomationEnvelopePoint {
  beat: number;
  value: number;
  curve: "linear" | "ease-in" | "ease-out";
}

export interface AutomationEnvelope {
  id: string;
  name: string;
  target: string;
  deckId: DeckId | null;
  color: string;
  lengthBeats: number;
  resolution: number;
  points: AutomationEnvelopePoint[];
  createdAt: number;
  updatedAt: number;
}

export interface LoopLibraryItem {
  id: string;
  name: string;
  bpm: number;
  key: string;
  waveform: Float32Array;
  mood: string;
  folder: string;
  createdAt: number;
  updatedAt: number;
  automationIds: string[];
  usageCount: number;
  lastLoadedAt: number | null;
  lastDeckId: DeckId | null;
  durationSeconds: number | null;
}

export type LoopLibraryDraft = {
  id?: string;
  name: string;
  bpm: number;
  key: string;
  waveform: Float32Array;
  mood: string;
  folder?: string;
  automationIds?: string[];
  durationSeconds?: number | null;
};

export type AutomationEnvelopeDraft = {
  id?: string;
  name: string;
  target: string;
  deckId?: DeckId | null;
  color: string;
  lengthBeats: number;
  resolution: number;
  points: AutomationEnvelopePoint[];
};

interface LoopLibraryState {
  loops: Record<string, LoopLibraryItem>;
  automation: Record<string, AutomationEnvelope>;
  folders: string[];
}

interface PersistedLoop extends Omit<LoopLibraryItem, "waveform"> {
  waveform: number[];
}

interface PersistedState {
  version: number;
  savedAt: number;
  folders: string[];
  loops: PersistedLoop[];
  automation: AutomationEnvelope[];
}

type LoopLibraryAction =
  | { type: "add-loop"; loop: LoopLibraryItem }
  | { type: "update-loop"; id: string; patch: Partial<LoopLibraryItem> }
  | { type: "remove-loop"; id: string }
  | { type: "add-automation"; envelope: AutomationEnvelope }
  | { type: "update-automation"; id: string; patch: Partial<AutomationEnvelope> }
  | { type: "remove-automation"; id: string }
  | { type: "link-automation"; loopId: string; envelopeId: string }
  | { type: "unlink-automation"; loopId: string; envelopeId: string }
  | { type: "register-load"; loopId: string; deckId: DeckId }
  | { type: "replace-state"; state: LoopLibraryState };

interface LoopLibraryContextValue {
  loops: LoopLibraryItem[];
  automationEnvelopes: AutomationEnvelope[];
  folders: string[];
  addLoop: (draft: LoopLibraryDraft) => string;
  updateLoop: (id: string, patch: Partial<LoopLibraryItem>) => void;
  removeLoop: (id: string) => void;
  addAutomation: (draft: AutomationEnvelopeDraft) => string;
  updateAutomation: (id: string, patch: Partial<AutomationEnvelope>) => void;
  removeAutomation: (id: string) => void;
  linkAutomation: (loopId: string, envelopeId: string) => void;
  unlinkAutomation: (loopId: string, envelopeId: string) => void;
  registerLoopLoad: (loopId: string, deckId: DeckId) => void;
  importFromSerialized: (serialized: string) => void;
  importFromFile: (file: File) => Promise<void>;
  exportToFile: () => Promise<void>;
  lastPersistedAt: number | null;
  persistError: string | null;
}

function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

function ensureFolders(folders?: string[]): string[] {
  if (!folders || folders.length === 0) {
    return [...DEFAULT_FOLDERS];
  }
  const merged = new Set([...DEFAULT_FOLDERS, ...folders]);
  return Array.from(merged);
}

function normalizeLoopDraft(draft: LoopLibraryDraft, timestamp = Date.now()): LoopLibraryItem {
  return {
    id: draft.id ?? createId("loop"),
    name: draft.name,
    bpm: draft.bpm,
    key: ensureCamelotKey(draft.waveform, draft.key),
    waveform: draft.waveform,
    mood: draft.mood,
    folder: draft.folder ?? DEFAULT_FOLDERS[0],
    createdAt: timestamp,
    updatedAt: timestamp,
    automationIds: Array.from(new Set(draft.automationIds ?? [])),
    usageCount: 0,
    lastLoadedAt: null,
    lastDeckId: null,
    durationSeconds: draft.durationSeconds ?? null,
  };
}

function normalizeEnvelopeDraft(
  draft: AutomationEnvelopeDraft,
  timestamp = Date.now(),
): AutomationEnvelope {
  return {
    id: draft.id ?? createId("env"),
    name: draft.name,
    target: draft.target,
    deckId: draft.deckId ?? null,
    color: draft.color,
    lengthBeats: draft.lengthBeats,
    resolution: draft.resolution,
    points: draft.points,
    createdAt: timestamp,
    updatedAt: timestamp,
  };
}

function createDefaultState(): LoopLibraryState {
  const now = Date.now();
  const envOne = normalizeEnvelopeDraft(
    {
      name: "Neon Filter Sweep",
      target: "Deck A Filter",
      deckId: "A",
      color: "rgba(132, 94, 255, 0.85)",
      lengthBeats: 8,
      resolution: 16,
      points: [
        { beat: 0, value: 0.12, curve: "linear" },
        { beat: 2, value: 0.58, curve: "ease-in" },
        { beat: 4, value: 0.35, curve: "ease-out" },
        { beat: 8, value: 0.62, curve: "linear" },
      ],
    },
    now - 1000 * 60 * 42,
  );
  const envTwo = normalizeEnvelopeDraft(
    {
      name: "Vapor Delay Mix",
      target: "Deck C Delay",
      deckId: "C",
      color: "rgba(120, 203, 220, 0.8)",
      lengthBeats: 16,
      resolution: 32,
      points: [
        { beat: 0, value: 0.18, curve: "ease-in" },
        { beat: 6, value: 0.72, curve: "linear" },
        { beat: 12, value: 0.44, curve: "ease-out" },
        { beat: 16, value: 0.56, curve: "linear" },
      ],
    },
    now - 1000 * 60 * 84,
  );
  const envThree = normalizeEnvelopeDraft(
    {
      name: "Bass Glide",
      target: "Master Pitch",
      deckId: null,
      color: "rgba(255, 148, 241, 0.78)",
      lengthBeats: 4,
      resolution: 8,
      points: [
        { beat: 0, value: 0.22, curve: "linear" },
        { beat: 1.5, value: 0.68, curve: "ease-out" },
        { beat: 4, value: 0.3, curve: "linear" },
      ],
    },
    now - 1000 * 60 * 12,
  );

  const loops = [
    normalizeLoopDraft(
      {
        name: "Neon Skyline",
        bpm: 124,
        key: "8A",
        waveform: createWaveform(0.82),
        mood: "Glasswave",
        folder: DEFAULT_FOLDERS[1],
        automationIds: [envOne.id],
        durationSeconds: 252,
      },
      now - 1000 * 60 * 90,
    ),
    normalizeLoopDraft(
      {
        name: "Chromatic Drift",
        bpm: 128,
        key: "2B",
        waveform: createWaveform(1.18),
        mood: "Retrograde",
        folder: DEFAULT_FOLDERS[2],
        automationIds: [envTwo.id],
        durationSeconds: 236,
      },
      now - 1000 * 60 * 64,
    ),
    normalizeLoopDraft(
      {
        name: "Vapor Trails",
        bpm: 118,
        key: "11B",
        waveform: createWaveform(0.44),
        mood: "Azure",
        folder: DEFAULT_FOLDERS[3],
        automationIds: [envThree.id],
        durationSeconds: 214,
      },
      now - 1000 * 60 * 38,
    ),
    normalizeLoopDraft(
      {
        name: "Night Echo",
        bpm: 122,
        key: "5A",
        waveform: createWaveform(1.62),
        mood: "Hypnotic",
        folder: DEFAULT_FOLDERS[1],
        automationIds: [envOne.id, envTwo.id],
        durationSeconds: 206,
      },
      now - 1000 * 60 * 18,
    ),
  ];

  const loopsRecord = Object.fromEntries(
    loops.map((loop, index) => {
      const created = loop.createdAt;
      const enriched: LoopLibraryItem = {
        ...loop,
        createdAt: created,
        updatedAt: created,
        usageCount: index === 0 ? 3 : 0,
        lastLoadedAt: index === 0 ? now - 1000 * 60 * 6 : null,
        lastDeckId: index === 0 ? "A" : null,
      };
      return [enriched.id, enriched];
    }),
  );

  const automationRecord = Object.fromEntries(
    [envOne, envTwo, envThree].map((envelope) => [envelope.id, envelope]),
  );

  return {
    loops: loopsRecord,
    automation: automationRecord,
    folders: [...DEFAULT_FOLDERS],
  };
}

function serializeState(state: LoopLibraryState): PersistedState {
  return {
    version: 1,
    savedAt: Date.now(),
    folders: [...state.folders],
    loops: Object.values(state.loops).map((loop) => ({
      ...loop,
      waveform: serializeWaveform(loop.waveform),
    })),
    automation: Object.values(state.automation).map((envelope) => ({ ...envelope })),
  };
}

function hydrateState(raw: PersistedState | null): LoopLibraryState {
  if (!raw) {
    return createDefaultState();
  }
  const loops: Record<string, LoopLibraryItem> = {};
  raw.loops.forEach((loop) => {
    const waveform = deserializeWaveform(loop.waveform);
    loops[loop.id] = {
      ...loop,
      key: ensureCamelotKey(waveform, loop.key),
      waveform,
      automationIds: Array.from(new Set(loop.automationIds)),
    };
  });
  const automation: Record<string, AutomationEnvelope> = {};
  raw.automation.forEach((envelope) => {
    automation[envelope.id] = { ...envelope };
  });
  return {
    loops,
    automation,
    folders: ensureFolders(raw.folders),
  };
}

function readPersistedStateWithMeta(): { state: LoopLibraryState; savedAt: number | null } {
  if (typeof window === "undefined") {
    return { state: createDefaultState(), savedAt: null };
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return { state: createDefaultState(), savedAt: null };
    }
    const parsed = JSON.parse(stored) as PersistedState;
    return { state: hydrateState(parsed), savedAt: parsed.savedAt ?? null };
  } catch (error) {
    console.warn("Failed to read Harmoniq loop library from storage", error);
    return { state: createDefaultState(), savedAt: null };
  }
}

const INITIAL_LOAD = readPersistedStateWithMeta();

function initializer(): LoopLibraryState {
  return INITIAL_LOAD.state;
}

function reducer(state: LoopLibraryState, action: LoopLibraryAction): LoopLibraryState {
  switch (action.type) {
    case "add-loop": {
      return {
        ...state,
        loops: {
          ...state.loops,
          [action.loop.id]: action.loop,
        },
      };
    }
    case "update-loop": {
      const existing = state.loops[action.id];
      if (!existing) return state;
      const patch: Partial<LoopLibraryItem> = { ...action.patch };
      const waveform = patch.waveform ?? existing.waveform;
      if (patch.key && !CAMELOT_KEY_SET.has(patch.key)) {
        patch.key = ensureCamelotKey(waveform, patch.key);
      } else if (!patch.key && patch.waveform) {
        patch.key = ensureCamelotKey(waveform, existing.key);
      }
      const next: LoopLibraryItem = {
        ...existing,
        ...patch,
        updatedAt: Date.now(),
      };
      return {
        ...state,
        loops: {
          ...state.loops,
          [action.id]: next,
        },
      };
    }
    case "remove-loop": {
      if (!state.loops[action.id]) return state;
      const { [action.id]: _removed, ...rest } = state.loops;
      return {
        ...state,
        loops: rest,
      };
    }
    case "add-automation": {
      return {
        ...state,
        automation: {
          ...state.automation,
          [action.envelope.id]: action.envelope,
        },
      };
    }
    case "update-automation": {
      const existing = state.automation[action.id];
      if (!existing) return state;
      const next: AutomationEnvelope = {
        ...existing,
        ...action.patch,
        updatedAt: Date.now(),
      };
      return {
        ...state,
        automation: {
          ...state.automation,
          [action.id]: next,
        },
      };
    }
    case "remove-automation": {
      if (!state.automation[action.id]) return state;
      const { [action.id]: _removed, ...rest } = state.automation;
      const loops = Object.fromEntries(
        Object.entries(state.loops).map(([id, loop]) => [
          id,
          {
            ...loop,
            automationIds: loop.automationIds.filter((entry) => entry !== action.id),
          },
        ]),
      );
      return {
        ...state,
        automation: rest,
        loops,
      };
    }
    case "link-automation": {
      const loop = state.loops[action.loopId];
      const automation = state.automation[action.envelopeId];
      if (!loop || !automation) return state;
      if (loop.automationIds.includes(action.envelopeId)) return state;
      const next: LoopLibraryItem = {
        ...loop,
        automationIds: [...loop.automationIds, action.envelopeId],
        updatedAt: Date.now(),
      };
      return {
        ...state,
        loops: {
          ...state.loops,
          [loop.id]: next,
        },
      };
    }
    case "unlink-automation": {
      const loop = state.loops[action.loopId];
      if (!loop) return state;
      if (!loop.automationIds.includes(action.envelopeId)) return state;
      const next: LoopLibraryItem = {
        ...loop,
        automationIds: loop.automationIds.filter((id) => id !== action.envelopeId),
        updatedAt: Date.now(),
      };
      return {
        ...state,
        loops: {
          ...state.loops,
          [loop.id]: next,
        },
      };
    }
    case "register-load": {
      const loop = state.loops[action.loopId];
      if (!loop) return state;
      const next: LoopLibraryItem = {
        ...loop,
        usageCount: loop.usageCount + 1,
        lastLoadedAt: Date.now(),
        lastDeckId: action.deckId,
        updatedAt: Date.now(),
      };
      return {
        ...state,
        loops: {
          ...state.loops,
          [loop.id]: next,
        },
      };
    }
    case "replace-state": {
      return {
        ...action.state,
        folders: ensureFolders(action.state.folders),
      };
    }
    default:
      return state;
  }
}

async function persistState(state: LoopLibraryState) {
  if (typeof window === "undefined") {
    return;
  }
  const serialized = JSON.stringify(serializeState(state));
  window.localStorage.setItem(STORAGE_KEY, serialized);
}

async function triggerDownload(filename: string, contents: string) {
  if (typeof window === "undefined") {
    return;
  }
  const anyWindow = window as unknown as {
    showSaveFilePicker?: (options?: unknown) => Promise<{
      createWritable: () => Promise<{ write: (data: string) => Promise<void>; close: () => Promise<void> }>;
    }>;
  };
  if (typeof anyWindow.showSaveFilePicker === "function") {
    try {
      const handle = await anyWindow.showSaveFilePicker({
        suggestedName: filename,
        types: [
          {
            description: "Harmoniq Loop Library",
            accept: { "application/json": [".json"] },
          },
        ],
      });
      const writable = await handle.createWritable();
      await writable.write(contents);
      await writable.close();
      return;
    } catch (error) {
      console.warn("File save cancelled or unavailable", error);
    }
  }
  if (typeof document === "undefined") return;
  const blob = new Blob([contents], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

const LoopLibraryContext = createContext<LoopLibraryContextValue | null>(null);

export function LoopLibraryProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initializer);
  const [lastPersistedAt, setLastPersistedAt] = useState<number | null>(
    INITIAL_LOAD.savedAt,
  );
  const [persistError, setPersistError] = useState<string | null>(null);

  useEffect(() => {
    persistState(state)
      .then(() => {
        setLastPersistedAt(Date.now());
        setPersistError(null);
      })
      .catch((error) => {
        console.error("Failed to persist Harmoniq library", error);
        setPersistError(error instanceof Error ? error.message : String(error));
      });
  }, [state]);

  const loops = useMemo(() => {
    return Object.values(state.loops).sort((a, b) => b.updatedAt - a.updatedAt);
  }, [state.loops]);

  const automationEnvelopes = useMemo(() => {
    return Object.values(state.automation).sort((a, b) => a.name.localeCompare(b.name));
  }, [state.automation]);

  const addLoop = useCallback(
    (draft: LoopLibraryDraft) => {
      const loop = normalizeLoopDraft(draft);
      dispatch({ type: "add-loop", loop });
      return loop.id;
    },
    [dispatch],
  );

  const updateLoop = useCallback(
    (id: string, patch: Partial<LoopLibraryItem>) => {
      dispatch({ type: "update-loop", id, patch });
    },
    [dispatch],
  );

  const removeLoop = useCallback(
    (id: string) => {
      dispatch({ type: "remove-loop", id });
    },
    [dispatch],
  );

  const addAutomation = useCallback(
    (draft: AutomationEnvelopeDraft) => {
      const envelope = normalizeEnvelopeDraft(draft);
      dispatch({ type: "add-automation", envelope });
      return envelope.id;
    },
    [dispatch],
  );

  const updateAutomation = useCallback(
    (id: string, patch: Partial<AutomationEnvelope>) => {
      dispatch({ type: "update-automation", id, patch });
    },
    [dispatch],
  );

  const removeAutomation = useCallback(
    (id: string) => {
      dispatch({ type: "remove-automation", id });
    },
    [dispatch],
  );

  const linkAutomation = useCallback(
    (loopId: string, envelopeId: string) => {
      dispatch({ type: "link-automation", loopId, envelopeId });
    },
    [dispatch],
  );

  const unlinkAutomation = useCallback(
    (loopId: string, envelopeId: string) => {
      dispatch({ type: "unlink-automation", loopId, envelopeId });
    },
    [dispatch],
  );

  const registerLoopLoad = useCallback(
    (loopId: string, deckId: DeckId) => {
      dispatch({ type: "register-load", loopId, deckId });
    },
    [dispatch],
  );

  const importFromSerialized = useCallback(
    (serialized: string) => {
      try {
        const parsed = JSON.parse(serialized) as PersistedState;
        const next = hydrateState(parsed);
        dispatch({ type: "replace-state", state: next });
        setPersistError(null);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setPersistError(message);
        throw error;
      }
    },
    [dispatch],
  );

  const importFromFile = useCallback(
    async (file: File) => {
      const text = await file.text();
      importFromSerialized(text);
    },
    [importFromSerialized],
  );

  const exportToFile = useCallback(async () => {
    const payload = serializeState(state);
    const contents = JSON.stringify(payload, null, 2);
    await triggerDownload("harmoniq-loop-library.json", contents);
  }, [state]);

  const value = useMemo<LoopLibraryContextValue>(
    () => ({
      loops,
      automationEnvelopes,
      folders: state.folders,
      addLoop,
      updateLoop,
      removeLoop,
      addAutomation,
      updateAutomation,
      removeAutomation,
      linkAutomation,
      unlinkAutomation,
      registerLoopLoad,
      importFromSerialized,
      importFromFile,
      exportToFile,
      lastPersistedAt,
      persistError,
    }),
    [
      loops,
      automationEnvelopes,
      state.folders,
      addLoop,
      updateLoop,
      removeLoop,
      addAutomation,
      updateAutomation,
      removeAutomation,
      linkAutomation,
      unlinkAutomation,
      registerLoopLoad,
      importFromSerialized,
      importFromFile,
      exportToFile,
      lastPersistedAt,
      persistError,
    ],
  );

  return <LoopLibraryContext.Provider value={value}>{children}</LoopLibraryContext.Provider>;
}

export function useLoopLibrary() {
  const ctx = useContext(LoopLibraryContext);
  if (!ctx) {
    throw new Error("useLoopLibrary must be used within LoopLibraryProvider");
  }
  return ctx;
}
