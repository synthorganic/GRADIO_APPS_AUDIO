import {
  createContext,
  createElement,
  useContext,
  useMemo,
  useReducer,
  type Dispatch,
  type ReactNode
} from "react";

export type DeckId = "A" | "B";

export interface DeckLoad {
  id: string;
  title: string;
  artist?: string;
  bpm?: number;
  key?: string;
  artworkUrl?: string;
}

export interface DeckState {
  id: DeckId;
  loop: DeckLoad | null;
}

export interface StemDescriptor {
  id: string;
  name: string;
  muted?: boolean;
  solo?: boolean;
}

export interface StemPerformanceState {
  id: string;
  name: string;
  muted: boolean;
  solo: boolean;
}

export type StemsByDeck = Record<DeckId, Record<string, StemPerformanceState>>;

export interface HarmonicSelectionState {
  currentKey: string | null;
  compatibleKeys: string[];
}

export type LoopFilterValue = string | number | boolean | [number, number];

export type LoopFilterState = Record<string, LoopFilterValue>;

export interface HarmoniqState {
  decks: Record<DeckId, DeckState>;
  stemsByDeck: StemsByDeck;
  crossfader: number;
  harmonicSelection: HarmonicSelectionState;
  loopFilters: LoopFilterState;
}

export type HarmoniqAction =
  | {
      type: "load-deck";
      deckId: DeckId;
      payload: DeckLoad | null;
      stems?: StemDescriptor[];
    }
  | {
      type: "set-stem-mute";
      deckId: DeckId;
      stemId: string;
      muted: boolean;
    }
  | {
      type: "set-stem-solo";
      deckId: DeckId;
      stemId: string;
      solo: boolean;
    }
  | {
      type: "set-crossfader";
      position: number;
    }
  | {
      type: "set-harmonic-selection";
      selection: HarmonicSelectionState | null;
    }
  | {
      type: "set-loop-filter";
      filterId: string;
      value: LoopFilterValue | null;
    };

const initialDeckState: Record<DeckId, DeckState> = {
  A: { id: "A", loop: null },
  B: { id: "B", loop: null }
};

const initialState: HarmoniqState = {
  decks: initialDeckState,
  stemsByDeck: {
    A: {},
    B: {}
  },
  crossfader: 0.5,
  harmonicSelection: {
    currentKey: null,
    compatibleKeys: []
  },
  loopFilters: {}
};

function normalizeStemDescriptors(stems?: StemDescriptor[]): Record<string, StemPerformanceState> {
  if (!stems || stems.length === 0) return {};

  return stems.reduce<Record<string, StemPerformanceState>>((acc, stem) => {
    acc[stem.id] = {
      id: stem.id,
      name: stem.name,
      muted: stem.muted ?? false,
      solo: stem.solo ?? false
    };
    return acc;
  }, {});
}

function harmoniqReducer(state: HarmoniqState, action: HarmoniqAction): HarmoniqState {
  switch (action.type) {
    case "load-deck": {
      const nextDecks: Record<DeckId, DeckState> = {
        ...state.decks,
        [action.deckId]: {
          id: action.deckId,
          loop: action.payload
        }
      };

      const nextStemsByDeck: StemsByDeck = action.payload
        ? {
            ...state.stemsByDeck,
            [action.deckId]:
              action.stems !== undefined
                ? normalizeStemDescriptors(action.stems)
                : state.stemsByDeck[action.deckId]
          }
        : {
            ...state.stemsByDeck,
            [action.deckId]: {}
          };

      return {
        ...state,
        decks: nextDecks,
        stemsByDeck: nextStemsByDeck
      };
    }

    case "set-stem-mute": {
      const deckStems = state.stemsByDeck[action.deckId];
      const targetStem = deckStems?.[action.stemId];
      if (!targetStem) return state;

      return {
        ...state,
        stemsByDeck: {
          ...state.stemsByDeck,
          [action.deckId]: {
            ...deckStems,
            [action.stemId]: {
              ...targetStem,
              muted: action.muted
            }
          }
        }
      };
    }

    case "set-stem-solo": {
      const deckStems = state.stemsByDeck[action.deckId];
      const targetStem = deckStems?.[action.stemId];
      if (!targetStem) return state;

      return {
        ...state,
        stemsByDeck: {
          ...state.stemsByDeck,
          [action.deckId]: {
            ...deckStems,
            [action.stemId]: {
              ...targetStem,
              solo: action.solo
            }
          }
        }
      };
    }

    case "set-crossfader": {
      const clamped = Math.max(0, Math.min(1, action.position));
      if (clamped === state.crossfader) return state;
      return {
        ...state,
        crossfader: clamped
      };
    }

    case "set-harmonic-selection": {
      return {
        ...state,
        harmonicSelection:
          action.selection ?? {
            currentKey: null,
            compatibleKeys: []
          }
      };
    }

    case "set-loop-filter": {
      const nextFilters: LoopFilterState = { ...state.loopFilters };
      if (action.value === null) {
        delete nextFilters[action.filterId];
      } else {
        nextFilters[action.filterId] = action.value;
      }

      return {
        ...state,
        loopFilters: nextFilters
      };
    }

    default:
      return state;
  }
}

interface HarmoniqContextValue extends HarmoniqState {
  dispatch: Dispatch<HarmoniqAction>;
}

const HarmoniqContext = createContext<HarmoniqContextValue | null>(null);

export function HarmoniqProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(harmoniqReducer, initialState);

  const value = useMemo(
    () => ({
      ...state,
      dispatch
    }),
    [state]
  );

  return createElement(HarmoniqContext.Provider, { value }, children);
}

export function useHarmoniqStore() {
  const ctx = useContext(HarmoniqContext);
  if (!ctx) throw new Error("HarmoniqStore must be used inside HarmoniqProvider");
  return ctx;
}

export function useHarmoniqDispatch() {
  const { dispatch } = useHarmoniqStore();
  return dispatch;
}
