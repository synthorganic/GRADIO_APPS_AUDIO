
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ChangeEvent,
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { nanoid } from "nanoid";
import { useProjectStore } from "../state/ProjectStore";
import type {
  AutomationChannel,
  AutomationPoint,
  MidiBlock,
  MidiChannel,
  MidiNote,
  Project,
  SampleClip,
  TimelineChannel,
} from "../types";
import { audioEngine } from "../lib/audioEngine";
import { useDemucsProcessing } from "../hooks/useDemucsProcessing";
import { theme } from "../theme";
import { sliceSampleSegment } from "../lib/sampleTools";
import { TrackEffectsPanel } from "./TrackEffectsPanel";
import { WaveformPreview } from "../shared/WaveformPreview";
import { describeHeuristics, getStemEngineDefinition } from "../stem_engines";

type SnapResolution = "measure" | "half-measure" | "beat" | "half-beat";

const BASE_MEASURE_WIDTH = 84;
const TIMELINE_PADDING_MEASURES = 8;
const CLIP_HEIGHT = 48;
const CLIP_WAVEFORM_HEIGHT = 28;
const CHANNEL_HEIGHT = 72;
const MAX_CLONE_COUNT = 32;

const MIDI_LOW = 48;
const MIDI_HIGH = 84;
const MIDI_GRID_DIVISOR = 8;
const DEFAULT_BLOCK_SIZE = 1;

const MIN_ZOOM = 0.005;
const MAX_ZOOM = 3;

const SNAP_DIVISORS: Record<SnapResolution, number> = {
  measure: 1,
  "half-measure": 2,
  beat: 4,
  "half-beat": 8,
};

interface TimelineProps {
  project: Project;
  selectedSampleId: string | null;
  onSelectSample: (id: string | null) => void;
}

function measureDurationSeconds(project: Project) {
  return (60 / project.masterBpm) * 4;
}

function quantizePosition(
  value: number,
  secondsPerMeasure: number,
  resolution: SnapResolution,
) {
  const divisor = SNAP_DIVISORS[resolution];
  const measures = value / secondsPerMeasure;
  const snapped = Math.round(measures * divisor) / divisor;
  return Math.max(0, snapped * secondsPerMeasure);
}

function ensurePointRange(points: AutomationPoint[]) {
  if (points.length >= 2) return points;
  if (points.length === 1) {
    return [points[0], { ...points[0], id: nanoid(), time: points[0].time + 1 }];
  }
  return [
    { id: nanoid(), time: 0, value: 0 },
    { id: nanoid(), time: 4, value: 0 },
  ];
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;

function parsePitchName(value?: string | null) {
  if (!value) return null;
  const match = value.trim().match(/^([A-Ga-g])(#|b)?(\d+)?/);
  if (!match) return null;
  const [, base, accidental, octaveString] = match;
  const normalizedBase = base.toUpperCase();
  const accidentalValue = accidental === "#" ? 1 : accidental === "b" ? -1 : 0;
  const baseIndex = NOTE_NAMES.findIndex((name) => name === normalizedBase);
  if (baseIndex === -1) return null;
  const octave = octaveString ? Number(octaveString) : 4;
  if (!Number.isFinite(octave)) return null;
  return 12 * (octave + 1) + baseIndex + accidentalValue;
}

function midiToNoteName(midi: number) {
  const clamped = Math.max(0, Math.min(127, Math.round(midi)));
  const octave = Math.floor(clamped / 12) - 1;
  const name = NOTE_NAMES[clamped % 12];
  return `${name}${octave}`;
}

function deriveHarmonicScale(source?: string | null) {
  if (!source) return "C Major";
  const trimmed = source.trim();
  if (!trimmed) return "C Major";
  const [root, mode] = trimmed.split(/\s+/, 2);
  if (mode) {
    return `C ${mode} (Equivalent of ${trimmed})`;
  }
  return `C Major (Equivalent of ${trimmed})`;
}

function findMidiBlock(blocks: MidiBlock[], position: number) {
  return blocks.find((block) => position >= block.start && position <= block.start + block.length);
}

function sortMidiBlocks(blocks: MidiBlock[]) {
  return [...blocks].sort((a, b) => a.start - b.start);
}

function findAdjacentMidiBlocks(blocks: MidiBlock[], blockId: string) {
  const sorted = sortMidiBlocks(blocks);
  const index = sorted.findIndex((block) => block.id === blockId);
  return {
    previous: index > 0 ? sorted[index - 1] : null,
    next: index >= 0 && index < sorted.length - 1 ? sorted[index + 1] : null,
  } as const;
}

const MIN_NOTE_DURATION = 1e-3;

function clampNotesToLength(notes: MidiNote[], maxLength: number) {
  return notes.reduce<MidiNote[]>((acc, note) => {
    if (note.start >= maxLength - MIN_NOTE_DURATION) {
      return acc;
    }
    const end = note.start + note.length;
    const clampedEnd = Math.min(end, maxLength);
    const adjustedLength = clampedEnd - note.start;
    if (adjustedLength <= MIN_NOTE_DURATION) {
      return acc;
    }
    if (Math.abs(clampedEnd - end) <= MIN_NOTE_DURATION) {
      acc.push(note);
      return acc;
    }
    acc.push({ ...note, length: adjustedLength });
    return acc;
  }, []);
}

function trimNotesFromStart(notes: MidiNote[], delta: number, maxLength: number) {
  if (delta <= 0) {
    return clampNotesToLength(notes, maxLength);
  }
  return notes.reduce<MidiNote[]>((acc, note) => {
    const start = note.start - delta;
    const end = start + note.length;
    if (end <= MIN_NOTE_DURATION) {
      return acc;
    }
    const clampedStart = Math.max(0, start);
    const clampedEnd = Math.min(end, maxLength);
    const length = clampedEnd - clampedStart;
    if (length <= MIN_NOTE_DURATION) {
      return acc;
    }
    acc.push({ ...note, start: clampedStart, length });
    return acc;
  }, []);
}

export function Timeline({ project, selectedSampleId, onSelectSample }: TimelineProps) {
  const { dispatch, currentProjectId, lastControlTarget, preferences } = useProjectStore();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const channelRefs = useRef(new Map<string, HTMLDivElement>());
  const dragState = useRef<{
    sampleId: string;
    pointerId: number;
    originX: number;
    originY: number;
    originPosition: number;
    channelId: string;
  } | null>(null);
  const channelGesture = useRef<
    | {
        type: "clone";
        pointerId: number;
        channelId: string;
        sourceSample: SampleClip;
        baseEnd: number;
        clones: string[];
      }
    | {
        type: "trim";
        pointerId: number;
        channelId: string;
        sampleId: string;
        originalLength: number;
        lastAppliedLength: number;
      }
    | null
  >(null);
  const [snapResolution, setSnapResolution] = useState<SnapResolution>("measure");
  const [playheadPosition, setPlayheadPosition] = useState(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);
  const [effectsSampleId, setEffectsSampleId] = useState<string | null>(null);
  const [openMidiEditor, setOpenMidiEditor] = useState<{
    channelId: string;
    blockId: string;
  } | null>(null);
  const [clipMenu, setClipMenu] = useState<{ sampleId: string; x: number; y: number } | null>(null);
  const [timelineZoom, setTimelineZoom] = useState(1);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [midiMenu, setMidiMenu] = useState<{
    channelId: string;
    blockId: string;
    x: number;
    y: number;
  } | null>(null);
  const midiDragState = useRef<
    | {
        blockId: string;
        pointerId: number;
        originX: number;
        originPosition: number;
        originLength: number;
        channelId: string;
        mode: "move" | "resize-start" | "resize-end";
      }
    | null
  >(null);
  const [isMidiRecording, setIsMidiRecording] = useState(false);
  const [recordPlayhead, setRecordPlayhead] = useState(0);
  const midiAccessRef = useRef<MIDIAccess | null>(null);
  const midiInputsRef = useRef(new Map<string, EventListener>());
  const recordAnimationRef = useRef<number | null>(null);
  const recordStartRef = useRef<number | null>(null);
  const pianoRollDrag = useRef<
    | {
        noteId: string;
        pointerId: number;
        originX: number;
        originY: number;
        originStart: number;
        originPitch: number;
        channelId: string;
        blockId: string;
      }
    | null
  >(null);
  const pianoRollRef = useRef<HTMLDivElement | null>(null);
  const [defaultNoteLengthSteps, setDefaultNoteLengthSteps] = useState(2);
  const [timelineHoverInfo, setTimelineHoverInfo] = useState<
    { sampleId: string; rect: DOMRect } | null
  >(null);

  const findMidiChannelById = useCallback(
    (channelId: string) =>
      project.channels.find(
        (item): item is MidiChannel => item.id === channelId && item.type === "midi",
      ) ?? null,
    [project.channels],
  );

  const setMidiBlocks = useCallback(
    (channelId: string, nextBlocks: MidiBlock[]) => {
      dispatch({
        type: "update-channel",
        projectId: currentProjectId,
        channelId,
        patch: { blocks: sortMidiBlocks(nextBlocks) },
      });
    },
    [currentProjectId, dispatch],
  );

  const updateMidiBlock = useCallback(
    (channelId: string, blockId: string, updater: (block: MidiBlock) => MidiBlock) => {
      const channel = findMidiChannelById(channelId);
      if (!channel) return;
      const nextBlocks = channel.blocks.map((block) =>
        block.id === blockId ? updater(block) : block,
      );
      setMidiBlocks(channelId, nextBlocks);
    },
    [findMidiChannelById, setMidiBlocks],
  );

  const updateProjectScale = useCallback(
    (source?: string | null) => {
      dispatch({
        type: "set-scale",
        projectId: currentProjectId,
        scale: deriveHarmonicScale(source),
      });
    },
    [currentProjectId, dispatch],
  );

  const setMidiBlockSize = useCallback(
    (channelId: string, value: number) => {
      const numeric = Number.isFinite(value) ? value : DEFAULT_BLOCK_SIZE;
      const size = Math.max(0.5, Math.min(16, Math.round(numeric * 2) / 2 || DEFAULT_BLOCK_SIZE));
      dispatch({
        type: "update-channel",
        projectId: currentProjectId,
        channelId,
        patch: { blockSizeMeasures: size },
      });
    },
    [currentProjectId, dispatch],
  );

  const removeMidiBlock = useCallback(
    (channelId: string, blockId: string) => {
      const channel = findMidiChannelById(channelId);
      if (!channel) return;
      const nextBlocks = channel.blocks.filter((block) => block.id !== blockId);
      setMidiBlocks(channelId, nextBlocks);
      if (openMidiEditor?.channelId === channelId && openMidiEditor.blockId === blockId) {
        setOpenMidiEditor(null);
      }
    },
    [findMidiChannelById, openMidiEditor, setMidiBlocks],
  );

  const addNoteToBlock = useCallback(
    (channelId: string, blockId: string, note: MidiNote) => {
      updateMidiBlock(channelId, blockId, (block) => ({ ...block, notes: [...block.notes, note] }));
    },
    [updateMidiBlock],
  );

  const removeNoteFromBlock = useCallback(
    (channelId: string, blockId: string, noteId: string) => {
      updateMidiBlock(channelId, blockId, (block) => ({
        ...block,
        notes: block.notes.filter((item) => item.id !== noteId),
      }));
    },
    [updateMidiBlock],
  );

  const midiEditorContext = useMemo(() => {
    if (!openMidiEditor) return null;
    const channel = findMidiChannelById(openMidiEditor.channelId);
    if (!channel) return null;
    const block = channel.blocks.find((item) => item.id === openMidiEditor.blockId);
    if (!block) return null;
    return { channel, block };
  }, [findMidiChannelById, openMidiEditor]);

  const demucsOptions = useMemo(
    () => ({ engine: preferences.stemEngine, heuristics: preferences.heuristics }),
    [preferences.heuristics, preferences.stemEngine],
  );

  const { processSample } = useDemucsProcessing((updated) => {
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: updated.id,
      sample: updated,
    });
  }, demucsOptions);

  const engineDefinition = useMemo(
    () => getStemEngineDefinition(preferences.stemEngine),
    [preferences.stemEngine],
  );

  const heuristicSummary = useMemo(
    () => describeHeuristics(engineDefinition, preferences.heuristics),
    [engineDefinition, preferences.heuristics],
  );

  const updateViewportWidth = useCallback(() => {
    const current = scrollRef.current;
    if (!current) return;
    setViewportWidth(current.clientWidth);
  }, []);

  useEffect(() => {
    updateViewportWidth();
    if (typeof window === "undefined") return;
    const container = scrollRef.current;
    if (!container) return;

    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (!entry) return;
        setViewportWidth(entry.contentRect.width);
      });
      observer.observe(container);
      return () => {
        observer.disconnect();
      };
    }

    window.addEventListener("resize", updateViewportWidth);
    return () => {
      window.removeEventListener("resize", updateViewportWidth);
    };
  }, [updateViewportWidth]);

  const secondsPerMeasure = useMemo(
    () => measureDurationSeconds(project),
    [project.masterBpm],
  );

  const measureWidth = useMemo(
    () => BASE_MEASURE_WIDTH * timelineZoom,
    [timelineZoom],
  );

  const halfMeasureDuration = useMemo(
    () => secondsPerMeasure / 2,
    [secondsPerMeasure],
  );

  const midiGridDuration = useMemo(
    () => secondsPerMeasure / MIDI_GRID_DIVISOR,
    [secondsPerMeasure],
  );

  const defaultNoteLength = useMemo(
    () => Math.max(1, defaultNoteLengthSteps) * midiGridDuration,
    [defaultNoteLengthSteps, midiGridDuration],
  );

  const defaultNoteLengthBeats = useMemo(
    () => (defaultNoteLength / secondsPerMeasure) * 4,
    [defaultNoteLength, secondsPerMeasure],
  );

  const incrementNoteLength = useCallback(() => {
    setDefaultNoteLengthSteps((value) => Math.min(32, value + 1));
  }, []);

  const decrementNoteLength = useCallback(() => {
    setDefaultNoteLengthSteps((value) => Math.max(1, value - 1));
  }, []);

  const handleNoteLengthInput = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const next = Number.parseInt(event.target.value, 10);
    if (!Number.isFinite(next)) return;
    setDefaultNoteLengthSteps(Math.max(1, Math.min(32, next)));
  }, []);

  useEffect(() => {
    const defaultChannel = project.channels.find((channel) => channel.type === "audio");
    if (!defaultChannel) return;
    const unassigned = project.samples.filter(
      (sample) => sample.isInTimeline !== false && !sample.channelId,
    );
    if (unassigned.length === 0) return;
    unassigned.forEach((sample) => {
      dispatch({
        type: "update-sample",
        projectId: currentProjectId,
        sampleId: sample.id,
        sample: { channelId: defaultChannel.id },
      });
    });
  }, [currentProjectId, dispatch, project.channels, project.samples]);

  const timelineSamples = useMemo(
    () => project.samples.filter((sample) => sample.isInTimeline !== false),
    [project.samples],
  );

  const totalDuration = useMemo(() => {
    const maxEnd = timelineSamples.reduce(
      (acc, sample) => Math.max(acc, sample.position + sample.length),
      0,
    );
    return Math.max(maxEnd, secondsPerMeasure * 4);
  }, [secondsPerMeasure, timelineSamples]);

  const timelineMeasures = useMemo(
    () =>
      Math.max(totalDuration / secondsPerMeasure + TIMELINE_PADDING_MEASURES, 4 + TIMELINE_PADDING_MEASURES),
    [secondsPerMeasure, totalDuration],
  );

  const paddedDuration = useMemo(
    () => timelineMeasures * secondsPerMeasure,
    [timelineMeasures, secondsPerMeasure],
  );

  const timelineWidth = useMemo(
    () => timelineMeasures * measureWidth,
    [measureWidth, timelineMeasures],
  );

  const fitZoom = useMemo(() => {
    if (viewportWidth <= 0) return MIN_ZOOM;
    const baseWidth = timelineMeasures * BASE_MEASURE_WIDTH;
    if (baseWidth <= 0) return MIN_ZOOM;
    const ratio = viewportWidth / baseWidth;
    if (!Number.isFinite(ratio) || ratio <= 0) return MIN_ZOOM;
    return ratio;
  }, [timelineMeasures, viewportWidth]);

  const minimumZoom = useMemo(
    () => Math.min(MIN_ZOOM, fitZoom),
    [fitZoom],
  );

  const clampZoom = useCallback(
    (value: number) =>
      Math.min(MAX_ZOOM, Math.max(minimumZoom, Number.isFinite(value) ? value : minimumZoom)),
    [minimumZoom],
  );

  useEffect(() => {
    setTimelineZoom((prev) => clampZoom(prev));
  }, [clampZoom]);

  const handleZoomChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setTimelineZoom(clampZoom(Number(event.target.value)));
    },
    [clampZoom],
  );

  const handleZoomToFit = useCallback(() => {
    if (viewportWidth <= 0) return;
    const baseWidth = timelineMeasures * BASE_MEASURE_WIDTH;
    if (baseWidth <= 0) return;
    const ratio = viewportWidth / baseWidth;
    if (!Number.isFinite(ratio) || ratio <= 0) return;
    const nextZoom = ratio > 1 ? 1 : ratio;
    setTimelineZoom(clampZoom(nextZoom));
  }, [clampZoom, timelineMeasures, viewportWidth]);

  const zoomLabel = useMemo(() => {
    const percentage = timelineZoom * 100;
    if (!Number.isFinite(percentage)) return "100%";
    if (percentage >= 10) return `${Math.round(percentage)}%`;
    return `${percentage.toFixed(1)}%`;
  }, [timelineZoom]);

  const arrangementClips = useMemo(
    () =>
      timelineSamples.filter((sample) => {
        if (!sample.channelId) return false;
        const channel = project.channels.find((item) => item.id === sample.channelId);
        return channel?.type === "audio" && channel.isFxEnabled !== false;
      }),
    [project.channels, timelineSamples],
  );

  const channelSamples = useMemo(() => {
    const map = new Map<string, SampleClip[]>();
    project.channels.forEach((channel) => {
      if (channel.type === "audio") {
        map.set(
          channel.id,
          timelineSamples
            .filter((sample) => sample.channelId === channel.id)
            .sort((a, b) => a.position - b.position),
        );
      }
    });
    return map;
  }, [project.channels, timelineSamples]);

  const timelineGrid = useMemo(() => {
    const divisor = SNAP_DIVISORS[snapResolution];
    const ticks: Array<{ position: number; accent: number }> = [];
    const totalTicks = Math.ceil(timelineMeasures * divisor);
    for (let index = 0; index < totalTicks; index += 1) {
      ticks.push({ position: index / divisor, accent: index % divisor });
    }
    return ticks;
  }, [snapResolution, timelineMeasures]);

  useEffect(() => {
    const handlePlay = (event: Event) => {
      const detail = (event as CustomEvent<{ timelineOffset?: number }>).detail;
      setIsTimelinePlaying(true);
      setPlayheadPosition(detail?.timelineOffset ?? 0);
    };
    const handleStop = () => {
      setIsTimelinePlaying(false);
    };
    const handleTick = (event: Event) => {
      const detail = (event as CustomEvent<{ timelineOffset: number; position: number }>).detail;
      setPlayheadPosition(detail.timelineOffset + detail.position);
    };

    window.addEventListener("audio-play", handlePlay as EventListener);
    window.addEventListener("audio-stop", handleStop as EventListener);
    window.addEventListener("audio-tick", handleTick as EventListener);

    return () => {
      window.removeEventListener("audio-play", handlePlay as EventListener);
      window.removeEventListener("audio-stop", handleStop as EventListener);
      window.removeEventListener("audio-tick", handleTick as EventListener);
    };
  }, []);

  useEffect(() => {
    if (!effectsSampleId) return;
    const exists = project.samples.some((sample) => sample.id === effectsSampleId);
    if (!exists) {
      setEffectsSampleId(null);
    }
  }, [effectsSampleId, project.samples]);

  useEffect(() => {
    const closeMenu = () => setClipMenu(null);
    window.addEventListener("click", closeMenu);
    return () => {
      window.removeEventListener("click", closeMenu);
    };
  }, []);

  useEffect(() => {
    if (!clipMenu) return;
    const exists = project.samples.some((sample) => sample.id === clipMenu.sampleId);
    if (!exists) {
      setClipMenu(null);
    }
  }, [clipMenu, project.samples]);

  useEffect(() => {
    const closeMidiMenu = () => setMidiMenu(null);
    window.addEventListener("click", closeMidiMenu);
    return () => {
      window.removeEventListener("click", closeMidiMenu);
    };
  }, []);

  useEffect(() => {
    if (!timelineHoverInfo) return;
    const exists = project.samples.some(
      (sample) => sample.id === timelineHoverInfo.sampleId && sample.isInTimeline !== false,
    );
    if (!exists) {
      setTimelineHoverInfo(null);
    }
  }, [project.samples, timelineHoverInfo]);

  useEffect(() => {
    if (!midiMenu) return;
    const channel = findMidiChannelById(midiMenu.channelId);
    const exists = channel?.blocks.some((block) => block.id === midiMenu.blockId);
    if (!exists) {
      setMidiMenu(null);
    }
  }, [findMidiChannelById, midiMenu]);

  const handleMidiMessage = useCallback(
    (event: MIDIMessageEvent) => {
      const data = event.data;
      if (!data || data.length < 3) return;
      const status = data[0];
      const noteNumber = data[1];
      const velocity = data[2];
      const command = status & 0xf0;
      if (!midiEditorContext) return;
      const { channel: midiChannel, block: midiBlock } = midiEditorContext;
      const sample = midiBlock.sampleId
        ? project.samples.find((item) => item.id === midiBlock.sampleId)
        : null;

      if (command === 0x90 && velocity > 0) {
        if (sample) {
          const baseMidi = midiBlock.baseMidi ?? 60;
          void audioEngine.triggerOneShot(sample, noteNumber - baseMidi);
        }

        if (isMidiRecording) {
          const gridStart = Math.max(
            0,
            Math.round(recordPlayhead / midiGridDuration) * midiGridDuration,
          );
          const note: MidiNote = {
            id: nanoid(),
            blockId: midiBlock.id,
            start: Math.min(midiBlock.length, gridStart),
            length: Math.max(
              midiGridDuration,
              midiBlock.length > 0
                ? Math.min(midiGridDuration, midiBlock.length - gridStart)
                : midiGridDuration,
            ),
            pitch: noteNumber,
            velocity: Math.min(1, velocity / 127),
            sampleId: midiBlock.sampleId,
          };
          addNoteToBlock(midiChannel.id, midiBlock.id, note);
        }
      }
    },
    [
      addNoteToBlock,
      isMidiRecording,
      midiEditorContext,
      midiGridDuration,
      project.samples,
      recordPlayhead,
    ],
  );

  useEffect(() => {
    if (!midiEditorContext) {
      midiInputsRef.current.forEach((listener, id) => {
        const access = midiAccessRef.current;
        const input = access?.inputs.get(id);
        if (input && listener) {
          input.removeEventListener("midimessage", listener as EventListener);
        }
      });
      midiInputsRef.current.clear();
      midiAccessRef.current = null;
      setIsMidiRecording(false);
      return;
    }

    if (typeof navigator === "undefined" || typeof navigator.requestMIDIAccess !== "function") {
      return;
    }

    let cancelled = false;

    void navigator.requestMIDIAccess().then((access) => {
      if (cancelled) return;
      midiAccessRef.current = access;

      const attachInput = (input: MIDIInput) => {
        const handler = (event: Event) => handleMidiMessage(event as MIDIMessageEvent);
        input.addEventListener("midimessage", handler);
        midiInputsRef.current.set(input.id, handler);
      };

      access.inputs.forEach((input) => attachInput(input));

      access.onstatechange = () => {
        midiInputsRef.current.forEach((_, id) => {
          const existing = access.inputs.get(id);
          if (!existing) {
            midiInputsRef.current.delete(id);
          }
        });
        access.inputs.forEach((input) => {
          if (!midiInputsRef.current.has(input.id)) {
            attachInput(input);
          }
        });
      };
    });

    return () => {
      cancelled = true;
      midiInputsRef.current.forEach((listener, id) => {
        const access = midiAccessRef.current;
        const input = access?.inputs.get(id);
        if (input && listener) {
          input.removeEventListener("midimessage", listener as EventListener);
        }
      });
      midiInputsRef.current.clear();
    };
  }, [handleMidiMessage, midiEditorContext]);

  useEffect(() => {
    if (!isMidiRecording || !midiEditorContext) {
      if (recordAnimationRef.current !== null) {
        cancelAnimationFrame(recordAnimationRef.current);
      }
      recordAnimationRef.current = null;
      recordStartRef.current = null;
      setRecordPlayhead(0);
      return;
    }

    const update = () => {
      const now = performance.now();
      if (recordStartRef.current === null) {
        recordStartRef.current = now;
      }
      const elapsedSeconds = (now - recordStartRef.current) / 1000;
      const blockLength = Math.max(0.0001, midiEditorContext.block.length);
      setRecordPlayhead(elapsedSeconds % blockLength);
      recordAnimationRef.current = requestAnimationFrame(update);
    };

    recordAnimationRef.current = requestAnimationFrame(update);

    return () => {
      if (recordAnimationRef.current !== null) {
        cancelAnimationFrame(recordAnimationRef.current);
      }
      recordAnimationRef.current = null;
      recordStartRef.current = null;
      setRecordPlayhead(0);
    };
  }, [isMidiRecording, midiEditorContext]);

  useEffect(() => {
    let cancelled = false;
    const pending = timelineSamples.filter((sample) => (sample.url || sample.file) && !sample.waveform);
    pending.forEach((sample) => {
      void (async () => {
        const peaks = await audioEngine.getWaveformPeaks(sample, 320);
        if (!peaks || cancelled) return;
        dispatch({
          type: "update-sample",
          projectId: currentProjectId,
          sampleId: sample.id,
          sample: { waveform: peaks },
        });
      })();
    });
    return () => {
      cancelled = true;
    };
  }, [currentProjectId, dispatch, timelineSamples]);

  const projectDropPosition = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const container = scrollRef.current;
      if (!container) return 0;
      const bounds = container.getBoundingClientRect();
      const relativeX = event.clientX - bounds.left + container.scrollLeft;
      const measuresFromStart = Math.max(0, relativeX / measureWidth);
      const divisor = SNAP_DIVISORS[snapResolution];
      const snappedMeasures = Math.round(measuresFromStart * divisor) / divisor;
      return snappedMeasures * secondsPerMeasure;
    },
    [measureWidth, secondsPerMeasure, snapResolution],
  );

  const pointerTimelinePosition = useCallback(
    (clientX: number) => {
      const container = scrollRef.current;
      if (!container) return 0;
      const bounds = container.getBoundingClientRect();
      const relativeX = clientX - bounds.left + container.scrollLeft;
      const measuresFromStart = Math.max(0, relativeX / measureWidth);
      return measuresFromStart * secondsPerMeasure;
    },
    [measureWidth, secondsPerMeasure],
  );


  const duplicateClip = useCallback(
    (sample: SampleClip) => {
      if (!sample.channelId) return;
      const nextPosition = quantizePosition(
        sample.position + sample.length,
        secondsPerMeasure,
        snapResolution,
      );
      const fragment = sliceSampleSegment(sample, 0, sample.length, {
        position: nextPosition,
        isFragment: sample.isFragment,
        isInTimeline: true,
        channelId: sample.channelId,
      });
      dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
      onSelectSample(fragment.id);
    },
    [currentProjectId, dispatch, onSelectSample, secondsPerMeasure, snapResolution],
  );

  const handleChannelPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channel: TimelineChannel) => {
      if (channel.type === "midi") {
        handleMidiLanePointerDown(event, channel);
        return;
      }
      if (channel.type !== "audio") return;
      const target = event.target as HTMLElement;
      if (target.closest('[data-clip-id]')) return;
      const samples = channelSamples.get(channel.id);
      if (!samples || samples.length === 0) return;

      if (event.button === 0) {
        const lastSample = samples[samples.length - 1];
        if (!lastSample.channelId) return;
        const pointerTime = pointerTimelinePosition(event.clientX);
        const baseEnd = lastSample.position + lastSample.length;
        if (pointerTime < baseEnd) return;
        channelGesture.current = {
          type: "clone",
          pointerId: event.pointerId,
          channelId: channel.id,
          sourceSample: lastSample,
          baseEnd,
          clones: [],
        };
        event.currentTarget.setPointerCapture(event.pointerId);
      }

      if (event.button === 2) {
        event.preventDefault();
        const pointerTime = pointerTimelinePosition(event.clientX);
        const candidate = [...samples]
          .reverse()
          .find((sample) => pointerTime >= sample.position + sample.length - 1e-3);
        if (!candidate) return;
        const candidateEnd = candidate.position + candidate.length;
        if (pointerTime < candidateEnd) return;
        channelGesture.current = {
          type: "trim",
          pointerId: event.pointerId,
          channelId: channel.id,
          sampleId: candidate.id,
          originalLength: candidate.length,
          lastAppliedLength: candidate.length,
        };
        event.currentTarget.setPointerCapture(event.pointerId);
      }
    },
    [channelSamples, pointerTimelinePosition],
  );

  const handleChannelPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channel: TimelineChannel) => {
      const gesture = channelGesture.current;
      if (!gesture || gesture.pointerId !== event.pointerId || gesture.channelId !== channel.id)
        return;

      if (gesture.type === "clone") {
        const pointerTime = pointerTimelinePosition(event.clientX);
        const delta = pointerTime - gesture.baseEnd;
        const unit = gesture.sourceSample.length;
        if (unit <= 0) return;
        const desiredCount = Math.min(
          MAX_CLONE_COUNT,
          Math.max(0, Math.floor(delta / unit)),
        );

        if (desiredCount < gesture.clones.length) {
          const toRemove = gesture.clones.splice(desiredCount);
          toRemove.forEach((sampleId) =>
            dispatch({ type: "remove-sample", projectId: currentProjectId, sampleId }),
          );
        }

        while (gesture.clones.length < desiredCount) {
          const position = gesture.baseEnd + gesture.clones.length * unit;
          const fragment = sliceSampleSegment(gesture.sourceSample, 0, unit, {
            position,
            isFragment: gesture.sourceSample.isFragment,
            isInTimeline: true,
            channelId: gesture.sourceSample.channelId,
          });
          gesture.clones.push(fragment.id);
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
        }
      }

      if (gesture.type === "trim") {
        const sample = project.samples.find((item) => item.id === gesture.sampleId);
        if (!sample) return;
        const sampleEnd = sample.position + gesture.originalLength;
        const pointerTime = Math.min(pointerTimelinePosition(event.clientX), sampleEnd);
        const gridSize = secondsPerMeasure / SNAP_DIVISORS[snapResolution];
        const delta = Math.max(0, sampleEnd - pointerTime);
        const steps = Math.floor(delta / gridSize);
        const minimumLength = Math.min(gesture.originalLength, gridSize);
        const newLength = Math.max(minimumLength, gesture.originalLength - steps * gridSize);
        if (Math.abs(newLength - gesture.lastAppliedLength) < 1e-3) return;

        const measures = sample.measures
          .filter((measure) => measure.start < newLength)
          .map((measure) => {
            const end = Math.min(measure.end, newLength);
            const beats = measure.beats
              ?.filter((beat) => beat.start < newLength)
              .map((beat) => ({
                ...beat,
                end: Math.min(beat.end, newLength),
              }));
            return { ...measure, end, beats };
          });

        dispatch({
          type: "update-sample",
          projectId: currentProjectId,
          sampleId: sample.id,
          sample: { length: newLength, duration: newLength, measures },
        });
        gesture.lastAppliedLength = newLength;
      }
    },
    [currentProjectId, dispatch, pointerTimelinePosition, project.samples, secondsPerMeasure, snapResolution],
  );

  const handleChannelPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const gesture = channelGesture.current;
      if (!gesture || gesture.pointerId !== event.pointerId) return;
      channelGesture.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
    },
    [],
  );

  const handleMidiBlockPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channel: MidiChannel, block: MidiBlock) => {
      event.stopPropagation();
      const container = scrollRef.current;
      if (!container) return;
      const target = event.target as HTMLElement;
      let mode: "move" | "resize-start" | "resize-end" = "move";
      const resizeHandle = target.getAttribute("data-resize");
      if (resizeHandle === "start") mode = "resize-start";
      if (resizeHandle === "end") mode = "resize-end";
      midiDragState.current = {
        blockId: block.id,
        pointerId: event.pointerId,
        originX: event.clientX + container.scrollLeft,
        originPosition: block.start,
        originLength: block.length,
        channelId: channel.id,
        mode,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const handleMidiBlockPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channel: MidiChannel, block: MidiBlock) => {
      const state = midiDragState.current;
      if (!state || state.pointerId !== event.pointerId || state.blockId !== block.id) return;
      const container = scrollRef.current;
      if (!container) return;

      const channelState = findMidiChannelById(channel.id);
      const blockState = channelState?.blocks.find((item) => item.id === block.id);
      if (!channelState || !blockState) return;

      const { previous, next } = findAdjacentMidiBlocks(channelState.blocks, blockState.id);
      const cursor = event.clientX + container.scrollLeft;
      const delta = cursor - state.originX;
      const measuresDelta = delta / measureWidth;

      if (state.mode === "move") {
        const measurePosition = state.originPosition / secondsPerMeasure + measuresDelta;
        const snappedMeasures = Math.round(measurePosition * 2) / 2;
        const blockLength = blockState.length;
        const minStart = previous ? previous.start + previous.length : 0;
        const maxStart = next ? next.start - blockLength : Number.POSITIVE_INFINITY;
        if (maxStart < minStart) return;
        let newStart = snappedMeasures * secondsPerMeasure;
        if (!Number.isFinite(newStart)) return;
        newStart = Math.min(Math.max(newStart, minStart), maxStart);
        newStart = Math.max(0, newStart);
        if (Math.abs(newStart - blockState.start) < 1e-3) return;
        updateMidiBlock(channelState.id, blockState.id, (current) => ({ ...current, start: newStart }));
        return;
      }

      if (state.mode === "resize-start") {
        const originEnd = state.originPosition + state.originLength;
        const candidate = state.originPosition + measuresDelta * secondsPerMeasure;
        let snappedMeasures = Math.round((candidate / secondsPerMeasure) * 2) / 2;
        let snappedStart = snappedMeasures * secondsPerMeasure;
        if (!Number.isFinite(snappedStart)) return;
        const maxStart = originEnd - halfMeasureDuration;
        snappedStart = Math.min(snappedStart, maxStart);
        const minStart = previous ? previous.start + previous.length : 0;
        snappedStart = Math.max(minStart, Math.max(0, snappedStart));
        const newLength = Math.max(halfMeasureDuration, originEnd - snappedStart);
        const deltaStart = snappedStart - blockState.start;
        if (
          Math.abs(deltaStart) < 1e-3 &&
          Math.abs(newLength - blockState.length) < 1e-3
        ) {
          return;
        }
        updateMidiBlock(channelState.id, blockState.id, (current) => {
          const trimAmount = Math.max(0, snappedStart - current.start);
          const notes =
            trimAmount > 0
              ? trimNotesFromStart(current.notes, trimAmount, newLength)
              : clampNotesToLength(current.notes, newLength);
          return {
            ...current,
            start: snappedStart,
            length: newLength,
            notes,
          };
        });
        return;
      }

      const pointerTime = pointerTimelinePosition(event.clientX);
      let snappedMeasures = Math.round((pointerTime / secondsPerMeasure) * 2) / 2;
      let snappedEnd = snappedMeasures * secondsPerMeasure;
      if (!Number.isFinite(snappedEnd)) return;
      const minEnd = blockState.start + halfMeasureDuration;
      const maxEnd = next ? next.start : Number.POSITIVE_INFINITY;
      if (maxEnd <= minEnd) {
        return;
      }
      snappedEnd = Math.min(Math.max(snappedEnd, minEnd), maxEnd);
      const newLength = Math.max(halfMeasureDuration, snappedEnd - blockState.start);
      if (Math.abs(newLength - blockState.length) < 1e-3) return;
      updateMidiBlock(channelState.id, blockState.id, (current) => ({
        ...current,
        length: newLength,
        notes: clampNotesToLength(current.notes, newLength),
      }));
    },
    [
      findMidiChannelById,
      halfMeasureDuration,
      measureWidth,
      pointerTimelinePosition,
      secondsPerMeasure,
      updateMidiBlock,
    ],
  );

  const handleMidiBlockPointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const state = midiDragState.current;
    if (!state || state.pointerId !== event.pointerId) return;
    midiDragState.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handleMidiBlockContextMenu = useCallback(
    (event: ReactMouseEvent, channel: MidiChannel, block: MidiBlock) => {
      event.preventDefault();
      event.stopPropagation();
      setMidiMenu({ channelId: channel.id, blockId: block.id, x: event.clientX, y: event.clientY });
    },
    [],
  );

  const handleMidiBlockDoubleClick = useCallback(
    (event: ReactMouseEvent, channel: MidiChannel, block: MidiBlock) => {
      event.stopPropagation();
      setOpenMidiEditor({ channelId: channel.id, blockId: block.id });
    },
    [],
  );

  const handlePianoNotePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channelId: string, block: MidiBlock, note: MidiNote) => {
      if (event.button !== 0) return;
      event.stopPropagation();
      pianoRollDrag.current = {
        noteId: note.id,
        pointerId: event.pointerId,
        originX: event.clientX,
        originY: event.clientY,
        originStart: note.start,
        originPitch: note.pitch,
        channelId,
        blockId: block.id,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [],
  );

  const handlePianoNotePointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      const drag = pianoRollDrag.current;
      if (!drag || drag.pointerId !== event.pointerId) return;
      if (!pianoRollRef.current) return;
      const channel = findMidiChannelById(drag.channelId);
      const block = channel?.blocks.find((item) => item.id === drag.blockId);
      if (!channel || !block) return;

      const bounds = pianoRollRef.current.getBoundingClientRect();
      const relativeX = Math.max(0, Math.min(event.clientX - bounds.left, bounds.width));
      const relativeY = Math.max(0, Math.min(event.clientY - bounds.top, bounds.height));
      const blockDuration = block.length;
      const positionRatio = bounds.width > 0 ? relativeX / bounds.width : 0;
      const newStart = Math.max(0, positionRatio * blockDuration);
      const quantizedStart = Math.max(0, Math.round(newStart / midiGridDuration) * midiGridDuration);
      const pitchRatio = bounds.height > 0 ? 1 - relativeY / bounds.height : 0;
      const pitch = Math.round(MIDI_LOW + pitchRatio * (MIDI_HIGH - MIDI_LOW));
      const clampedPitch = Math.max(MIDI_LOW, Math.min(MIDI_HIGH, pitch));

      updateMidiBlock(drag.channelId, drag.blockId, (current) => ({
        ...current,
        notes: current.notes.map((item) =>
          item.id === drag.noteId ? { ...item, start: quantizedStart, pitch: clampedPitch } : item,
        ),
      }));
    },
    [findMidiChannelById, midiGridDuration, updateMidiBlock],
  );

  const handlePianoNotePointerUp = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = pianoRollDrag.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    pianoRollDrag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handlePianoRollPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channelId: string, block: MidiBlock) => {
      if (event.button !== 0) return;
      event.stopPropagation();
      const target = event.target as HTMLElement;
      if (target.closest("[data-midi-note-id]")) return;
      if (!pianoRollRef.current) return;
      const bounds = pianoRollRef.current.getBoundingClientRect();
      const clampedX = Math.max(0, Math.min(event.clientX - bounds.left, bounds.width));
      const clampedY = Math.max(0, Math.min(event.clientY - bounds.top, bounds.height));
      const totalSteps =
        bounds.width > 0
          ? Math.round((clampedX / bounds.width) * (block.length / midiGridDuration))
          : 0;
      const maxStartStep = Math.max(0, Math.floor(block.length / midiGridDuration));
      const startSteps = Math.min(totalSteps, maxStartStep);
      let start = startSteps * midiGridDuration;
      if (start > block.length - MIN_NOTE_DURATION) {
        start = Math.max(0, block.length - midiGridDuration);
      }
      const remaining = Math.max(0, block.length - start);
      if (remaining <= MIN_NOTE_DURATION) {
        return;
      }
      const desiredSteps = Math.max(1, defaultNoteLengthSteps);
      const maxUsableSteps = Math.max(1, Math.floor(remaining / midiGridDuration));
      const lengthSteps = Math.min(desiredSteps, maxUsableSteps);
      const length =
        remaining < midiGridDuration
          ? remaining
          : Math.max(midiGridDuration, lengthSteps * midiGridDuration);
      const pitchRatio = bounds.height > 0 ? 1 - clampedY / bounds.height : 0;
      const pitch = Math.round(MIDI_LOW + pitchRatio * (MIDI_HIGH - MIDI_LOW));
      const clampedPitch = Math.max(MIDI_LOW, Math.min(MIDI_HIGH, pitch));
      const note: MidiNote = {
        id: nanoid(),
        blockId: block.id,
        start,
        length: Math.min(length, Math.max(MIN_NOTE_DURATION, block.length - start)),
        pitch: clampedPitch,
        velocity: 0.85,
        sampleId: block.sampleId,
      };
      addNoteToBlock(channelId, block.id, note);
    },
    [addNoteToBlock, defaultNoteLengthSteps, midiGridDuration],
  );

  const handleChannelDrop = useCallback(
    (channel: TimelineChannel, event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const dropPosition = projectDropPosition(event);
      const measureData = event.dataTransfer.getData("application/x-measure");
      const sampleData = event.dataTransfer.getData("application/x-sample");
      const beatData = event.dataTransfer.getData("application/x-beat");
      const stemData = event.dataTransfer.getData("application/x-stem");
      const stemFragmentData = event.dataTransfer.getData("application/x-stem-fragment");

      if (channel.type === "midi") {
        if (!sampleData) return;
        const sample = project.samples.find((item) => item.id === sampleData);
        if (!sample) return;

        const channelBlocks = channel.blocks ?? [];
        let targetBlock = findMidiBlock(channelBlocks, dropPosition);
        const blockSize = channel.blockSizeMeasures ?? DEFAULT_BLOCK_SIZE;
        let blocks = channelBlocks;
        if (!targetBlock) {
          const measuresStep = Math.round((dropPosition / secondsPerMeasure) * 2) / 2;
          const start = Math.max(0, measuresStep * secondsPerMeasure);
          const length = Math.max(blockSize * secondsPerMeasure, halfMeasureDuration);
          targetBlock = {
            id: nanoid(),
            start,
            length,
            notes: [],
            rootNote: 60,
            baseMidi: 60,
          };
          blocks = [...channelBlocks, targetBlock];
        }

        const detectedPitch =
          sample.measures[0]?.tunedPitch ??
          sample.measures[0]?.detectedPitch ??
          (sample.key ? sample.key.split(" ")[0] : undefined);
        const baseMidi = parsePitchName(detectedPitch) ?? 60;
        const harmonicLabel = sample.key ?? detectedPitch ?? midiToNoteName(baseMidi);
        const relativeStart = Math.max(0, dropPosition - targetBlock.start);
        const quantizedStart = Math.max(
          0,
          Math.round(relativeStart / midiGridDuration) * midiGridDuration,
        );
        const noteLength = Math.max(
          midiGridDuration,
          Math.min(sample.length, targetBlock.length - quantizedStart),
        );
        const note: MidiNote = {
          id: nanoid(),
          blockId: targetBlock.id,
          start: quantizedStart,
          length: noteLength,
          pitch: 60,
          velocity: 0.85,
          sampleId: sample.id,
        };

        const updatedBlock: MidiBlock = {
          ...targetBlock,
          notes: [...targetBlock.notes, note],
          rootNote: 60,
          baseMidi,
          sampleId: targetBlock.sampleId ?? sample.id,
          harmonicSource: harmonicLabel,
        };

        const nextBlocks = blocks.map((block) =>
          block.id === updatedBlock.id ? updatedBlock : block,
        );

        dispatch({
          type: "update-channel",
          projectId: currentProjectId,
          channelId: channel.id,
          patch: { blocks: nextBlocks },
        });
        updateProjectScale(harmonicLabel);
        void audioEngine.triggerOneShot(sample, 60 - baseMidi);
        return;
      }

      if (channel.type !== "audio") {
        return;
      }

      if (measureData) {
        try {
          const payload = JSON.parse(measureData) as { sampleId: string; measureId: string };
          const sourceSample = project.samples.find((sample) => sample.id === payload.sampleId);
          if (!sourceSample) return;
          if (sourceSample.stems.length === 0) {
            void processSample(sourceSample);
            return;
          }
          const measureIndex = sourceSample.measures.findIndex((measure) => measure.id === payload.measureId);
          if (measureIndex === -1) return;
          const measure = sourceSample.measures[measureIndex];
          const fragment = sliceSampleSegment(sourceSample, measure.start, measure.end, {
            variantLabel: `Measure ${measureIndex + 1}`,
            position: dropPosition,
            channelId: channel.id,
          });
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
          onSelectSample(fragment.id);
          return;
        } catch (error) {
          console.warn("Failed to parse measure drop payload", error);
        }
      }

      if (beatData) {
        try {
          const payload = JSON.parse(beatData) as {
            sampleId: string;
            measureId: string;
            beatId: string;
          };
          const sourceSample = project.samples.find((sample) => sample.id === payload.sampleId);
          if (!sourceSample) return;
          const measure = sourceSample.measures.find((item) => item.id === payload.measureId);
          const beat = measure?.beats?.find((item) => item.id === payload.beatId);
          if (!measure || !beat) return;
          const fragment = sliceSampleSegment(sourceSample, beat.start, beat.end, {
            variantLabel: `Beat ${beat.index + 1}`,
            position: dropPosition,
            channelId: channel.id,
          });
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
          onSelectSample(fragment.id);
          return;
        } catch (error) {
          console.warn("Failed to parse beat payload", error);
        }
      }

      if (stemFragmentData) {
        try {
          const payload = JSON.parse(stemFragmentData) as {
            sampleId: string;
            stemId: string;
            start: number;
            end: number;
          };
          const sourceSample = project.samples.find((sample) => sample.id === payload.sampleId);
          const stem = sourceSample?.stems.find(
            (item) => item.id === payload.stemId || item.sourceStemId === payload.stemId,
          );
          if (!sourceSample || !stem) return;
          const fragment = sliceSampleSegment(sourceSample, payload.start, payload.end, {
            variantLabel: stem.name,
            stems: [stem],
            position: dropPosition,
            channelId: channel.id,
          });
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
          onSelectSample(fragment.id);
          return;
        } catch (error) {
          console.warn("Failed to parse stem fragment payload", error);
        }
      }

      if (stemData) {
        try {
          const payload = JSON.parse(stemData) as { sampleId: string; stemId: string };
          const sourceSample = project.samples.find((sample) => sample.id === payload.sampleId);
          const stem = sourceSample?.stems.find((item) => item.id === payload.stemId);
          if (!sourceSample || !stem) return;
          const fragment = sliceSampleSegment(sourceSample, 0, sourceSample.length, {
            variantLabel: stem.name,
            stems: [stem],
            position: dropPosition,
            channelId: channel.id,
          });
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
          onSelectSample(fragment.id);
          return;
        } catch (error) {
          console.warn("Failed to parse stem payload", error);
        }
      }

      if (sampleData) {
        const sample = project.samples.find((item) => item.id === sampleData);
        if (!sample) return;
        if (sample.stems.length === 0) {
          void processSample(sample);
        }
        dispatch({
          type: "update-sample",
          projectId: currentProjectId,
          sampleId: sample.id,
          sample: { position: dropPosition, isInTimeline: true, channelId: channel.id },
        });
        onSelectSample(sample.id);
      }
    },
    [
      currentProjectId,
      dispatch,
      halfMeasureDuration,
      midiGridDuration,
      onSelectSample,
      processSample,
      project.samples,
      projectDropPosition,
      secondsPerMeasure,
      updateProjectScale,
    ],
  );

  const playClip = useCallback((sample: SampleClip) => {
    void audioEngine.play(sample, sample.measures, { emitTimelineEvents: false });
  }, []);

  const updateTimelineHover = useCallback((sampleId: string, element: HTMLElement | null) => {
    if (!element) return;
    setTimelineHoverInfo({ sampleId, rect: element.getBoundingClientRect() });
  }, []);

  const handleClipPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, sample: SampleClip) => {
      if (!sample.channelId) return;
      const container = scrollRef.current;
      if (!container) return;
      setClipMenu(null);
      event.currentTarget.setPointerCapture(event.pointerId);
      dragState.current = {
        sampleId: sample.id,
        pointerId: event.pointerId,
        originX: event.clientX + container.scrollLeft,
        originY: event.clientY,
        originPosition: sample.position,
        channelId: sample.channelId,
      };
      updateTimelineHover(sample.id, event.currentTarget);
    },
    [setClipMenu, updateTimelineHover],
  );

  const handleClipPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, sample: SampleClip) => {
      const state = dragState.current;
      if (!state || state.pointerId !== event.pointerId) return;
      const container = scrollRef.current;
      if (!container) return;
      const cursor = event.clientX + container.scrollLeft;
      const delta = cursor - state.originX;
      const measuresMoved = delta / measureWidth;
      const newMeasurePosition = Math.max(
        0,
        state.originPosition / secondsPerMeasure + measuresMoved,
      );
      const divisor = SNAP_DIVISORS[snapResolution];
      const quantisedMeasures = Math.round(newMeasurePosition * divisor) / divisor;
      const newPositionSeconds = quantisedMeasures * secondsPerMeasure;
      if (Math.abs(newPositionSeconds - sample.position) > 1e-4) {
        dispatch({
          type: "update-sample",
          projectId: currentProjectId,
          sampleId: sample.id,
          sample: { position: newPositionSeconds },
        });
      }
      updateTimelineHover(sample.id, event.currentTarget);
    },
    [currentProjectId, dispatch, measureWidth, secondsPerMeasure, snapResolution, updateTimelineHover],
  );

  const handleClipPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, sample: SampleClip) => {
      const state = dragState.current;
      if (!state || state.pointerId !== event.pointerId) return;
      dragState.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
      const pointerY = event.clientY;
      for (const [channelId, element] of channelRefs.current.entries()) {
        const bounds = element.getBoundingClientRect();
        if (pointerY >= bounds.top && pointerY <= bounds.bottom && sample.channelId !== channelId) {
          const target = project.channels.find((item) => item.id === channelId);
          if (target?.type === "audio") {
            dispatch({
              type: "update-sample",
              projectId: currentProjectId,
              sampleId: sample.id,
              sample: { channelId },
            });
          }
          break;
        }
      }
      updateTimelineHover(sample.id, event.currentTarget);
    },
    [currentProjectId, dispatch, project.channels, updateTimelineHover],
  );

  const handleRemoveClip = useCallback(
    (sample: SampleClip) => {
      setClipMenu(null);
      if (sample.isFragment) {
        dispatch({ type: "remove-sample", projectId: currentProjectId, sampleId: sample.id });
      } else {
        dispatch({
          type: "update-sample",
          projectId: currentProjectId,
          sampleId: sample.id,
          sample: { isInTimeline: false },
        });
      }
      if (selectedSampleId === sample.id) {
        onSelectSample(null);
      }
      if (effectsSampleId === sample.id) {
        setEffectsSampleId(null);
      }
    },
    [currentProjectId, dispatch, effectsSampleId, onSelectSample, selectedSampleId, setClipMenu],
  );

  const handleClipContextMenu = useCallback(
    (event: ReactMouseEvent, sample: SampleClip) => {
      event.preventDefault();
      event.stopPropagation();
      setClipMenu({ sampleId: sample.id, x: event.clientX, y: event.clientY });
    },
    [setClipMenu],
  );

  const clipGradient = useCallback((stems: SampleClip["stems"]) => {
    if (stems.length === 0) return theme.surface;
    const stops = stems.map((stem, index) => {
      const percentStart = (index / stems.length) * 100;
      const percentEnd = ((index + 1) / stems.length) * 100;
      return `${stem.color}33 ${percentStart}%, ${stem.color}55 ${percentEnd}%`;
    });
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  }, []);

  const clipOutline = useCallback(
    (stems: SampleClip["stems"]) => stems[0]?.color ?? theme.button.primary,
    [],
  );


  const handleMidiLanePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, channel: MidiChannel) => {
      const target = event.target as HTMLElement;
      if (target.closest('[data-midi-block-id]')) return;
      if (event.button !== 0) return;
      const pointerTime = pointerTimelinePosition(event.clientX);
      const blockSize = channel.blockSizeMeasures ?? DEFAULT_BLOCK_SIZE;
      const measuresStep = Math.round((pointerTime / secondsPerMeasure) * 2) / 2;
      const start = Math.max(0, measuresStep * secondsPerMeasure);
      const length = Math.max(0.5, blockSize) * secondsPerMeasure;
      const overlaps = channel.blocks.some(
        (block) => start < block.start + block.length && start + length > block.start,
      );
      if (overlaps) return;
      const block: MidiBlock = {
        id: nanoid(),
        start,
        length,
        notes: [],
        rootNote: 60,
        baseMidi: 60,
      };
      setMidiBlocks(channel.id, [...channel.blocks, block]);
    },
    [halfMeasureDuration, pointerTimelinePosition, secondsPerMeasure, setMidiBlocks],
  );

  const handlePlayArrangement = () => {
    if (isTimelinePlaying) {
      audioEngine.stop();
      setIsTimelinePlaying(false);
      return;
    }
    if (arrangementClips.length === 0) {
      return;
    }
    void audioEngine.playTimeline(arrangementClips);
  };

  const handleStopArrangement = () => {
    audioEngine.stop();
    setIsTimelinePlaying(false);
  };

  const handleAddAudioChannel = () => {
    const index = project.channels.filter((channel) => channel.type === "audio").length + 1;
    dispatch({
      type: "add-channel",
      projectId: currentProjectId,
      channel: {
        id: nanoid(),
        name: `Channel ${index}`,
        type: "audio",
        color: theme.surface,
        isFxEnabled: true,
        volume: 0.85,
        pan: 0,
      },
    });
  };

  const handleAddAutomation = () => {
    const target =
      lastControlTarget ?? ({
        id: "master-bpm",
        label: "Master BPM",
        value: project.masterBpm,
      } as const);
    const index = project.channels.filter((channel) => channel.type === "automation").length + 1;
    const channel: AutomationChannel = {
      id: nanoid(),
      name: `Automation ${index}`,
      type: "automation",
      color: "#07120d",
      isFxEnabled: true,
      volume: 0.85,
      pan: 0,
      parameterId: target.id,
      parameterLabel: target.label,
      points: [
        { id: nanoid(), time: 0, value: target.value },
        { id: nanoid(), time: Math.max(totalDuration, secondsPerMeasure), value: target.value },
      ],
    };
    dispatch({ type: "add-channel", projectId: currentProjectId, channel });
  };

  const handleAddMidi = () => {
    const index = project.channels.filter((channel) => channel.type === "midi").length + 1;
    const channel: MidiChannel = {
      id: nanoid(),
      name: `MIDI ${index}`,
      type: "midi",
      color: "#123b1b",
      isFxEnabled: true,
      volume: 0.85,
      pan: 0,
      instrument: "Chromatic Sampler",
      blocks: [],
      blockSizeMeasures: DEFAULT_BLOCK_SIZE,
    };
    dispatch({ type: "add-channel", projectId: currentProjectId, channel });
  };

  const renderAutomationLane = (channel: AutomationChannel) => {
    const sorted = ensurePointRange([...channel.points]).sort((a, b) => a.time - b.time);
    const values = sorted.map((point) => point.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue || 1;
    const timelineLength = paddedDuration;
    const height = CHANNEL_HEIGHT - 32;
    const width = timelineWidth;
    const pathPoints = sorted
      .map((point) => {
        const x = (point.time / timelineLength) * width;
        const y = height - ((point.value - minValue) / range) * height;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

    return (
      <div
        style={{
          position: "relative",
          height: `${CHANNEL_HEIGHT}px`,
          background: "#030a07",
          borderRadius: "8px",
          border: `1px solid ${theme.button.outline}33`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width={width} height={height} style={{ overflow: "visible" }}>
          <polyline
            points={pathPoints}
            fill="none"
            stroke="#2efc85"
            strokeWidth={2}
            strokeLinecap="round"
          />
        </svg>
      </div>
    );
  };

  const renderMidiLane = (channel: MidiChannel) => {
    const laneHeight = CHANNEL_HEIGHT - 12;
    const innerHeight = laneHeight - 28;
    const pitchRange = Math.max(1, MIDI_HIGH - MIDI_LOW);

    return (
      <div
        style={{
          position: "relative",
          height: `${CHANNEL_HEIGHT}px`,
          background: "#0f2918",
          borderRadius: "10px",
          border: `1px solid ${theme.button.outline}55`,
          padding: "6px 8px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: "12px 8px 12px 8px",
            background: "repeating-linear-gradient(90deg, rgba(46, 252, 133, 0.08) 0, rgba(46, 252, 133, 0.08) 12px, transparent 12px, transparent 48px)",
            borderRadius: "8px",
          }}
        />
        {channel.blocks.map((block) => {
          const blockLeft = (block.start / secondsPerMeasure) * measureWidth;
          const blockWidth = Math.max((block.length / secondsPerMeasure) * measureWidth, 36);
          const blockSample = block.sampleId
            ? project.samples.find((item) => item.id === block.sampleId)
            : null;
          const blockBars = block.length / secondsPerMeasure;
          const barsLabel = Number.isFinite(blockBars)
            ? `${blockBars >= 10 ? blockBars.toFixed(1) : blockBars.toFixed(2)} bars`
            : "--";
          return (
            <div
              key={block.id}
              data-midi-block-id={block.id}
              style={{
                position: "absolute",
                top: "12px",
                left: `${blockLeft}px`,
                width: `${blockWidth}px`,
                height: `${laneHeight}px`,
                borderRadius: "10px",
                border: `1px solid ${theme.button.outline}88`,
                background: "linear-gradient(180deg, rgba(60, 244, 125, 0.22), rgba(3, 20, 10, 0.82))",
                boxShadow: "0 0 18px rgba(46, 252, 133, 0.2)",
                overflow: "hidden",
                cursor: "pointer",
              }}
              onPointerDown={(event) => handleMidiBlockPointerDown(event, channel, block)}
              onPointerMove={(event) => handleMidiBlockPointerMove(event, channel, block)}
              onPointerUp={(event) => handleMidiBlockPointerUp(event)}
              onContextMenu={(event) => handleMidiBlockContextMenu(event, channel, block)}
              onDoubleClick={(event) => handleMidiBlockDoubleClick(event, channel, block)}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "6px 10px",
                  fontSize: "0.7rem",
                  color: theme.text,
                  textShadow: "0 1px 0 rgba(0,0,0,0.4)",
                  background: "rgba(4, 20, 12, 0.45)",
                }}
              >
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {blockSample ? blockSample.name : "Sampler block"}
                </span>
                <span style={{ color: theme.textMuted, fontSize: "0.65rem" }}>
                  {barsLabel}
                </span>
              </div>
              <div
                style={{
                  position: "absolute",
                  top: "28px",
                  left: "6px",
                  right: "6px",
                  bottom: "6px",
                  borderRadius: "8px",
                  background: "rgba(6, 18, 12, 0.55)",
                }}
              >
                {block.notes.map((note) => {
                  const left = Math.max(0, (note.start / secondsPerMeasure) * measureWidth);
                  const width = Math.max(8, (note.length / secondsPerMeasure) * measureWidth);
                  const clampedPitch = Math.max(MIDI_LOW, Math.min(MIDI_HIGH, note.pitch));
                  const pitchRatio = (clampedPitch - MIDI_LOW) / pitchRange;
                  const top = (1 - pitchRatio) * (innerHeight - 18) + 4;
                  return (
                    <div
                      key={note.id}
                      style={{
                        position: "absolute",
                        top: `${top}px`,
                        left: `${left}px`,
                        width: `${width}px`,
                        height: "16px",
                        borderRadius: "6px",
                        background: theme.button.primary,
                        color: theme.button.primaryText,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.62rem",
                        boxShadow: "0 0 8px rgba(46, 252, 133, 0.35)",
                      }}
                    >
                      {midiToNoteName(note.pitch)}
                    </div>
                  );
                })}
              </div>
              <div
                data-resize="start"
                style={{
                  position: "absolute",
                  top: "0",
                  bottom: "0",
                  left: "0",
                  width: "10px",
                  cursor: "ew-resize",
                  background: "linear-gradient(90deg, rgba(6, 24, 13, 0.4), transparent)",
                }}
              />
              <div
                data-resize="end"
                style={{
                  position: "absolute",
                  top: "0",
                  bottom: "0",
                  right: "0",
                  width: "10px",
                  cursor: "ew-resize",
                  background: "linear-gradient(270deg, rgba(6, 24, 13, 0.4), transparent)",
                }}
              />
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div
      style={{
        flex: 1,
        background: theme.surfaceOverlay,
        borderRadius: "12px",
        padding: "10px 12px",
        display: "grid",
        gridTemplateRows: "auto 1fr",
        gap: "10px",
        border: `1px solid ${theme.border}`,
        boxShadow: theme.shadow,
        color: theme.text,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "8px",
          fontSize: "0.72rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <button
            type="button"
            onClick={handleStopArrangement}
            style={{
              padding: "6px 12px",
              borderRadius: "999px",
              border: `1px solid ${theme.button.outline}`,
              background: theme.surface,
              color: theme.text,
              fontSize: "0.75rem",
              cursor: "pointer",
            }}
          >
            Stop
          </button>
          <button
            type="button"
            onClick={handlePlayArrangement}
            style={{
              padding: "6px 14px",
              borderRadius: "999px",
              border: `1px solid ${theme.button.outline}`,
              background: theme.button.primary,
              color: theme.button.primaryText,
              fontWeight: 600,
              fontSize: "0.75rem",
              cursor: "pointer",
            }}
          >
            {isTimelinePlaying ? "Pause" : "Play"}
          </button>
          <span style={{ color: theme.textMuted, fontSize: "0.7rem" }}>
            {totalDuration.toFixed(1)}s timeline
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button
              type="button"
              onClick={handleZoomToFit}
              title="Zoom to fit timeline"
              style={{
                width: "32px",
                height: "32px",
                borderRadius: "50%",
                border: "1px solid rgba(76, 199, 194, 0.45)",
                background: "rgba(76, 199, 194, 0.12)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: theme.button.primary,
                cursor: "pointer",
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="11" cy="11" r="6" />
                <line x1="16.65" y1="16.65" x2="21" y2="21" />
              </svg>
            </button>
            <input
              type="range"
              min={minimumZoom}
              max={MAX_ZOOM}
              step={0.001}
              value={timelineZoom}
              onChange={handleZoomChange}
              aria-label="Timeline zoom"
              title={`Zoom ${zoomLabel}`}
              className="timeline-zoom-slider"
              style={{ "--timeline-zoom-color": theme.button.primary } as CSSProperties}
            />
            <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>{zoomLabel}</span>
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.72rem" }}>
            Snap
            <select
              value={snapResolution}
              onChange={(event) => setSnapResolution(event.target.value as SnapResolution)}
              style={{
                padding: "4px 8px",
                borderRadius: "8px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.72rem",
              }}
            >
              <option value="measure">1 bar</option>
              <option value="half-measure">1/2 bar</option>
              <option value="beat">Beat</option>
              <option value="half-beat">1/2 beat</option>
            </select>
          </label>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "0.72rem" }}>Note Len</span>
            <button
              type="button"
              onClick={decrementNoteLength}
              style={{
                width: "24px",
                height: "24px",
                borderRadius: "6px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              -
            </button>
            <input
              type="number"
              min={1}
              max={32}
              value={defaultNoteLengthSteps}
              onChange={handleNoteLengthInput}
              aria-label="Default MIDI note length"
              style={{
                width: "48px",
                padding: "4px 6px",
                borderRadius: "6px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surfaceOverlay,
                color: theme.text,
                fontSize: "0.72rem",
                textAlign: "center",
              }}
            />
            <button
              type="button"
              onClick={incrementNoteLength}
              style={{
                width: "24px",
                height: "24px",
                borderRadius: "6px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.8rem",
                cursor: "pointer",
              }}
            >
              +
            </button>
            <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>
              {defaultNoteLengthBeats.toFixed(2)} beats
            </span>
          </div>
          <div style={{ display: "flex", gap: "6px" }}>
            <button
              type="button"
              onClick={handleAddAudioChannel}
              style={{
                padding: "6px 10px",
                borderRadius: "8px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.72rem",
                cursor: "pointer",
              }}
            >
              +Audio
            </button>
            <button
              type="button"
              onClick={handleAddAutomation}
              style={{
                padding: "6px 10px",
                borderRadius: "8px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.72rem",
                cursor: "pointer",
              }}
            >
              +Automation
            </button>
            <button
              type="button"
              onClick={handleAddMidi}
              style={{
                padding: "6px 10px",
                borderRadius: "8px",
                border: `1px solid ${theme.button.outline}`,
                background: theme.surface,
                color: theme.text,
                fontSize: "0.72rem",
                cursor: "pointer",
              }}
            >
              +MIDI
            </button>
          </div>
        </div>
      </div>

      <div
        ref={scrollRef}
        style={{ position: "relative", overflow: "auto", borderRadius: "12px", background: theme.surface }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
      >
        <div style={{ position: "relative", minHeight: "100%", minWidth: `${timelineWidth + 240}px` }}>
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 200,
              right: 0,
              height: "28px",
              display: "flex",
              pointerEvents: "none",
            }}
          >
            {timelineGrid.map((tick, index) => (
              <div
                key={`${tick.position}-${index}`}
                style={{
                  flex: "0 0 auto",
                  width: `${measureWidth / SNAP_DIVISORS[snapResolution]}px`,
                  borderRight:
                    tick.accent === 0
                      ? `1px solid ${theme.button.outline}`
                      : `1px dashed ${theme.button.outline}33`,
                  opacity: tick.accent === 0 ? 0.6 : 0.3,
                }}
              />
            ))}
          </div>

          <div style={{ display: "grid", gap: "10px", padding: "32px 0 40px" }}>
            {project.channels.map((channel) => (
              <div
                key={channel.id}
                style={{
                  display: "grid",
                  gridTemplateColumns: "190px 1fr",
                  alignItems: "stretch",
                  gap: "12px",
                  padding: "0 18px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    gap: "6px",
                    fontSize: "0.75rem",
                  }}
                >
                  <strong style={{ fontSize: "0.78rem", letterSpacing: "0.04em" }}>{channel.name}</strong>
                  {channel.type === "automation" ? (
                    <span style={{ color: theme.textMuted, fontSize: "0.7rem" }}>
                      {channel.parameterLabel}
                    </span>
                  ) : null}
                  {channel.type === "midi" ? (
                    <span style={{ color: theme.textMuted, fontSize: "0.7rem" }}>
                      {channel.instrument}
                    </span>
                  ) : null}
                  <button
                    type="button"
                    onClick={() =>
                      dispatch({
                        type: "update-channel",
                        projectId: currentProjectId,
                        channelId: channel.id,
                        patch: { isFxEnabled: channel.isFxEnabled === false ? true : false },
                      })
                    }
                    style={{
                      alignSelf: "flex-start",
                      padding: "4px 10px",
                      borderRadius: "999px",
                      border: `1px solid ${theme.button.outline}`,
                      background: channel.isFxEnabled === false ? theme.surface : theme.button.base,
                      color: channel.isFxEnabled === false ? theme.textMuted : theme.text,
                      fontSize: "0.7rem",
                      cursor: "pointer",
                    }}
                  >
                    FX {channel.isFxEnabled === false ? "Off" : "On"}
                  </button>
                  {channel.type === "midi" ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      <span style={{ fontSize: "0.64rem", color: theme.textMuted }}>Block size (measures)</span>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <button
                          type="button"
                          onClick={() => setMidiBlockSize(channel.id, (channel.blockSizeMeasures ?? DEFAULT_BLOCK_SIZE) - 0.5)}
                          style={{
                            width: "24px",
                            height: "24px",
                            borderRadius: "6px",
                            border: `1px solid ${theme.button.outline}`,
                            background: theme.surface,
                            color: theme.text,
                            fontSize: "0.8rem",
                            cursor: "pointer",
                          }}
                        >
                          -
                        </button>
                        <input
                          type="number"
                          step={0.5}
                          min={0.5}
                          value={(channel.blockSizeMeasures ?? DEFAULT_BLOCK_SIZE).toFixed(1)}
                          onChange={(event) => setMidiBlockSize(channel.id, Number(event.target.value))}
                          style={{
                            width: "54px",
                            padding: "4px 6px",
                            borderRadius: "6px",
                            border: `1px solid ${theme.button.outline}`,
                            background: theme.surfaceOverlay,
                            color: theme.text,
                            fontSize: "0.7rem",
                            textAlign: "center",
                          }}
                        />
                        <button
                          type="button"
                          onClick={() => setMidiBlockSize(channel.id, (channel.blockSizeMeasures ?? DEFAULT_BLOCK_SIZE) + 0.5)}
                          style={{
                            width: "24px",
                            height: "24px",
                            borderRadius: "6px",
                            border: `1px solid ${theme.button.outline}`,
                            background: theme.surface,
                            color: theme.text,
                            fontSize: "0.8rem",
                            cursor: "pointer",
                          }}
                        >
                          +
                        </button>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div
                  ref={(element) => {
                    if (element) {
                      channelRefs.current.set(channel.id, element);
                    } else {
                      channelRefs.current.delete(channel.id);
                    }
                  }}
                  style={{
                    position: "relative",
                    minHeight: `${CHANNEL_HEIGHT}px`,
                    padding: "4px",
                    borderRadius: "10px",
                    background: channel.type === "audio" ? theme.surfaceOverlay : "transparent",
                    border: `1px solid ${theme.button.outline}22`,
                    overflow: "hidden",
                  }}
                  onDragOver={(event) => {
                    event.preventDefault();
                    event.dataTransfer.dropEffect = "copy";
                  }}
                  onDrop={(event) => handleChannelDrop(channel, event)}
                  onPointerDown={(event) => handleChannelPointerDown(event, channel)}
                  onPointerMove={(event) => handleChannelPointerMove(event, channel)}
                  onPointerUp={(event) => handleChannelPointerUp(event)}
                  onContextMenu={(event) => event.preventDefault()}
                >
                  {channel.type === "audio" && (
                    <>
                      {channelSamples.get(channel.id)?.map((sample) => {
                        const widthMeasures = sample.length / secondsPerMeasure || 1;
                        const clipWidth = Math.max(80, widthMeasures * measureWidth);
                        const left = (sample.position / secondsPerMeasure) * measureWidth;
                        const colorStripe = clipOutline(sample.stems);
                        const clipLabel = sample.variantLabel
                          ? `${sample.name} — ${sample.variantLabel}`
                          : sample.name;
                        return (
                          <div
                            key={sample.id}
                            data-clip-id={sample.id}
                            style={{
                              position: "absolute",
                              top: "6px",
                              left: `${left}px`,
                              width: `${clipWidth}px`,
                              height: `${CLIP_HEIGHT}px`,
                              borderRadius: "10px",
                              border: `1px solid ${
                                selectedSampleId === sample.id ? theme.button.primary : colorStripe
                              }`,
                              boxShadow: selectedSampleId === sample.id ? theme.cardGlow : "none",
                              background: theme.surfaceOverlay,
                              cursor: "grab",
                              display: "flex",
                              flexDirection: "column",
                              gap: "6px",
                              padding: "6px 8px 8px",
                            }}
                            onPointerDown={(event) => handleClipPointerDown(event, sample)}
                            onPointerMove={(event) => handleClipPointerMove(event, sample)}
                            onPointerUp={(event) => handleClipPointerUp(event, sample)}
                            onClick={() => onSelectSample(sample.id)}
                            onContextMenu={(event) => handleClipContextMenu(event, sample)}
                            onDoubleClick={(event) => {
                              event.stopPropagation();
                              playClip(sample);
                            }}
                            onMouseEnter={(event) => updateTimelineHover(sample.id, event.currentTarget)}
                            onMouseMove={(event) => updateTimelineHover(sample.id, event.currentTarget)}
                            onMouseLeave={() =>
                              setTimelineHoverInfo((current) =>
                                current?.sampleId === sample.id ? null : current,
                              )
                            }
                            title={`${clipLabel} • ${sample.length.toFixed(2)}s`}
                          >
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                gap: "8px",
                              }}
                            >
                              <strong
                                style={{
                                  fontSize: "0.72rem",
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {clipLabel}
                              </strong>
                              <span style={{ color: theme.textMuted, fontSize: "0.62rem" }}>{channel.name}</span>
                            </div>
                            <div
                              style={{
                                position: "relative",
                                borderRadius: "6px",
                                overflow: "hidden",
                                background: clipGradient(sample.stems),
                                height: `${CLIP_WAVEFORM_HEIGHT}px`,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                              }}
                            >
                              {sample.waveform && sample.waveform.length > 0 ? (
                                <WaveformPreview waveform={sample.waveform} />
                              ) : (
                                <span style={{ fontSize: "0.6rem", color: theme.textMuted }}>
                                  Analyzing waveform…
                                </span>
                              )}
                            </div>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                fontSize: "0.6rem",
                                color: theme.textMuted,
                                gap: "8px",
                              }}
                            >
                              <div style={{ display: "flex", alignItems: "center", gap: "3px" }}>
                                {sample.stems.map((stem) => (
                                  <span
                                    key={`${sample.id}-${stem.id}-chip`}
                                    style={{
                                      width: "6px",
                                      height: "6px",
                                      borderRadius: "50%",
                                      background: stem.color,
                                    }}
                                  />
                                ))}
                              </div>
                              <div style={{ display: "flex", gap: "8px" }}>
                                <span>{sample.bpm ? `${sample.bpm} BPM` : "BPM…"}</span>
                                <span>{sample.key ? `Key ${sample.key}` : "Key…"}</span>
                                <span>{sample.length.toFixed(2)}s</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </>
                  )}
                  {channel.type === "automation" && renderAutomationLane(channel)}
                  {channel.type === "midi" && renderMidiLane(channel)}
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              position: "absolute",
              top: "28px",
              bottom: "40px",
              left: `${200 + (playheadPosition / secondsPerMeasure) * measureWidth}px`,
              width: "2px",
              background: theme.button.primary,
              boxShadow: `0 0 12px ${theme.button.primary}`,
            }}
          />
      </div>
    </div>

      {clipMenu && (() => {
        const sample = project.samples.find((item) => item.id === clipMenu.sampleId);
        if (!sample) return null;
        const viewportHeight = typeof window === "undefined" ? 0 : window.innerHeight;
        const viewportWidth = typeof window === "undefined" ? 0 : window.innerWidth;
        const menuTop = viewportHeight ? Math.min(clipMenu.y, viewportHeight - 180) : clipMenu.y;
        const menuLeft = viewportWidth ? Math.min(clipMenu.x, viewportWidth - 220) : clipMenu.x;
        return (
          <div
            style={{
              position: "fixed",
              top: `${Math.max(16, menuTop)}px`,
              left: `${Math.max(16, menuLeft)}px`,
              background: theme.surfaceOverlay,
              border: `1px solid ${theme.button.outline}`,
              borderRadius: "12px",
              padding: "8px 0",
              minWidth: "200px",
              zIndex: 60,
              boxShadow: theme.shadow,
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setClipMenu(null);
                playClip(sample);
              }}
            >
              Preview clip
            </button>
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setClipMenu(null);
                duplicateClip(sample);
              }}
            >
              Duplicate clip
            </button>
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setClipMenu(null);
                setEffectsSampleId(sample.id);
              }}
            >
              Open effects panel
            </button>
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setClipMenu(null);
                handleRemoveClip(sample);
              }}
            >
              Remove from timeline
            </button>
          </div>
        );
      })()}

      {midiMenu && (() => {
        const channel = findMidiChannelById(midiMenu.channelId);
        const block = channel?.blocks.find((item) => item.id === midiMenu.blockId);
        if (!channel || !block) return null;
        const viewportHeight = typeof window === "undefined" ? 0 : window.innerHeight;
        const viewportWidth = typeof window === "undefined" ? 0 : window.innerWidth;
        const menuTop = viewportHeight ? Math.min(midiMenu.y, viewportHeight - 160) : midiMenu.y;
        const menuLeft = viewportWidth ? Math.min(midiMenu.x, viewportWidth - 220) : midiMenu.x;
        return (
          <div
            style={{
              position: "fixed",
              top: `${Math.max(16, menuTop)}px`,
              left: `${Math.max(16, menuLeft)}px`,
              background: theme.surfaceOverlay,
              border: `1px solid ${theme.button.outline}`,
              borderRadius: "12px",
              padding: "8px 0",
              minWidth: "200px",
              zIndex: 60,
              boxShadow: theme.shadow,
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setMidiMenu(null);
                setOpenMidiEditor({ channelId: channel.id, blockId: block.id });
              }}
            >
              Edit in piano roll
            </button>
            <button
              type="button"
              style={{
                width: "100%",
                padding: "10px 16px",
                background: "transparent",
                border: "none",
                color: theme.text,
                textAlign: "left",
                fontSize: "0.82rem",
                cursor: "pointer",
              }}
              onClick={(event) => {
                event.stopPropagation();
                setMidiMenu(null);
                removeMidiBlock(channel.id, block.id);
              }}
            >
              Delete block
            </button>
          </div>
        );
      })()}

      {effectsSampleId && (() => {
        const target = project.samples.find((sample) => sample.id === effectsSampleId);
        if (!target) return null;
        return (
          <TrackEffectsPanel
            sample={target}
            onClose={() => setEffectsSampleId(null)}
            onUpdate={(effects) =>
              dispatch({
                type: "update-sample",
                projectId: currentProjectId,
                sampleId: target.id,
                sample: { effects },
              })
            }
          />
        );
      })()}

      {timelineHoverInfo && typeof window !== "undefined" && (() => {
        const sample = project.samples.find((item) => item.id === timelineHoverInfo.sampleId);
        if (!sample) return null;
        const cardWidth = 260;
        const cardHeight = 188;
        const scrollX = window.scrollX ?? 0;
        const scrollY = window.scrollY ?? 0;
        const rect = timelineHoverInfo.rect;
        let left = rect.left + rect.width + scrollX + 16;
        if (left + cardWidth > scrollX + window.innerWidth - 16) {
          left = Math.max(scrollX + 16, rect.left + scrollX - cardWidth - 16);
        }
        let top = rect.top + scrollY - 12;
        if (top < scrollY + 16) {
          top = scrollY + 16;
        }
        if (top + cardHeight > scrollY + window.innerHeight - 16) {
          top = scrollY + window.innerHeight - cardHeight - 16;
        }
        const channel = sample.channelId
          ? project.channels.find((channelItem) => channelItem.id === sample.channelId)
          : undefined;
        const percussionNotes =
          heuristicSummary.percussion.length > 0
            ? heuristicSummary.percussion.join(" • ")
            : "Disabled";
        const vocalNotes =
          heuristicSummary.vocals.length > 0
            ? heuristicSummary.vocals.join(" • ")
            : "Disabled";
        return (
          <div
            style={{
              position: "fixed",
              top: `${top}px`,
              left: `${left}px`,
              width: `${cardWidth}px`,
              pointerEvents: "none",
              background: "rgba(6, 16, 23, 0.92)",
              borderRadius: "14px",
              border: `1px solid ${theme.button.outline}`,
              boxShadow: theme.cardGlow,
              padding: "16px 18px",
              color: theme.text,
              zIndex: 80,
            }}
          >
            <strong style={{ display: "block", fontSize: "0.82rem" }}>{sample.name}</strong>
            {sample.variantLabel && (
              <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                {sample.variantLabel}
              </span>
            )}
            <dl
              style={{
                display: "grid",
                gridTemplateColumns: "auto 1fr",
                gap: "4px 12px",
                margin: "10px 0 0",
                fontSize: "0.7rem",
              }}
            >
              <dt style={{ color: theme.textMuted, margin: 0 }}>Start</dt>
              <dd style={{ margin: 0 }}>{sample.position.toFixed(2)}s</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Length</dt>
              <dd style={{ margin: 0 }}>{sample.length.toFixed(2)}s</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Channel</dt>
              <dd style={{ margin: 0 }}>{channel ? channel.name : "Unassigned"}</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Stems</dt>
              <dd style={{ margin: 0 }}>{sample.stems.length}</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Engine</dt>
              <dd style={{ margin: 0 }}>{engineDefinition.name}</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Percussion</dt>
              <dd style={{ margin: 0 }}>{percussionNotes}</dd>
              <dt style={{ color: theme.textMuted, margin: 0 }}>Vocals</dt>
              <dd style={{ margin: 0 }}>{vocalNotes}</dd>
            </dl>
          </div>
        );
      })()}

      {midiEditorContext && (() => {
        const { channel: midiChannel, block: midiBlock } = midiEditorContext;
        const blockSample = midiBlock.sampleId
          ? project.samples.find((item) => item.id === midiBlock.sampleId)
          : null;
        const sortedNotes = midiBlock.notes.slice().sort((a, b) => a.start - b.start);
        const recordProgress = midiBlock.length > 0 ? Math.min(1, recordPlayhead / midiBlock.length) : 0;
        const gridColumns = Math.max(1, Math.ceil(midiBlock.length / midiGridDuration));
        const horizontalSteps = MIDI_HIGH - MIDI_LOW + 1;
        const noteGuide = [84, 79, 74, 69, 64, 60, 55, 48];
        return (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(5, 12, 18, 0.7)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 40,
            }}
            onClick={() => {
              setOpenMidiEditor(null);
              setIsMidiRecording(false);
            }}
          >
            <div
              style={{
                width: "min(820px, 92vw)",
                background: theme.surfaceOverlay,
                borderRadius: "18px",
                border: `1px solid ${theme.button.outline}`,
                boxShadow: theme.cardGlow,
                padding: "28px",
                color: theme.text,
                display: "grid",
                gap: "18px",
              }}
              onClick={(event) => event.stopPropagation()}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: "1.1rem" }}>{midiChannel.name} piano roll</h3>
                  <span style={{ fontSize: "0.75rem", color: theme.textMuted }}>
                    Block starts at {(midiBlock.start / secondsPerMeasure).toFixed(2)} bars · length {(
                      midiBlock.length / secondsPerMeasure
                    ).toFixed(2)} bars
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setOpenMidiEditor(null);
                    setIsMidiRecording(false);
                  }}
                  style={{
                    border: "none",
                    background: "transparent",
                    color: theme.text,
                    cursor: "pointer",
                    fontSize: "1.2rem",
                  }}
                >
                  ✕
                </button>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <span style={{ fontSize: "0.78rem", color: theme.textMuted }}>
                    Harmonic source · {midiBlock.harmonicSource ?? "C Major"}
                  </span>
                  <span style={{ fontSize: "0.78rem", color: theme.textMuted }}>
                    Trigger sample · {blockSample ? blockSample.name : "None assigned"}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <button
                    type="button"
                    onClick={() => setIsMidiRecording((prev) => !prev)}
                    style={{
                      padding: "8px 14px",
                      borderRadius: "999px",
                      border: `1px solid ${theme.button.outline}`,
                      background: isMidiRecording ? theme.button.primary : theme.surface,
                      color: isMidiRecording ? theme.button.primaryText : theme.text,
                      fontWeight: 600,
                      fontSize: "0.75rem",
                      cursor: "pointer",
                    }}
                  >
                    {isMidiRecording ? "Stop MIDI record" : "MIDI record"}
                  </button>
                  <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>
                    Live MIDI {midiInputsRef.current.size > 0 ? "connected" : "not available"}
                  </span>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "60px 1fr", gap: "14px" }}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    height: "240px",
                    padding: "6px 0",
                  }}
                >
                  {noteGuide.map((pitch) => (
                    <span key={pitch} style={{ fontSize: "0.68rem", color: theme.textMuted }}>
                      {midiToNoteName(pitch)}
                    </span>
                  ))}
                </div>
                <div
                  ref={pianoRollRef}
                  style={{
                    position: "relative",
                    height: "240px",
                    borderRadius: "12px",
                    border: `1px solid ${theme.button.outline}`,
                    background: "rgba(3, 14, 9, 0.7)",
                    overflow: "hidden",
                  }}
                  onPointerDown={(event) => handlePianoRollPointerDown(event, midiChannel.id, midiBlock)}
                  onPointerMove={handlePianoNotePointerMove}
                  onPointerUp={handlePianoNotePointerUp}
                  onContextMenu={(event) => event.preventDefault()}
                >
                  {Array.from({ length: gridColumns + 1 }).map((_, index) => (
                    <div
                      key={`grid-x-${index}`}
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${(index / gridColumns) * 100}%`,
                        borderLeft:
                          index % 4 === 0
                            ? `1px solid ${theme.button.outline}`
                            : `1px dashed ${theme.button.outline}55`,
                        opacity: index % 4 === 0 ? 0.5 : 0.25,
                      }}
                    />
                  ))}
                  {Array.from({ length: horizontalSteps + 1 }).map((_, index) => (
                    <div
                      key={`grid-y-${index}`}
                      style={{
                        position: "absolute",
                        left: 0,
                        right: 0,
                        top: `${(index / horizontalSteps) * 100}%`,
                        borderTop: `1px solid ${theme.button.outline}22`,
                      }}
                    />
                  ))}
                  {isMidiRecording && (
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        bottom: 0,
                        left: `${recordProgress * 100}%`,
                        width: "2px",
                        background: theme.button.primary,
                        boxShadow: `0 0 12px ${theme.button.primary}`,
                      }}
                    />
                  )}
                  {sortedNotes.map((note) => {
                    const leftPercent = midiBlock.length > 0 ? (note.start / midiBlock.length) * 100 : 0;
                    const widthPercent = midiBlock.length > 0 ? (note.length / midiBlock.length) * 100 : 0;
                    const clampedPitch = Math.max(MIDI_LOW, Math.min(MIDI_HIGH, note.pitch));
                    const pitchRatio = (clampedPitch - MIDI_LOW) / (MIDI_HIGH - MIDI_LOW);
                    const topPercent = 100 - pitchRatio * 100;
                    return (
                      <div
                        key={note.id}
                        data-midi-note-id={note.id}
                        onPointerDown={(event) => handlePianoNotePointerDown(event, midiChannel.id, midiBlock, note)}
                        onContextMenu={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          removeNoteFromBlock(midiChannel.id, midiBlock.id, note.id);
                        }}
                        style={{
                          position: "absolute",
                          top: `calc(${topPercent}% - 10px)`,
                          left: `${leftPercent}%`,
                          width: `${Math.min(100, Math.max(4, widthPercent))}%`,
                          minWidth: "8px",
                          height: "20px",
                          borderRadius: "6px",
                          background: theme.button.primary,
                          color: theme.button.primaryText,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: "0.65rem",
                          cursor: "grab",
                          boxShadow: "0 0 10px rgba(46, 252, 133, 0.35)",
                          userSelect: "none",
                        }}
                      >
                        {midiToNoteName(note.pitch)}
                      </div>
                    );
                  })}
                </div>
              </div>

              <div style={{ display: "grid", gap: "8px" }}>
                {sortedNotes.length === 0 ? (
                  <span style={{ color: theme.textMuted, fontSize: "0.75rem" }}>
                    No notes yet — drop samples on the MIDI lane or play your keyboard while recording.
                  </span>
                ) : (
                  sortedNotes.map((note) => (
                    <div
                      key={`list-${note.id}`}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.75rem",
                        padding: "6px 12px",
                        borderRadius: "8px",
                        background: theme.surface,
                        border: `1px solid ${theme.button.outline}33`,
                      }}
                    >
                      <span>
                        {midiToNoteName(note.pitch)} · {note.start.toFixed(2)}s → {(note.start + note.length).toFixed(2)}s
                      </span>
                      <span style={{ color: theme.textMuted }}>
                        Vel {(note.velocity * 127).toFixed(0)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
