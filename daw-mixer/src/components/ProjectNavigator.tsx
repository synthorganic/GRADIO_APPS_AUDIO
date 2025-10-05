import {
  ChangeEvent,
  MouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react";
import { nanoid } from "nanoid";
import { audioEngine } from "../lib/audioEngine";
import { retuneSampleMeasures } from "../lib/pitchTools";
import { useProjectStore } from "../state/ProjectStore";
import type { Beat, Measure, SampleClip, StemInfo } from "../types";
import { useDemucsProcessing } from "../hooks/useDemucsProcessing";
import { theme } from "../theme";
import {
  combineMeasures,
  generateBeatsForMeasure,
  sliceSampleSegment,
  sliceStemsForRange
} from "../lib/sampleTools";
import { createDefaultTrackEffects } from "../lib/effectPresets";

interface ProjectNavigatorProps {
  selectedSampleId: string | null;
  onSelectSample: (id: string | null) => void;
}

interface ContextMenuState {
  sampleId: string;
  x: number;
  y: number;
}

interface MeasureContextMenuState extends ContextMenuState {
  measureId: string;
}

const NAV_ITEM_HEIGHT = 44;
const MAX_VISIBLE_ITEMS = 8;

type HoverCardPayload =
  | { type: "sample"; sample: SampleClip; rect: DOMRect }
  | { type: "stem"; sample: SampleClip; stem: StemInfo; rect: DOMRect; measure?: Measure }
  | { type: "measure"; sample: SampleClip; measure: Measure; rect: DOMRect }
  | { type: "beat"; sample: SampleClip; measure: Measure; beat: Beat; rect: DOMRect };

type HoverCardRequest =
  | { type: "sample"; sample: SampleClip }
  | { type: "stem"; sample: SampleClip; stem: StemInfo; measure?: Measure }
  | { type: "measure"; sample: SampleClip; measure: Measure }
  | { type: "beat"; sample: SampleClip; measure: Measure; beat: Beat };

function getMeasureDuration(measure: Measure) {
  return Math.max(0, measure.end - measure.start);
}

export function ProjectNavigator({ selectedSampleId, onSelectSample }: ProjectNavigatorProps) {
  const { currentProjectId, projects, dispatch, getPalette } = useProjectStore();
  const project = projects[currentProjectId];
  const palette = getPalette();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const { processSample, isProcessing } = useDemucsProcessing((updated) => {
    dispatch({ type: "update-sample", projectId: currentProjectId, sampleId: updated.id, sample: updated });
  });

  const [expandedSamples, setExpandedSamples] = useState<Record<string, boolean>>({});
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [measureMenu, setMeasureMenu] = useState<MeasureContextMenuState | null>(null);
  const [hoverCard, setHoverCard] = useState<HoverCardPayload | null>(null);

  const showHoverCard = useCallback((element: HTMLElement, payload: HoverCardRequest) => {
    const rect = element.getBoundingClientRect();
    setHoverCard({ ...payload, rect } as HoverCardPayload);
  }, []);

  const clearHoverCard = useCallback((predicate?: (info: HoverCardPayload) => boolean) => {
    setHoverCard((previous) => {
      if (!previous) {
        return null;
      }
      if (predicate && !predicate(previous)) {
        return previous;
      }
      return null;
    });
  }, []);

  const formatSeconds = (value?: number) => {
    if (value === undefined || Number.isNaN(value)) {
      return "—";
    }
    return `${value.toFixed(2)}s`;
  };

  useEffect(() => {
    const closeMenu = () => {
      setContextMenu(null);
      setMeasureMenu(null);
    };
    const handleAudioStop = () => {
      setPlayingId(null);
    };
    window.addEventListener("click", closeMenu);
    window.addEventListener("audio-stop", handleAudioStop);
    return () => {
      window.removeEventListener("click", closeMenu);
      window.removeEventListener("audio-stop", handleAudioStop);
      audioEngine.stop();
    };
  }, []);

  useEffect(() => {
    if (!hoverCard) return;
    const sampleExists = project.samples.some((item) => item.id === hoverCard.sample.id);
    if (!sampleExists) {
      setHoverCard(null);
      return;
    }
    if (hoverCard.type !== "sample") {
      const isExpanded = expandedSamples[hoverCard.sample.id];
      if (!isExpanded) {
        setHoverCard(null);
      }
    }
  }, [expandedSamples, hoverCard, project.samples]);

  const addSamples = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      let indexOffset = 0;
      const defaultChannelId = project.channels.find((channel) => channel.type === "audio")?.id;
      for (const file of Array.from(files)) {
        const id = nanoid();
        const sample: SampleClip = {
          id,
          name: file.name.replace(/\.[^.]+$/, ""),
          file,
          url: URL.createObjectURL(file),
          bpm: undefined,
          key: undefined,
          measures: [],
          stems: [],
          position: (project.samples.length + indexOffset) * 4,
          length: 4,
          isLooping: false,
          startOffset: 0,
          isFragment: false,
          effects: createDefaultTrackEffects(),
          isInTimeline: false,
          channelId: defaultChannelId
        };
        dispatch({ type: "add-sample", projectId: currentProjectId, sample });
        void processSample(sample);
        indexOffset += 1;
      }
    },
    [currentProjectId, dispatch, processSample, project.channels, project.samples.length]
  );

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    void addSamples(event.target.files);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const groupedSamples = useMemo(
    () => project.samples.filter((sample) => !sample.isFragment),
    [project.samples]
  );

  const toggleExpand = useCallback((sampleId: string) => {
    setExpandedSamples((previous) => ({
      ...previous,
      [sampleId]: !previous[sampleId]
    }));
  }, []);

  const toggleSamplePlayback = async (sample: SampleClip) => {
    if (playingId === sample.id) {
      audioEngine.stop();
      setPlayingId(null);
      return;
    }
    await audioEngine.play(sample, sample.measures, { emitTimelineEvents: false, loop: true });
    setPlayingId(sample.id);
  };

  const playMeasure = async (sample: SampleClip, measure: Measure) => {
    const segmentDuration = getMeasureDuration(measure);
    if (segmentDuration <= 0) return;
    if (playingId === measure.id) {
      audioEngine.stop();
      setPlayingId(null);
      return;
    }
    await audioEngine.playSegment(sample, measure.start, segmentDuration, {
      emitTimelineEvents: false,
      loop: true
    });
    setPlayingId(measure.id);
  };

  const onSampleContextMenu = (event: MouseEvent, sampleId: string) => {
    event.preventDefault();
    setContextMenu({ sampleId, x: event.clientX, y: event.clientY });
  };

  const handleSeparateMeasures = async (sample: SampleClip) => {
    setContextMenu(null);
    await processSample(sample);
    setExpandedSamples((previous) => ({ ...previous, [sample.id]: true }));
  };

  const handleCombineMeasures = (sample: SampleClip) => {
    setContextMenu(null);
    if (!sample.measures.length) return;
    const input = window.prompt("Group how many measures together?", "2");
    if (!input) return;
    const groupSize = Number.parseInt(input, 10);
    if (!Number.isFinite(groupSize) || groupSize <= 1) return;
    const merged = combineMeasures(sample.measures, groupSize);
    const length = merged.length ? merged[merged.length - 1].end : sample.length;
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: { measures: merged, length, duration: length }
    });
    setExpandedSamples((previous) => ({ ...previous, [sample.id]: true }));
  };

  const handleRekey = (sample: SampleClip) => {
    if (sample.measures.length === 0) return;
    const { measures, retuneMap, tunedKey } = retuneSampleMeasures(sample);
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: {
        measures,
        retuneMap,
        rekeyedAt: new Date().toISOString(),
        key: tunedKey
      }
    });
    setContextMenu(null);
    setExpandedSamples((previous) => ({ ...previous, [sample.id]: true }));
  };

  const handleTrim = (sample: SampleClip) => {
    setContextMenu(null);
    const startInput = window.prompt("Trim start (seconds)", "0");
    if (startInput === null) return;
    const endInput = window.prompt("Trim end (seconds)", `${sample.length.toFixed(2)}`);
    if (endInput === null) return;
    const start = Number.parseFloat(startInput);
    const end = Number.parseFloat(endInput);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return;
    const fragment = sliceSampleSegment(sample, start, end, {
      variantLabel: `Trim ${start.toFixed(2)}s-${end.toFixed(2)}s`,
      isFragment: false
    });
    dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
    setExpandedSamples((previous) => ({ ...previous, [fragment.id]: true }));
    onSelectSample(fragment.id);
  };

  const handleDeleteSample = useCallback(
    (sample: SampleClip) => {
      setContextMenu(null);
      const idsToRemove = new Set<string>([sample.id]);
      project.samples.forEach((item) => {
        if (item.originSampleId === sample.id) {
          idsToRemove.add(item.id);
        }
      });
      idsToRemove.forEach((sampleId) =>
        dispatch({ type: "remove-sample", projectId: currentProjectId, sampleId }),
      );
      const removedIds = [...idsToRemove];
      if (selectedSampleId && removedIds.includes(selectedSampleId)) {
        onSelectSample(null);
      }
      if (playingId && removedIds.includes(playingId)) {
        audioEngine.stop();
        setPlayingId(null);
      }
    },
    [currentProjectId, dispatch, onSelectSample, playingId, project.samples, selectedSampleId],
  );

  const handleSliceMeasure = useCallback(
    (sample: SampleClip, measure: Measure) => {
      setMeasureMenu(null);
      const measureIndex = sample.measures.findIndex((item) => item.id === measure.id);
      if (measureIndex === -1) return;
      const fragment = sliceSampleSegment(sample, measure.start, measure.end, {
        variantLabel: `Measure ${measureIndex + 1}`,
        position: sample.position + measure.start,
        channelId: sample.channelId,
        isInTimeline: true,
      });
      dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
      setExpandedSamples((previous) => ({ ...previous, [sample.id]: true }));
      onSelectSample(fragment.id);
    },
    [currentProjectId, dispatch, onSelectSample],
  );

  const handleGenerateBeats = (sample: SampleClip, measure: Measure) => {
    setMeasureMenu(null);
    const beats = generateBeatsForMeasure(sample, measure);
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: {
        measures: sample.measures.map((item) =>
          item.id === measure.id ? { ...item, beats } : item
        )
      }
    });
    setExpandedSamples((previous) => ({ ...previous, [sample.id]: true }));
  };

  const handleMeasureContextMenu = (event: MouseEvent, sampleId: string, measureId: string) => {
    event.preventDefault();
    setMeasureMenu({ sampleId, measureId, x: event.clientX, y: event.clientY });
  };

  const playStem = async (sample: SampleClip, stem: StemInfo) => {
    if (playingId === stem.id) {
      audioEngine.stop();
      setPlayingId(null);
      return;
    }
    await audioEngine.playStem(sample, stem, sample.measures, {
      emitTimelineEvents: false,
      loop: true
    });
    setPlayingId(stem.id);
  };

  const playBeat = async (sample: SampleClip, beat: Beat) => {
    const duration = Math.max(0, beat.end - beat.start);
    if (duration <= 0) return;
    if (playingId === beat.id) {
      audioEngine.stop();
      setPlayingId(null);
      return;
    }
    await audioEngine.playSegment(sample, beat.start, duration, {
      emitTimelineEvents: false,
      loop: true
    });
    setPlayingId(beat.id);
  };

  return (
    <div
      style={{
        flex: 1,
        overflow: "hidden",
        padding: "12px 16px",
        display: "flex",
        flexDirection: "column",
          gap: "14px",
        background: theme.surface,
        borderBottom: `1px solid ${theme.divider}`,
        color: theme.text
      }}
    >
      <section
        style={{
          padding: "14px",
          border: `1px dashed ${theme.button.outline}`,
          borderRadius: "10px",
          textAlign: "center",
          background: theme.surfaceOverlay,
          color: theme.textMuted,
          position: "relative",
          boxShadow: theme.cardGlow
        }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={(event) => {
          event.preventDefault();
          void addSamples(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          multiple
          style={{ display: "none" }}
          onChange={handleFileInput}
        />
        <p style={{ margin: "0 0 8px", fontWeight: 600, letterSpacing: "0.04em", color: theme.text, fontSize: "0.75rem" }}>
          Drop audio or browse to start a technicolor session
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          style={{
            border: `1px solid ${theme.button.outline}`,
            background: theme.button.primary,
            color: theme.button.primaryText,
            padding: "8px 16px",
            borderRadius: "999px",
            fontWeight: 600,
            fontSize: "0.7rem",
            cursor: "pointer",
            boxShadow: theme.cardGlow
          }}
        >
          Select audio
        </button>
        <p style={{ margin: "10px 0 0", fontSize: "0.72rem", color: theme.textMuted }}>
          Samples auto-stretch to {project.masterBpm} BPM. DEMUCS will unfold stems on demand.
        </p>
        {isProcessing && (
          <span
            style={{
              position: "absolute",
              top: "12px",
              right: "14px",
              fontSize: "0.65rem",
              color: theme.button.primary
            }}
          >
            Separating stems…
          </span>
        )}
      </section>

      <h2
        style={{
          margin: 0,
          fontSize: "0.75rem",
          letterSpacing: "0.06em",
          color: theme.text,
          textTransform: "uppercase"
        }}
      >
        Project navigator
      </h2>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          padding: 0,
          margin: 0,
          overflowY: "auto"
        }}
      >
        {groupedSamples.map((sample) => {
          const expanded = expandedSamples[sample.id] ?? false;
          const isSelected = selectedSampleId === sample.id;
          const isSampleHovered = hoverCard?.type === "sample" && hoverCard.sample.id === sample.id;
          const highlightSample = isSelected || isSampleHovered;
          return (
            <div
              key={sample.id}
              style={{
                background: highlightSample ? theme.surfaceRaised : theme.surfaceOverlay,
                borderRadius: "12px",
                border: `1px solid ${highlightSample ? theme.button.primary : theme.border}`,
                boxShadow: highlightSample ? theme.cardGlow : "none",
                transition: "border 0.2s ease, box-shadow 0.2s ease",
                overflow: "visible",
                position: "relative"
              }}
              onContextMenu={(event) => onSampleContextMenu(event, sample.id)}
              onClick={() => onSelectSample(sample.id)}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "0 14px",
                  minHeight: `${NAV_ITEM_HEIGHT}px`,
                  gap: "10px",
                  fontSize: "0.78rem"
                }}
                onMouseEnter={(event) => {
                  showHoverCard(event.currentTarget, { type: "sample", sample });
                }}
                onMouseLeave={() =>
                  clearHoverCard((info) => info.type === "sample" && info.sample.id === sample.id)
                }
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    flex: 1,
                    minWidth: 0
                  }}
                >
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      toggleExpand(sample.id);
                    }}
                    style={{
                      width: "22px",
                      height: "22px",
                      borderRadius: "6px",
                      border: `1px solid ${theme.button.outline}`,
                      background: theme.surface,
                      color: theme.text,
                      fontSize: "0.65rem",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      padding: 0
                    }}
                    aria-label={expanded ? "Collapse sample" : "Expand sample"}
                  >
                    {expanded ? "▾" : "▸"}
                  </button>
                  <strong
                    style={{
                      fontSize: "0.8rem",
                      color: theme.text,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis"
                    }}
                  >
                    {sample.name}
                  </strong>
                  {sample.variantLabel && (
                    <span
                      style={{
                        fontSize: "0.68rem",
                        color: theme.button.primary,
                        background: `${theme.button.primary}20`,
                        borderRadius: "999px",
                        padding: "2px 8px"
                      }}
                    >
                      {sample.variantLabel}
                    </span>
                  )}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      void toggleSamplePlayback(sample);
                    }}
                    style={{
                      width: "28px",
                      height: "28px",
                      borderRadius: "50%",
                      border: `1px solid ${theme.button.outline}`,
                      background: playingId === sample.id ? theme.button.primary : theme.button.base,
                      color: playingId === sample.id ? theme.button.primaryText : theme.text,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 0,
                      boxShadow: playingId === sample.id ? theme.cardGlow : "none"
                    }}
                    aria-label={playingId === sample.id ? "Stop sample preview" : "Play sample preview"}
                  >
                    {playingId === sample.id ? "■" : "▶"}
                  </button>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      handleDeleteSample(sample);
                    }}
                    style={{
                      width: "26px",
                      height: "26px",
                      borderRadius: "50%",
                      border: `1px solid ${theme.button.outline}`,
                      background: theme.surface,
                      color: theme.textMuted,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 0
                    }}
                    aria-label="Delete sample"
                    title="Delete sample"
                  >
                    ✕
                  </button>
                </div>
              </div>

              {expanded && (
                <div
                  style={{
                    borderTop: `1px solid ${theme.border}`,
                    padding: "10px 12px 12px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px"
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      minHeight: `${NAV_ITEM_HEIGHT}px`,
                      padding: "0 12px",
                      borderRadius: "10px",
                      border: `1px solid ${theme.border}`,
                      background: theme.surface
                    }}
                    draggable
                    onDragStart={(event) => {
                      event.dataTransfer.effectAllowed = "copy";
                      event.dataTransfer.setData("application/x-sample", sample.id);
                    }}
                    onMouseEnter={(event) => showHoverCard(event.currentTarget, { type: "sample", sample })}
                    onMouseLeave={() =>
                      clearHoverCard((info) => info.type === "sample" && info.sample.id === sample.id)
                    }
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", minWidth: 0 }}>
                      <strong style={{ fontSize: "0.8rem", color: theme.text }}>Full sample</strong>
                      <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>
                        {sample.measures.length} measures
                      </span>
                      {sample.isFragment && (
                        <span style={{ fontSize: "0.68rem", color: theme.textMuted }}>(Fragment)</span>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        void toggleSamplePlayback(sample);
                      }}
                      style={{
                        width: "28px",
                        height: "28px",
                        borderRadius: "50%",
                        border: `1px solid ${theme.button.outline}`,
                        background: playingId === sample.id ? theme.button.primary : theme.button.base,
                        color: playingId === sample.id ? theme.button.primaryText : theme.text,
                        cursor: "pointer",
                        boxShadow: playingId === sample.id ? theme.cardGlow : "none"
                      }}
                    >
                      {playingId === sample.id ? "■" : "▶"}
                    </button>
                  </div>

                  <details
                    open
                    style={{
                      borderRadius: "12px",
                      border: `1px solid ${theme.border}`,
                      background: theme.surface,
                      overflow: "hidden"
                    }}
                  >
                    <summary
                      style={{
                        cursor: "pointer",
                        fontSize: "0.8rem",
                        color: theme.text,
                        listStyle: "none",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "0 12px",
                        minHeight: `${NAV_ITEM_HEIGHT}px`
                      }}
                    >
                      <span>Full stems</span>
                      <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                        {sample.stems.length}
                      </span>
                    </summary>
                    <div
                      style={{
                        maxHeight: `${NAV_ITEM_HEIGHT * MAX_VISIBLE_ITEMS}px`,
                        overflowY: "auto",
                        padding: "6px 0"
                      }}
                    >
                      {sample.stems.length === 0 ? (
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            minHeight: `${NAV_ITEM_HEIGHT}px`,
                            padding: "0 12px",
                            fontSize: "0.75rem",
                            color: theme.textMuted
                          }}
                        >
                          Stems will appear after separation
                        </div>
                      ) : (
                        <ul
                          style={{
                            listStyle: "none",
                            margin: 0,
                            padding: 0,
                            display: "flex",
                            flexDirection: "column",
                            gap: "6px"
                          }}
                        >
                          {sample.stems.map((stem) => {
                            const isHoveringStem =
                              hoverCard?.type === "stem" &&
                              hoverCard.sample.id === sample.id &&
                              hoverCard.stem.id === stem.id;
                            return (
                              <li
                                key={stem.id}
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "space-between",
                                  height: `${NAV_ITEM_HEIGHT}px`,
                                  padding: "0 12px",
                                  borderRadius: "10px",
                                  border: `1px solid ${
                                    isHoveringStem ? theme.button.primary : theme.border
                                  }`,
                                  background: `${stem.color}1a`,
                                  boxShadow: isHoveringStem ? theme.cardGlow : "none"
                                }}
                                draggable
                                onDragStart={(event) => {
                                  event.dataTransfer.effectAllowed = "copy";
                                  event.dataTransfer.setData(
                                    "application/x-stem",
                                    JSON.stringify({ sampleId: sample.id, stemId: stem.id })
                                  );
                                }}
                                onMouseEnter={(event) =>
                                  showHoverCard(event.currentTarget, { type: "stem", sample, stem })
                                }
                                onMouseLeave={() =>
                                  clearHoverCard(
                                    (info) =>
                                      info.type === "stem" &&
                                      info.sample.id === sample.id &&
                                      info.stem.id === stem.id
                                  )
                                }
                              >
                                <span
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "8px",
                                    color: theme.text,
                                    minWidth: 0
                                  }}
                                >
                                  <span
                                    style={{
                                      width: "10px",
                                      height: "10px",
                                      borderRadius: "50%",
                                      background: stem.color
                                    }}
                                  ></span>
                                  <span
                                    style={{
                                      whiteSpace: "nowrap",
                                      overflow: "hidden",
                                      textOverflow: "ellipsis"
                                    }}
                                  >
                                    {stem.name}
                                  </span>
                                </span>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void playStem(sample, stem);
                                  }}
                                  style={{
                                    width: "26px",
                                    height: "26px",
                                    borderRadius: "50%",
                                    border: `1px solid ${theme.button.outline}`,
                                    background:
                                      playingId === stem.id ? theme.button.primary : theme.button.base,
                                    color:
                                      playingId === stem.id ? theme.button.primaryText : theme.text,
                                    cursor: "pointer",
                                    fontSize: "0.68rem"
                                  }}
                                >
                                  {playingId === stem.id ? "■" : "▶"}
                                </button>
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>
                  </details>

                  <details
                    open
                    style={{
                      borderRadius: "12px",
                      border: `1px solid ${theme.border}`,
                      background: theme.surface,
                      overflow: "hidden"
                    }}
                  >
                    <summary
                      style={{
                        cursor: "pointer",
                        fontSize: "0.8rem",
                        color: theme.text,
                        listStyle: "none",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "0 12px",
                        minHeight: `${NAV_ITEM_HEIGHT}px`
                      }}
                    >
                      <span>Measures</span>
                      <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                        {sample.measures.length}
                      </span>
                    </summary>
                    <div
                      style={{
                        maxHeight: `${NAV_ITEM_HEIGHT * MAX_VISIBLE_ITEMS}px`,
                        overflowY: "auto",
                        padding: "6px 0"
                      }}
                    >
                      {sample.measures.length === 0 ? (
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            minHeight: `${NAV_ITEM_HEIGHT}px`,
                            padding: "0 12px",
                            fontSize: "0.75rem",
                            color: theme.textMuted
                          }}
                        >
                          Waiting for measure detection…
                        </div>
                      ) : (
                        <ul
                          style={{
                            listStyle: "none",
                            margin: 0,
                            padding: 0,
                            display: "flex",
                            flexDirection: "column",
                            gap: "8px"
                          }}
                        >
                          {sample.measures.map((measure, index) => {
                            const measureStems = sliceStemsForRange(
                              sample,
                              measure.start,
                              measure.end
                            );
                            const hoveredBeatId =
                              hoverCard?.type === "beat" &&
                              hoverCard.sample.id === sample.id &&
                              hoverCard.measure.id === measure.id
                                ? hoverCard.beat.id
                                : null;
                            return (
                              <li
                                key={measure.id}
                                style={{
                                  border: `1px solid ${theme.border}`,
                                  borderRadius: "12px",
                                  background: theme.surface,
                                  padding: "8px 10px",
                                  display: "flex",
                                  flexDirection: "column",
                                  gap: "6px"
                                }}
                                onContextMenu={(event) =>
                                  handleMeasureContextMenu(event, sample.id, measure.id)
                                }
                              >
                                <div
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    minHeight: `${NAV_ITEM_HEIGHT}px`,
                                    padding: "0 2px"
                                  }}
                                  draggable
                                  onDragStart={(event) => {
                                    event.dataTransfer.effectAllowed = "copy";
                                    event.dataTransfer.setData(
                                      "application/x-measure",
                                      JSON.stringify({ sampleId: sample.id, measureId: measure.id })
                                    );
                                  }}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    onSelectSample(sample.id);
                                  }}
                                  onMouseEnter={(event) =>
                                    showHoverCard(event.currentTarget, {
                                      type: "measure",
                                      sample,
                                      measure
                                    })
                                  }
                                  onMouseLeave={() =>
                                    clearHoverCard(
                                      (info) =>
                                        info.type === "measure" &&
                                        info.sample.id === sample.id &&
                                        info.measure.id === measure.id
                                    )
                                  }
                                >
                                  <div
                                    style={{
                                      display: "flex",
                                      alignItems: "center",
                                      gap: "8px",
                                      minWidth: 0
                                    }}
                                  >
                                    <strong style={{ fontSize: "0.78rem", color: theme.text }}>
                                      Measure {index + 1}
                                    </strong>
                                    <span
                                      style={{
                                        fontSize: "0.7rem",
                                        color: theme.textMuted,
                                        whiteSpace: "nowrap"
                                      }}
                                    >
                                      {measure.beatCount} beats
                                    </span>
                                  </div>
                                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                    <button
                                      type="button"
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        void playMeasure(sample, measure);
                                      }}
                                      style={{
                                        width: "28px",
                                        height: "28px",
                                        borderRadius: "50%",
                                        border: `1px solid ${theme.button.outline}`,
                                        background:
                                          playingId === measure.id
                                            ? theme.button.primary
                                            : theme.button.base,
                                        color:
                                          playingId === measure.id
                                            ? theme.button.primaryText
                                            : theme.text,
                                        cursor: "pointer",
                                        fontSize: "0.75rem"
                                      }}
                                    >
                                      {playingId === measure.id ? "■" : "▶"}
                                    </button>
                                  </div>
                                </div>
                                {measureStems.length > 0 && (
                                  <div
                                    style={{
                                      display: "flex",
                                      gap: "6px",
                                      overflowX: "auto",
                                      padding: "0 2px 0 2px"
                                    }}
                                  >
                                    {measureStems.map((stem) => {
                                      const isHoveringStem =
                                        hoverCard?.type === "stem" &&
                                        hoverCard.sample.id === sample.id &&
                                        hoverCard.stem.id === stem.id;
                                      return (
                                        <div
                                          key={stem.id}
                                          style={{
                                            flex: "0 0 auto",
                                            display: "flex",
                                            alignItems: "center",
                                            gap: "6px",
                                            padding: "0 12px",
                                            height: `${NAV_ITEM_HEIGHT - 8}px`,
                                            borderRadius: "999px",
                                            border: `1px solid ${
                                              isHoveringStem ? theme.button.primary : theme.border
                                            }`,
                                            background: `${stem.color}24`,
                                            boxShadow: isHoveringStem ? theme.cardGlow : "none"
                                          }}
                                          draggable
                                          onDragStart={(event) => {
                                            event.dataTransfer.effectAllowed = "copy";
                                            event.dataTransfer.setData(
                                              "application/x-stem-fragment",
                                              JSON.stringify({
                                                sampleId: sample.id,
                                                stemId: stem.sourceStemId ?? stem.id,
                                                start: measure.start,
                                                end: measure.end
                                              })
                                            );
                                          }}
                                          onMouseEnter={(event) =>
                                            showHoverCard(event.currentTarget, {
                                              type: "stem",
                                              sample,
                                              stem,
                                              measure
                                            })
                                          }
                                          onMouseLeave={() =>
                                            clearHoverCard(
                                              (info) =>
                                                info.type === "stem" &&
                                                info.sample.id === sample.id &&
                                                info.stem.id === stem.id
                                            )
                                          }
                                        >
                                          <span
                                            style={{
                                              width: "10px",
                                              height: "10px",
                                              borderRadius: "50%",
                                              background: stem.color
                                            }}
                                          ></span>
                                          <span
                                            style={{
                                              fontSize: "0.72rem",
                                              color: theme.text,
                                              whiteSpace: "nowrap"
                                            }}
                                          >
                                            {stem.name}
                                          </span>
                                          <button
                                            type="button"
                                            onClick={(event) => {
                                              event.stopPropagation();
                                              void playStem(sample, stem);
                                            }}
                                            style={{
                                              width: "24px",
                                              height: "24px",
                                              borderRadius: "50%",
                                              border: `1px solid ${theme.button.outline}`,
                                              background:
                                                playingId === stem.id
                                                  ? theme.button.primary
                                                  : theme.button.base,
                                              color:
                                                playingId === stem.id
                                                  ? theme.button.primaryText
                                                  : theme.text,
                                              cursor: "pointer",
                                              fontSize: "0.68rem"
                                            }}
                                          >
                                            {playingId === stem.id ? "■" : "▶"}
                                          </button>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                                {measure.beats && measure.beats.length > 0 && (
                                  <div
                                    style={{
                                      display: "flex",
                                      gap: "6px",
                                      overflowX: "auto",
                                      padding: "0 2px 2px"
                                    }}
                                  >
                                    {measure.beats.map((beat) => {
                                      const isActive = hoveredBeatId === beat.id;
                                      return (
                                        <button
                                          key={beat.id}
                                          type="button"
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            void playBeat(sample, beat);
                                          }}
                                          onMouseEnter={(event) =>
                                            showHoverCard(event.currentTarget, {
                                              type: "beat",
                                              sample,
                                              measure,
                                              beat
                                            })
                                          }
                                          onMouseLeave={() =>
                                            clearHoverCard(
                                              (info) =>
                                                info.type === "beat" &&
                                                info.sample.id === sample.id &&
                                                info.beat.id === beat.id
                                            )
                                          }
                                          style={{
                                            flex: "0 0 auto",
                                            display: "flex",
                                            alignItems: "center",
                                            gap: "6px",
                                            height: `${NAV_ITEM_HEIGHT - 8}px`,
                                            padding: "0 12px",
                                            borderRadius: "999px",
                                            border: `1px solid ${
                                              isActive ? theme.button.primary : theme.border
                                            }`,
                                            background: isActive
                                              ? `${theme.button.primary}24`
                                              : theme.surfaceOverlay,
                                            color: theme.text,
                                            fontSize: "0.72rem",
                                            cursor: "pointer",
                                            boxShadow: isActive ? theme.cardGlow : "none"
                                          }}
                                          draggable
                                          onDragStart={(event) => {
                                            event.dataTransfer.effectAllowed = "copy";
                                            event.dataTransfer.setData(
                                              "application/x-beat",
                                              JSON.stringify({
                                                sampleId: sample.id,
                                                measureId: measure.id,
                                                beatId: beat.id
                                              })
                                            );
                                          }}
                                        >
                                          <span style={{ fontWeight: 600 }}>B{beat.index + 1}</span>
                                          <span style={{ color: theme.textMuted }}>
                                            {(beat.end - beat.start).toFixed(2)}s
                                          </span>
                                          <span style={{ color: theme.button.primary }}>
                                            {playingId === beat.id ? "■" : "▶"}
                                          </span>
                                        </button>
                                      );
                                    })}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>
                  </details>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {hoverCard && typeof window !== "undefined" && (() => {
        const cardWidth = 280;
        const estimatedHeight =
          hoverCard.type === "sample"
            ? 220
            : hoverCard.type === "measure"
            ? 210
            : 190;
        const scrollX = window.scrollX ?? 0;
        const scrollY = window.scrollY ?? 0;
        let top = hoverCard.rect.top + scrollY;
        let left = hoverCard.rect.right + scrollX + 16;
        if (left + cardWidth > scrollX + window.innerWidth - 16) {
          left = Math.max(16 + scrollX, hoverCard.rect.left + scrollX - cardWidth - 16);
        }
        const maxTop = scrollY + window.innerHeight - estimatedHeight - 16;
        top = Math.max(16 + scrollY, Math.min(top, maxTop));
        const accent =
          hoverCard.type === "stem"
            ? hoverCard.stem.color
            : hoverCard.type === "beat"
            ? theme.button.primary
            : theme.button.outline;
        const gridStyle = {
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: "4px 12px",
          margin: "8px 0 0",
          fontSize: "0.72rem"
        };
        const labelStyle = { color: theme.textMuted, margin: 0 };
        const valueStyle = { margin: 0, color: theme.text };

        const content = (() => {
          switch (hoverCard.type) {
            case "sample": {
              const { sample } = hoverCard;
              const channel = sample.channelId
                ? project.channels.find((channelItem) => channelItem.id === sample.channelId)
                : undefined;
              const clipLength = sample.duration ?? sample.length;
              const waveformStatus =
                sample.waveform && sample.waveform.length > 0 ? "Ready" : "Analyzing";
              return (
                <>
                  <strong style={{ fontSize: "0.8rem", display: "block", color: theme.text }}>
                    {sample.name}
                  </strong>
                  {sample.variantLabel && (
                    <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                      {sample.variantLabel}
                    </span>
                  )}
                  <dl style={gridStyle}>
                    <dt style={labelStyle}>Length</dt>
                    <dd style={valueStyle}>{formatSeconds(clipLength)}</dd>
                    <dt style={labelStyle}>Measures</dt>
                    <dd style={valueStyle}>{sample.measures.length}</dd>
                    <dt style={labelStyle}>BPM</dt>
                    <dd style={valueStyle}>{sample.bpm ?? "—"}</dd>
                    <dt style={labelStyle}>Key</dt>
                    <dd style={valueStyle}>{sample.key ?? "—"}</dd>
                    <dt style={labelStyle}>Looping</dt>
                    <dd style={valueStyle}>{sample.isLooping ? "Enabled" : "Off"}</dd>
                    <dt style={labelStyle}>Channel</dt>
                    <dd style={valueStyle}>{channel ? channel.name : "Unassigned"}</dd>
                    <dt style={labelStyle}>Waveform</dt>
                    <dd style={valueStyle}>{waveformStatus}</dd>
                  </dl>
                </>
              );
            }
            case "stem": {
              const { stem, sample, measure } = hoverCard;
              const duration = stem.duration ?? sample.duration ?? sample.length;
              const sourceStem = stem.sourceStemId
                ? sample.stems.find((item) => item.id === stem.sourceStemId)
                : undefined;
              const measureIndex = measure
                ? sample.measures.findIndex((item) => item.id === measure.id)
                : -1;
              return (
                <>
                  <strong style={{ fontSize: "0.8rem", display: "block", color: theme.text }}>
                    {stem.name}
                  </strong>
                  <dl style={gridStyle}>
                    <dt style={labelStyle}>Type</dt>
                    <dd style={valueStyle}>{stem.type}</dd>
                    <dt style={labelStyle}>Duration</dt>
                    <dd style={valueStyle}>{formatSeconds(duration)}</dd>
                    <dt style={labelStyle}>Start</dt>
                    <dd style={valueStyle}>{formatSeconds(stem.startOffset)}</dd>
                    <dt style={labelStyle}>BPM</dt>
                    <dd style={valueStyle}>{stem.bpm ?? sample.bpm ?? "—"}</dd>
                    <dt style={labelStyle}>Key</dt>
                    <dd style={valueStyle}>{stem.key ?? sample.key ?? "—"}</dd>
                    {measure && (
                      <>
                        <dt style={labelStyle}>Measure</dt>
                        <dd style={valueStyle}>
                          {measureIndex >= 0 ? `Measure ${measureIndex + 1}` : "—"}
                        </dd>
                      </>
                    )}
                    {stem.extractionModel && (
                      <>
                        <dt style={labelStyle}>Model</dt>
                        <dd style={valueStyle}>{stem.extractionModel}</dd>
                      </>
                    )}
                    {stem.processingNotes && (
                      <>
                        <dt style={labelStyle}>Notes</dt>
                        <dd style={valueStyle}>{stem.processingNotes}</dd>
                      </>
                    )}
                    {sourceStem && (
                      <>
                        <dt style={labelStyle}>Source</dt>
                        <dd style={valueStyle}>{sourceStem.name}</dd>
                      </>
                    )}
                  </dl>
                </>
              );
            }
            case "measure": {
              const { measure, sample } = hoverCard;
              const index = sample.measures.findIndex((item) => item.id === measure.id);
              const tunedPitch = measure.tunedPitch ?? measure.detectedPitch ?? "—";
              const pitchLabel =
                measure.tunedPitch &&
                measure.detectedPitch &&
                measure.tunedPitch !== measure.detectedPitch
                  ? `${measure.detectedPitch} → ${measure.tunedPitch}`
                  : tunedPitch;
              return (
                <>
                  <strong style={{ fontSize: "0.8rem", display: "block", color: theme.text }}>
                    Measure {index >= 0 ? index + 1 : ""}
                  </strong>
                  <dl style={gridStyle}>
                    <dt style={labelStyle}>Start</dt>
                    <dd style={valueStyle}>{formatSeconds(measure.start)}</dd>
                    <dt style={labelStyle}>End</dt>
                    <dd style={valueStyle}>{formatSeconds(measure.end)}</dd>
                    <dt style={labelStyle}>Duration</dt>
                    <dd style={valueStyle}>{formatSeconds(getMeasureDuration(measure))}</dd>
                    <dt style={labelStyle}>Beats</dt>
                    <dd style={valueStyle}>{measure.beatCount}</dd>
                    <dt style={labelStyle}>Pitch</dt>
                    <dd style={valueStyle}>{pitchLabel}</dd>
                    <dt style={labelStyle}>Energy</dt>
                    <dd style={valueStyle}>
                      {measure.energy !== undefined ? `${Math.round(measure.energy * 100)}%` : "—"}
                    </dd>
                  </dl>
                </>
              );
            }
            case "beat": {
              const { beat } = hoverCard;
              return (
                <>
                  <strong style={{ fontSize: "0.8rem", display: "block", color: theme.text }}>
                    Beat {beat.index + 1}
                  </strong>
                  <dl style={gridStyle}>
                    <dt style={labelStyle}>Start</dt>
                    <dd style={valueStyle}>{formatSeconds(beat.start)}</dd>
                    <dt style={labelStyle}>End</dt>
                    <dd style={valueStyle}>{formatSeconds(beat.end)}</dd>
                    <dt style={labelStyle}>Duration</dt>
                    <dd style={valueStyle}>{formatSeconds(beat.end - beat.start)}</dd>
                    <dt style={labelStyle}>Fragments</dt>
                    <dd style={valueStyle}>
                      {beat.stems.length > 0
                        ? beat.stems.map((item) => item.name).join(", ")
                        : "—"}
                    </dd>
                  </dl>
                </>
              );
            }
            default:
              return null;
          }
        })();

        if (!content) {
          return null;
        }

        return (
          <div
            style={{
              position: "fixed",
              top: `${top}px`,
              left: `${left}px`,
              width: `${cardWidth}px`,
              background: theme.surfaceOverlay,
              border: `1px solid ${accent}`,
              borderRadius: "14px",
              padding: "14px 16px",
              boxShadow: theme.shadow,
              color: theme.text,
              zIndex: 40,
              pointerEvents: "none"
            }}
          >
            {content}
          </div>
        );
      })()}

      {contextMenu && (
        <div
          style={{
            position: "fixed",
            top: `${contextMenu.y}px`,
            left: `${contextMenu.x}px`,
            background: theme.surfaceOverlay,
            border: `1px solid ${theme.button.outline}`,
            borderRadius: "12px",
            padding: "8px 0",
            zIndex: 20,
            minWidth: "180px",
            boxShadow: theme.shadow
          }}
        >
          {(() => {
            const sample = project.samples.find((item) => item.id === contextMenu.sampleId);
            if (!sample) return null;
            return (
              <>
                {sample.measures.length > 0 ? (
                  <button
                    type="button"
                    style={{
                      width: "100%",
                      padding: "10px 14px",
                      background: "transparent",
                      border: "none",
                      color: theme.text,
                      textAlign: "left",
                      fontSize: "0.85rem",
                      cursor: "pointer"
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleCombineMeasures(sample);
                    }}
                  >
                    Combine measures…
                  </button>
                ) : (
                  <button
                    type="button"
                    style={{
                      width: "100%",
                      padding: "10px 14px",
                      background: "transparent",
                      border: "none",
                      color: theme.text,
                      textAlign: "left",
                      fontSize: "0.85rem",
                      cursor: "pointer"
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleSeparateMeasures(sample);
                    }}
                  >
                    Separate into measures
                  </button>
                )}
                {sample.measures.length > 0 && (
                  <button
                    type="button"
                    style={{
                      width: "100%",
                      padding: "10px 14px",
                      background: "transparent",
                      border: "none",
                      color: theme.text,
                      textAlign: "left",
                      fontSize: "0.85rem",
                      cursor: "pointer"
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleSeparateMeasures(sample);
                    }}
                  >
                    Re-split measures
                  </button>
                )}
                <button
                  type="button"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    color: theme.text,
                    textAlign: "left",
                    fontSize: "0.85rem",
                    cursor: "pointer"
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleRekey(sample);
                  }}
                >
                  Re-key to project palette
                </button>
                <button
                  type="button"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    color: theme.text,
                    textAlign: "left",
                    fontSize: "0.85rem",
                    cursor: "pointer"
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleTrim(sample);
                  }}
                >
                  Trim selection…
                </button>
                <button
                  type="button"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    color: "#ff6b6b",
                    textAlign: "left",
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleDeleteSample(sample);
                  }}
                >
                  Delete sample
                </button>
              </>
            );
          })()}
        </div>
      )}

      {measureMenu && (
        <div
          style={{
            position: "fixed",
            top: `${measureMenu.y}px`,
            left: `${measureMenu.x}px`,
            background: theme.surfaceOverlay,
            border: `1px solid ${theme.button.outline}`,
            borderRadius: "12px",
            padding: "8px 0",
            zIndex: 20,
            minWidth: "180px",
            boxShadow: theme.shadow
          }}
        >
          {(() => {
            const sample = project.samples.find((item) => item.id === measureMenu.sampleId);
            const measure = sample?.measures.find((item) => item.id === measureMenu.measureId);
            if (!sample || !measure) return null;
            return (
              <>
                <button
                  type="button"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    color: theme.text,
                    textAlign: "left",
                    fontSize: "0.85rem",
                    cursor: "pointer",
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleSliceMeasure(sample, measure);
                  }}
                >
                  Create measure clip
                </button>
                <button
                  type="button"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: "transparent",
                    border: "none",
                    color: theme.text,
                    textAlign: "left",
                    fontSize: "0.85rem",
                    cursor: "pointer"
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleGenerateBeats(sample, measure);
                  }}
                >
                  Generate beats
                </button>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
