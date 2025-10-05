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
          isInTimeline: true,
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
    await audioEngine.play(sample, sample.measures, { emitTimelineEvents: false });
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
    await audioEngine.playSegment(sample, measure.start, segmentDuration, { emitTimelineEvents: false });
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
    await audioEngine.playStem(sample, stem, sample.measures, { emitTimelineEvents: false });
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
    await audioEngine.playSegment(sample, beat.start, duration, { emitTimelineEvents: false });
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
        gap: "10px",
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
          gap: "6px",
          padding: 0,
          margin: 0,
          overflowY: "auto"
        }}
      >
        {groupedSamples.map((sample) => {
          const expanded = expandedSamples[sample.id] ?? false;
          const isSelected = selectedSampleId === sample.id;
          return (
            <div
              key={sample.id}
              style={{
                background: isSelected ? theme.surfaceRaised : theme.surfaceOverlay,
                borderRadius: "6px",
                border: `1px solid ${isSelected ? theme.button.primary : theme.border}`,
                boxShadow: isSelected ? theme.cardGlow : "none",
                transition: "border 0.2s ease, box-shadow 0.2s ease",
                overflow: "hidden"
              }}
              onContextMenu={(event) => onSampleContextMenu(event, sample.id)}
              onClick={() => onSelectSample(sample.id)}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "4px 6px",
                  gap: "6px",
                  fontSize: "0.68rem"
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
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
                      width: "16px",
                      height: "16px",
                      borderRadius: "4px",
                      border: `1px solid ${theme.button.outline}`,
                      background: theme.surface,
                      color: theme.text,
                      fontSize: "0.55rem",
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
                      fontSize: "0.7rem",
                      color: theme.text,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis"
                    }}
                  >
                    {sample.name}
                  </strong>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      void toggleSamplePlayback(sample);
                    }}
                    style={{
                      width: "24px",
                      height: "24px",
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
                      width: "22px",
                      height: "22px",
                      borderRadius: "50%",
                      border: `1px solid ${theme.button.outline}`,
                      background: theme.surface,
                      color: theme.textMuted,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      padding: 0,
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
                    padding: "8px 10px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px"
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "4px"
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "8px 10px",
                        borderRadius: "8px",
                        border: `1px solid ${theme.border}`,
                        background: theme.surface
                      }}
                      draggable
                      onDragStart={(event) => {
                        event.dataTransfer.effectAllowed = "copy";
                        event.dataTransfer.setData("application/x-sample", sample.id);
                      }}
                    >
                      <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                        <strong style={{ fontSize: "0.75rem", color: theme.text }}>Full sample</strong>
                        <span style={{ fontSize: "0.65rem", color: theme.textMuted }}>
                          {sample.length.toFixed(2)}s • {sample.measures.length} measures
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void toggleSamplePlayback(sample);
                        }}
                        style={{
                          width: "26px",
                          height: "26px",
                          borderRadius: "50%",
                          border: `1px solid ${theme.button.outline}`,
                          background: playingId === sample.id ? theme.button.primary : theme.button.base,
                          color: playingId === sample.id ? theme.button.primaryText : theme.text,
                          cursor: "pointer",
                          fontSize: "0.6rem",
                          boxShadow: playingId === sample.id ? theme.cardGlow : "none"
                        }}
                      >
                        {playingId === sample.id ? "■" : "▶"}
                      </button>
                    </div>
                  </div>

                  <details open>
                    <summary
                      style={{
                        cursor: "pointer",
                        fontSize: "0.72rem",
                        color: theme.text,
                        listStyle: "none"
                      }}
                    >
                      Full stems
                    </summary>
                    <ul
                      style={{
                        listStyle: "none",
                        margin: "6px 0 0",
                        padding: 0,
                        display: "flex",
                        flexDirection: "column",
                        gap: "6px"
                      }}
                    >
                      {sample.stems.map((stem) => (
                        <li
                          key={stem.id}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "6px 8px",
                            borderRadius: "8px",
                            border: `1px solid ${theme.border}`,
                            background: `${stem.color}1a`
                          }}
                          draggable
                          onDragStart={(event) => {
                            event.dataTransfer.effectAllowed = "copy";
                            event.dataTransfer.setData(
                              "application/x-stem",
                              JSON.stringify({ sampleId: sample.id, stemId: stem.id })
                            );
                          }}
                        >
                          <span style={{ display: "flex", alignItems: "center", gap: "8px", color: theme.text }}>
                            <span
                              style={{
                                width: "8px",
                                height: "8px",
                                borderRadius: "50%",
                                background: stem.color
                              }}
                            ></span>
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
                              background: playingId === stem.id ? theme.button.primary : theme.button.base,
                              color: playingId === stem.id ? theme.button.primaryText : theme.text,
                              cursor: "pointer",
                              fontSize: "0.6rem"
                            }}
                          >
                            {playingId === stem.id ? "■" : "▶"}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </details>

                  <details open>
                    <summary
                      style={{
                        cursor: "pointer",
                        fontSize: "0.72rem",
                        color: theme.text,
                        listStyle: "none"
                      }}
                    >
                      Measures
                    </summary>
                    <ul
                      style={{
                        listStyle: "none",
                        margin: "10px 0 0",
                        padding: 0,
                        display: "flex",
                        flexDirection: "column",
                        gap: "10px"
                      }}
                    >
                      {sample.measures.length === 0 && (
                        <li style={{ fontSize: "0.78rem", color: theme.textMuted }}>
                          Waiting for measure detection…
                        </li>
                      )}
                      {sample.measures.map((measure, index) => {
                        const duration = getMeasureDuration(measure);
                        const tunedPitch = measure.tunedPitch ?? measure.detectedPitch;
                        const hasRetune =
                          measure.tunedPitch &&
                          measure.detectedPitch &&
                          measure.tunedPitch !== measure.detectedPitch;
                        const measureStems = sliceStemsForRange(sample, measure.start, measure.end);
                        return (
                          <li
                            key={measure.id}
                            style={{
                              border: `1px solid ${theme.border}`,
                              borderRadius: "10px",
                              background: theme.surface,
                              padding: "10px 12px",
                              display: "flex",
                              flexDirection: "column",
                              gap: "10px"
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
                                gap: "12px"
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
                            >
                              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        <strong style={{ fontSize: "0.72rem", color: theme.text }}>
                                  Measure {index + 1}
                                </strong>
                                <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>
                                  {measure.start.toFixed(2)}s → {measure.end.toFixed(2)}s ·
                                  {duration.toFixed(2)}s · {measure.beatCount} beats
                                </span>
                                <span style={{ fontSize: "0.72rem", color: theme.button.primary }}>
                                  {measure.detectedPitch ?? "?"} → {tunedPitch}
                                  {hasRetune ? " (re-keyed)" : ""}
                                </span>
                              </div>
                              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                                  Energy {Math.round((measure.energy ?? 0) * 100)}%
                                </span>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void playMeasure(sample, measure);
                                  }}
                                  style={{
                                    width: "30px",
                                    height: "30px",
                                    borderRadius: "50%",
                                    border: `1px solid ${theme.button.outline}`,
                                    background: playingId === measure.id
                                      ? theme.button.primary
                                      : theme.button.base,
                                    color: playingId === measure.id
                                      ? theme.button.primaryText
                                      : theme.text,
                                    cursor: "pointer"
                                  }}
                                >
                                  {playingId === measure.id ? "■" : "▶"}
                                </button>
                              </div>
                            </div>

                            <details>
                              <summary
                                style={{
                                  cursor: "pointer",
                                  fontSize: "0.78rem",
                                  color: theme.text,
                                  listStyle: "none"
                                }}
                              >
                                Measure stems
                              </summary>
                              <ul
                                style={{
                                  listStyle: "none",
                                  margin: "10px 0 0",
                                  padding: 0,
                                  display: "flex",
                                  flexDirection: "column",
                                  gap: "6px"
                                }}
                              >
                                {measureStems.map((stem) => (
                                  <li
                                    key={stem.id}
                                    style={{
                                      display: "flex",
                                      justifyContent: "space-between",
                                      alignItems: "center",
                                      padding: "6px 8px",
                                      borderRadius: "8px",
                                      border: `1px solid ${theme.border}`,
                                      background: `${stem.color}1f`
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
                                  >
                                    <span
                                      style={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: "8px",
                                        color: theme.text
                                      }}
                                    >
                                      <span
                                        style={{
                                          width: "8px",
                                          height: "8px",
                                          borderRadius: "50%",
                                          background: stem.color
                                        }}
                                      ></span>
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
                                        background: playingId === stem.id
                                          ? theme.button.primary
                                          : theme.button.base,
                                        color: playingId === stem.id
                                          ? theme.button.primaryText
                                          : theme.text,
                                        cursor: "pointer"
                                      }}
                                    >
                                      {playingId === stem.id ? "■" : "▶"}
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            </details>

                            {measure.beats && measure.beats.length > 0 && (
                              <details>
                                <summary
                                  style={{
                                    cursor: "pointer",
                                    fontSize: "0.78rem",
                                    color: theme.text,
                                    listStyle: "none"
                                  }}
                                >
                                  Beats
                                </summary>
                                <ul
                                  style={{
                                    listStyle: "none",
                                    margin: "10px 0 0",
                                    padding: 0,
                                    display: "flex",
                                    flexDirection: "column",
                                    gap: "8px"
                                  }}
                                >
                                  {measure.beats.map((beat) => (
                                    <li
                                      key={beat.id}
                                      style={{
                                        border: `1px solid ${theme.border}`,
                                        borderRadius: "8px",
                                        padding: "8px 10px",
                                        background: theme.surfaceOverlay,
                                        display: "flex",
                                        flexDirection: "column",
                                        gap: "8px"
                                      }}
                                    >
                                      <div
                                        style={{
                                          display: "flex",
                                          alignItems: "center",
                                          justifyContent: "space-between",
                                          gap: "8px"
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
                                        <span style={{ fontSize: "0.76rem", color: theme.text }}>
                                          Beat {beat.index + 1} • {(beat.end - beat.start).toFixed(2)}s
                                        </span>
                                        <button
                                          type="button"
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            void playBeat(sample, beat);
                                          }}
                                          style={{
                                            width: "24px",
                                            height: "24px",
                                            borderRadius: "50%",
                                            border: `1px solid ${theme.button.outline}`,
                                            background: playingId === beat.id
                                              ? theme.button.primary
                                              : theme.button.base,
                                            color: playingId === beat.id
                                              ? theme.button.primaryText
                                              : theme.text,
                                            cursor: "pointer"
                                          }}
                                        >
                                          {playingId === beat.id ? "■" : "▶"}
                                        </button>
                                      </div>
                                      <details>
                                        <summary
                                          style={{
                                            cursor: "pointer",
                                            fontSize: "0.74rem",
                                            color: theme.text,
                                            listStyle: "none"
                                          }}
                                        >
                                          Beat stems
                                        </summary>
                                        <ul
                                          style={{
                                            listStyle: "none",
                                            margin: "8px 0 0",
                                            padding: 0,
                                            display: "flex",
                                            flexDirection: "column",
                                            gap: "6px"
                                          }}
                                        >
                                          {beat.stems.map((stem) => (
                                            <li
                                              key={stem.id}
                                              style={{
                                                display: "flex",
                                                justifyContent: "space-between",
                                                alignItems: "center",
                                                padding: "6px 8px",
                                                borderRadius: "8px",
                                                border: `1px solid ${theme.border}`,
                                                background: `${stem.color}24`
                                              }}
                                              draggable
                                              onDragStart={(event) => {
                                                event.dataTransfer.effectAllowed = "copy";
                                                event.dataTransfer.setData(
                                                  "application/x-stem-fragment",
                                                  JSON.stringify({
                                                    sampleId: sample.id,
                                                    stemId: stem.sourceStemId ?? stem.id,
                                                    start: beat.start,
                                                    end: beat.end
                                                  })
                                                );
                                              }}
                                            >
                                              <span
                                                style={{
                                                  display: "flex",
                                                  alignItems: "center",
                                                  gap: "8px",
                                                  color: theme.text
                                                }}
                                              >
                                                <span
                                                  style={{
                                                    width: "8px",
                                                    height: "8px",
                                                    borderRadius: "50%",
                                                    background: stem.color
                                                  }}
                                                ></span>
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
                                                  background: playingId === stem.id
                                                    ? theme.button.primary
                                                    : theme.button.base,
                                                  color: playingId === stem.id
                                                    ? theme.button.primaryText
                                                    : theme.text,
                                                  cursor: "pointer"
                                                }}
                                              >
                                                {playingId === stem.id ? "■" : "▶"}
                                              </button>
                                            </li>
                                          ))}
                                        </ul>
                                      </details>
                                    </li>
                                  ))}
                                </ul>
                              </details>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                </div>
              )}
            </div>
          );
        })}
      </div>

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
