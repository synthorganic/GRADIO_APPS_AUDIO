import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent, MouseEvent as ReactMouseEvent } from "react";
import { motion } from "framer-motion";
import { useProjectStore } from "../state/ProjectStore";
import type { Project, SampleClip, StemInfo } from "../types";
import { audioEngine } from "../lib/audioEngine";
import { useDemucsProcessing } from "../hooks/useDemucsProcessing";
import { theme } from "../theme";
import { sliceSampleSegment } from "../lib/sampleTools";
import { TrackEffectsPanel } from "./TrackEffectsPanel";

const TRACK_HEIGHT = 120;
const MEASURE_WIDTH = 140;

type SnapResolution = "measure" | "half-measure" | "beat" | "half-beat";

const SNAP_DIVISORS: Record<SnapResolution, number> = {
  measure: 1,
  "half-measure": 2,
  beat: 4,
  "half-beat": 8
};

interface TimelineProps {
  project: Project;
  selectedSampleId: string | null;
  onSelectSample: (id: string | null) => void;
}

function measureDurationSeconds(project: Project) {
  return (60 / project.masterBpm) * 4;
}

export function Timeline({ project, selectedSampleId, onSelectSample }: TimelineProps) {
  const { dispatch, currentProjectId } = useProjectStore();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [snapResolution, setSnapResolution] = useState<SnapResolution>("measure");
  const [clipMenu, setClipMenu] = useState<{ sampleId: string; x: number; y: number } | null>(null);
  const [playhead, setPlayhead] = useState<{ start: number; position: number } | null>(null);
  const [effectsSampleId, setEffectsSampleId] = useState<string | null>(null);
  const { processSample } = useDemucsProcessing((updated) => {
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: updated.id,
      sample: updated
    });
  });

  const secondsPerMeasure = useMemo(() => measureDurationSeconds(project), [project.masterBpm]);

  const totalDuration = useMemo(() => {
    const maxEnd = project.samples.reduce(
      (acc, sample) => Math.max(acc, sample.position + sample.length),
      0
    );
    return Math.max(maxEnd, 16);
  }, [project.samples]);

  const timelineGrid = useMemo(() => {
    const divisor = SNAP_DIVISORS[snapResolution];
    const ticks: Array<{ position: number; accent: number }> = [];
    const measures = Math.ceil(totalDuration / secondsPerMeasure);
    for (let measure = 0; measure < measures * divisor; measure++) {
      ticks.push({ position: measure / divisor, accent: measure % divisor });
    }
    return ticks;
  }, [secondsPerMeasure, snapResolution, totalDuration]);

  useEffect(() => {
    const handlePlay = (event: Event) => {
      const detail = (event as CustomEvent<{ timelineOffset?: number }>).detail;
      setPlayhead({ start: detail?.timelineOffset ?? 0, position: 0 });
    };
    const handleStop = () => setPlayhead(null);
    const handleTick = (event: Event) => {
      const detail = (event as CustomEvent<{ timelineOffset: number; position: number }>).detail;
      setPlayhead({ start: detail.timelineOffset, position: detail.position });
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
    const stillExists = project.samples.some((sample) => sample.id === effectsSampleId);
    if (!stillExists) {
      setEffectsSampleId(null);
    }
  }, [effectsSampleId, project.samples]);

  const onDrag = (sample: SampleClip, deltaX: number) => {
    const measuresMoved = deltaX / MEASURE_WIDTH;
    const newMeasurePosition = Math.max(0, sample.position / secondsPerMeasure + measuresMoved);
    const divisor = SNAP_DIVISORS[snapResolution];
    const quantisedMeasures = Math.round(newMeasurePosition * divisor) / divisor;
    const newPositionSeconds = quantisedMeasures * secondsPerMeasure;
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: { position: newPositionSeconds }
    });
  };

  const projectDropPosition = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const container = scrollRef.current;
      if (!container) return 0;
      const bounds = container.getBoundingClientRect();
      const relativeX = event.clientX - bounds.left + container.scrollLeft;
      const measuresFromStart = Math.max(0, relativeX / MEASURE_WIDTH);
      const divisor = SNAP_DIVISORS[snapResolution];
      const snappedMeasures = Math.round(measuresFromStart * divisor) / divisor;
      return snappedMeasures * secondsPerMeasure;
    },
    [secondsPerMeasure, snapResolution]
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const dropPosition = projectDropPosition(event);
      const measureData = event.dataTransfer.getData("application/x-measure");
      const sampleData = event.dataTransfer.getData("application/x-sample");
      const beatData = event.dataTransfer.getData("application/x-beat");
      const stemData = event.dataTransfer.getData("application/x-stem");
      const stemFragmentData = event.dataTransfer.getData("application/x-stem-fragment");

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
            variantLabel: `Measure ${measureIndex + 1}`
          });
          fragment.position = dropPosition;
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
            variantLabel: `Beat ${beat.index + 1}`
          });
          fragment.position = dropPosition;
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
          const stem = sourceSample?.stems.find((item) => item.id === payload.stemId || item.sourceStemId === payload.stemId);
          if (!sourceSample || !stem) return;
          const fragment = sliceSampleSegment(sourceSample, payload.start, payload.end, {
            variantLabel: stem.name,
            stems: [stem]
          });
          fragment.position = dropPosition;
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
            stems: [stem]
          });
          fragment.position = dropPosition;
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
          sample: { position: dropPosition }
        });
        onSelectSample(sample.id);
      }
    },
    [currentProjectId, dispatch, onSelectSample, processSample, project.samples, projectDropPosition]
  );

  useEffect(() => {
    const closeMenu = () => setClipMenu(null);
    window.addEventListener("click", closeMenu);
    return () => window.removeEventListener("click", closeMenu);
  }, []);

  const handleClipContextMenu = (event: ReactMouseEvent, sampleId: string) => {
    event.preventDefault();
    event.stopPropagation();
    setClipMenu({ sampleId, x: event.clientX, y: event.clientY });
  };

  const handleRemoveClip = (sampleId: string) => {
    setClipMenu(null);
    dispatch({ type: "remove-sample", projectId: currentProjectId, sampleId });
    if (selectedSampleId === sampleId) {
      onSelectSample(null);
    }
    if (effectsSampleId === sampleId) {
      setEffectsSampleId(null);
    }
  };

  const playClip = (sample: SampleClip) => {
    void audioEngine.play(sample, sample.measures, { timelineOffset: sample.position });
  };

  const clipGradient = (stems: StemInfo[]) => {
    if (stems.length === 0) return theme.surface;
    const stops = stems.map((stem, index) => {
      const percentStart = (index / stems.length) * 100;
      const percentEnd = ((index + 1) / stems.length) * 100;
      return `${stem.color}33 ${percentStart}%, ${stem.color}55 ${percentEnd}%`;
    });
    return `linear-gradient(90deg, ${stops.join(", ")})`;
  };

  const clipOutline = (stems: StemInfo[]) => stems[0]?.color ?? theme.button.primary;

  return (
    <div
      style={{
        flex: 1,
        background: theme.surfaceOverlay,
        borderRadius: "18px",
        padding: "18px",
        display: "grid",
        gap: "18px",
        gridTemplateRows: "auto 1fr",
        border: `1px solid ${theme.border}`,
        boxShadow: theme.shadow,
        color: theme.text
      }}
    >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h2 style={{ margin: 0, fontSize: "1.1rem", color: theme.text, letterSpacing: "0.06em" }}>
              Chromatic timeline
            </h2>
            <p style={{ margin: 0, fontSize: "0.82rem", color: theme.textMuted }}>
            Drag samples, measures, beats, or stems here to arrange your collage. Clips adopt their stem
            colors for quick recognition.
          </p>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "center", color: theme.text }}>
            <label
              style={{ fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "8px", color: theme.text }}
            >
              Master BPM
              <input
                type="number"
                value={project.masterBpm}
                onChange={(event) =>
                  dispatch({
                    type: "set-project",
                    project: { ...project, masterBpm: Number(event.target.value) }
                  })
                }
                style={{
                  padding: "6px 10px",
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  background: theme.surface,
                  color: theme.text,
                  width: "88px"
                }}
              />
            </label>
            <label
              style={{ fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "8px", color: theme.text }}
            >
              Snap to
              <select
                value={snapResolution}
                onChange={(event) => setSnapResolution(event.target.value as SnapResolution)}
                style={{
                  padding: "6px 10px",
                  borderRadius: "8px",
                  border: `1px solid ${theme.border}`,
                  background: theme.surface,
                  color: theme.text
                }}
              >
                <option value="measure">Measure</option>
                <option value="half-measure">1/2 measure</option>
                <option value="beat">Beat</option>
                <option value="half-beat">1/2 beat</option>
              </select>
            </label>
          </div>
        </div>

      <div
        ref={scrollRef}
        style={{ position: "relative", overflow: "auto", borderRadius: "14px" }}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDrop={handleDrop}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "24px",
            display: "flex"
          }}
        >
          {timelineGrid.map((tick, index) => (
            <div
              key={`${tick.position}-${index}`}
              style={{
                flex: "0 0 auto",
                width: `${MEASURE_WIDTH / SNAP_DIVISORS[snapResolution]}px`,
                borderRight:
                  tick.accent === 0
                    ? `2px solid ${theme.button.outline}`
                    : `1px dashed ${theme.button.outline}33`,
                opacity: tick.accent === 0 ? 0.9 : 0.4
              }}
            ></div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "18px",
            paddingTop: "38px",
            paddingBottom: "28px",
            position: "relative"
          }}
        >
          {project.samples.map((sample, index) => {
            const clipLabel = sample.variantLabel ? `${sample.name} — ${sample.variantLabel}` : sample.name;
            const widthMeasures = sample.length / secondsPerMeasure || 1;
            const clipWidth = Math.max(120, widthMeasures * MEASURE_WIDTH);
            const colorStripe = clipOutline(sample.stems);
            return (
              <motion.div
                key={sample.id}
                drag="x"
                dragMomentum={false}
                onDragEnd={(event, info) => onDrag(sample, info.offset.x)}
                style={{
                  position: "relative",
                  height: `${TRACK_HEIGHT}px`,
                  borderRadius: "18px",
                  background: theme.surface,
                  cursor: "grab",
                  overflow: "hidden",
                  border: `2px solid ${selectedSampleId === sample.id ? theme.button.primary : colorStripe}`,
                  boxShadow: selectedSampleId === sample.id ? theme.cardGlow : "none",
                  transform: `translateX(${(sample.position / secondsPerMeasure) * MEASURE_WIDTH}px)`,
                  width: `${clipWidth}px`
                }}
                onClick={() => onSelectSample(sample.id)}
                onContextMenu={(event) => handleClipContextMenu(event, sample.id)}
              >
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: clipGradient(sample.stems)
                  }}
                ></div>
                <div
                  style={{
                    position: "relative",
                    zIndex: 1,
                    padding: "14px 18px",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "flex-start",
                    height: "100%",
                    color: theme.text
                  }}
                >
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <strong style={{ fontSize: "1rem" }}>{clipLabel}</strong>
                    <div
                      style={{ fontSize: "0.78rem", color: theme.textMuted, display: "flex", gap: "10px" }}
                    >
                      <span>{sample.bpm ? `${sample.bpm} BPM` : "Analyzing…"}</span>
                      <span>{sample.key ? `Key ${sample.key}` : "Key pending"}</span>
                      <span>{sample.length ? `${sample.length.toFixed(2)}s` : "Length TBD"}</span>
                    </div>
                    <div style={{ display: "flex", gap: "6px" }}>
                      {sample.stems.map((stem) => (
                        <span
                          key={`${sample.id}-${stem.id}-chip`}
                          style={{
                            width: "12px",
                            height: "12px",
                            borderRadius: "50%",
                            background: stem.color,
                            boxShadow: "0 0 8px rgba(0,0,0,0.35)"
                          }}
                        ></span>
                      ))}
                    </div>
                    {sample.retuneMap && (
                      <div style={{ fontSize: "0.72rem", color: theme.button.primary }}>
                        {sample.retuneMap.join(" · ")}
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "10px" }}>
                    <span style={{ fontSize: "0.78rem", color: theme.textMuted }}>Track {index + 1}</span>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        setEffectsSampleId(sample.id);
                      }}
                      style={{
                        borderRadius: "999px",
                        padding: "6px 14px",
                        border: `1px solid ${theme.button.outline}`,
                        background: theme.surfaceOverlay,
                        color: theme.text,
                        cursor: "pointer",
                        fontSize: "0.78rem",
                        letterSpacing: "0.05em"
                      }}
                    >
                      Effects
                    </button>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        playClip(sample);
                      }}
                      style={{
                        border: `1px solid ${theme.button.outline}`,
                        borderRadius: "999px",
                        padding: "6px 12px",
                        background: theme.button.base,
                        color: theme.text,
                        cursor: "pointer"
                      }}
                    >
                      Play clip
                    </button>
                  </div>
                </div>
              </motion.div>
            );
          })}

          {playhead && (
            <>
              <div
                style={{
                  position: "absolute",
                  top: "24px",
                  bottom: "24px",
                  left: `${(playhead.start / secondsPerMeasure) * MEASURE_WIDTH}px`,
                  width: "2px",
                  background: `${theme.button.outline}80`
                }}
              ></div>
              <div
                style={{
                  position: "absolute",
                  top: "24px",
                  bottom: "24px",
                  left: `${((playhead.start + playhead.position) / secondsPerMeasure) * MEASURE_WIDTH}px`,
                  width: "3px",
                  background: theme.button.primary,
                  boxShadow: `0 0 10px ${theme.button.primary}`
                }}
              ></div>
            </>
          )}
        </div>
      </div>

      {clipMenu && (
        <div
          style={{
            position: "fixed",
            top: `${clipMenu.y}px`,
            left: `${clipMenu.x}px`,
            background: theme.surfaceOverlay,
            border: `1px solid ${theme.button.outline}`,
            borderRadius: "12px",
            padding: "8px 0",
            zIndex: 30,
            minWidth: "160px",
            boxShadow: theme.shadow
          }}
        >
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
            onClick={() => handleRemoveClip(clipMenu.sampleId)}
          >
            Remove from timeline
          </button>
        </div>
      )}
      {effectsSampleId && (
        (() => {
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
                  sample: { effects }
                })
              }
            />
          );
        })()
      )}
    </div>
  );
}
