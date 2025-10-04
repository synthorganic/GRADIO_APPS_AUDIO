import { useCallback, useMemo, useRef } from "react";
import type { DragEvent } from "react";
import { motion } from "framer-motion";
import { nanoid } from "nanoid";
import { useProjectStore } from "../state/ProjectStore";
import type { Measure, Project, SampleClip } from "../types";
import { audioEngine } from "../lib/audioEngine";
import { useDemucsProcessing } from "../hooks/useDemucsProcessing";
import { theme } from "../theme";

const TRACK_HEIGHT = 148;
const MEASURE_WIDTH = 140;

interface TimelineProps {
  project: Project;
  selectedSampleId: string | null;
  onSelectSample: (id: string | null) => void;
}

const rainbowOrder: Array<{ key: SampleClip["stems"][number]["type"] | "full"; label: string }> = [
  { key: "full", label: "Full mix" },
  { key: "vocals", label: "Vocals" },
  { key: "leads", label: "Leads" },
  { key: "percussion", label: "High Drums" },
  { key: "kicks", label: "Kicks" },
  { key: "bass", label: "Bassline" }
];

function measureDurationSeconds(project: Project) {
  return (60 / project.masterBpm) * 4;
}

function createMeasureFragment(
  source: SampleClip,
  measure: Measure,
  positionSeconds: number,
  measureIndex: number
): SampleClip {
  const duration = Math.max(0, measure.end - measure.start);
  const offset = (source.startOffset ?? 0) + measure.start;
  return {
    id: nanoid(),
    name: `${source.name}`,
    variantLabel: `Measure ${measureIndex + 1}`,
    file: source.file,
    url: source.url,
    bpm: source.bpm,
    key: source.key,
    measures: [
      {
        ...measure,
        id: nanoid(),
        start: 0,
        end: duration
      }
    ],
    stems: source.stems.map((stem) => ({
      ...stem,
      id: nanoid(),
      sourceStemId: stem.id,
      startOffset: (stem.startOffset ?? source.startOffset ?? 0) + measure.start,
      duration
    })),
    position: positionSeconds,
    length: duration,
    isLooping: source.isLooping,
    startOffset: offset,
    duration,
    originSampleId: source.id,
    isFragment: true,
    retuneMap: source.retuneMap,
    rekeyedAt: source.rekeyedAt
  };
}

export function Timeline({ project, selectedSampleId, onSelectSample }: TimelineProps) {
  const { dispatch, currentProjectId, getPalette } = useProjectStore();
  const palette = getPalette();
  const scrollRef = useRef<HTMLDivElement | null>(null);
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
    const beatsPerMeasure = 4;
    const ticks: Array<{ position: number; isMeasure: boolean }> = [];
    const totalBeats = Math.ceil(totalDuration * beatsPerMeasure);
    for (let beat = 0; beat <= totalBeats; beat++) {
      ticks.push({ position: beat / beatsPerMeasure, isMeasure: beat % beatsPerMeasure === 0 });
    }
    return ticks;
  }, [totalDuration]);

  const onDrag = (sample: SampleClip, deltaX: number) => {
    const measuresMoved = deltaX / MEASURE_WIDTH;
    const newMeasurePosition = Math.max(0, sample.position / secondsPerMeasure + measuresMoved);
    const quantisedMeasures = Math.round(newMeasurePosition * 4) / 4;
    const newPositionSeconds = quantisedMeasures * secondsPerMeasure;
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: sample.id,
      sample: { position: newPositionSeconds }
    });
  };

  const handleStripeClick = (
    sample: SampleClip,
    type: SampleClip["stems"][number]["type"] | "full"
  ) => {
    if (type === "full") {
      void audioEngine.play(sample, sample.measures);
      return;
    }
    const target = sample.stems.find((stem) => stem.type === type);
    if (target) {
      void audioEngine.play(target, sample.measures);
    }
  };

  const projectDropPosition = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      const container = scrollRef.current;
      if (!container) return 0;
      const bounds = container.getBoundingClientRect();
      const relativeX = event.clientX - bounds.left + container.scrollLeft;
      const measuresFromStart = Math.max(0, relativeX / MEASURE_WIDTH);
      return measuresFromStart * secondsPerMeasure;
    },
    [secondsPerMeasure]
  );

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const dropPosition = projectDropPosition(event);
      const measureData = event.dataTransfer.getData("application/x-measure");
      const sampleData = event.dataTransfer.getData("application/x-sample");

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
          const fragment = createMeasureFragment(sourceSample, measure, dropPosition, measureIndex);
          dispatch({ type: "add-sample", projectId: currentProjectId, sample: fragment });
          onSelectSample(fragment.id);
          return;
        } catch (error) {
          console.warn("Failed to parse measure drop payload", error);
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
    [currentProjectId, dispatch, onSelectSample, project.samples, projectDropPosition]
  );

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
            Drag samples or individual measures here to arrange your collage. Click rainbow stripes to
            audition stems.
          </p>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", color: theme.text }}>
          <label style={{ fontSize: "0.82rem", display: "flex", alignItems: "center", gap: "8px", color: theme.text }}>
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
          {timelineGrid.map((tick) => (
            <div
              key={`${tick.position}-${tick.isMeasure}`}
              style={{
                flex: "0 0 auto",
                width: tick.isMeasure ? `${MEASURE_WIDTH}px` : `${MEASURE_WIDTH / 4}px`,
                borderRight: tick.isMeasure
                  ? `2px solid ${theme.button.outline}`
                  : "1px dashed rgba(122,116,255,0.2)",
                opacity: tick.isMeasure ? 0.8 : 0.4
              }}
            ></div>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "18px", paddingTop: "38px", paddingBottom: "28px" }}>
          {project.samples.map((sample, index) => {
            const measures = Math.max(sample.measures.length, 1);
            const clipLabel = sample.variantLabel ? `${sample.name} — ${sample.variantLabel}` : sample.name;
            return (
              <motion.div
                key={sample.id}
                drag="x"
                dragMomentum={false}
                onDragEnd={(event, info) => onDrag(sample, info.point.x - info.offset.x)}
                style={{
                  position: "relative",
                  height: `${TRACK_HEIGHT}px`,
                  borderRadius: "18px",
                  background: selectedSampleId === sample.id ? theme.surfaceRaised : theme.surface,
                  cursor: "grab",
                  overflow: "hidden",
                  border: selectedSampleId === sample.id
                    ? `1px solid ${theme.button.primary}`
                    : `1px solid ${theme.border}`,
                  boxShadow: selectedSampleId === sample.id ? theme.cardGlow : "none",
                  transform: `translateX(${(sample.position / secondsPerMeasure) * MEASURE_WIDTH}px)`
                }}
                onClick={() => onSelectSample(sample.id)}
              >
                <div
                  style={{
                    padding: "14px 18px",
                    display: "flex",
                    justifyContent: "space-between",
                    color: theme.text
                  }}
                >
                  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                    <strong style={{ fontSize: "1rem" }}>{clipLabel}</strong>
                    <div
                      style={{ fontSize: "0.78rem", color: theme.textMuted, display: "flex", gap: "10px" }}
                    >
                      <span>{sample.bpm ? `${sample.bpm} BPM` : "Analyzing…"}</span>
                      <span>{sample.key ? `Key ${sample.key}` : "Key pending"}</span>
                      <span>{sample.measures.length ? `${sample.measures.length} measures` : "Detecting"}</span>
                    </div>
                    {sample.retuneMap && (
                      <div style={{ fontSize: "0.72rem", color: theme.button.primary }}>
                        {sample.retuneMap.join(" · ")}
                      </div>
                    )}
                  </div>
                  <span style={{ fontSize: "0.78rem", color: theme.textMuted }}>Track {index + 1}</span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: `repeat(${measures}, ${MEASURE_WIDTH}px)`,
                    gridAutoFlow: "column",
                    width: `${measures * MEASURE_WIDTH}px`
                  }}
                >
                  {(sample.measures.length ? sample.measures : [null]).map((measure, measureIndex) => (
                    <div key={measure ? measure.id : `${sample.id}-ghost-${measureIndex}`} style={{ display: "grid", gridTemplateRows: `repeat(${rainbowOrder.length}, 1fr)` }}>
                      {rainbowOrder.map((stripe) => {
                        const color = palette[stripe.key as keyof typeof palette] ?? theme.accentBeam[0];
                        return (
                          <button
                            key={`${measure ? measure.id : `ghost-${measureIndex}`}-${stripe.key}`}
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              handleStripeClick(sample, stripe.key);
                            }}
                            style={{
                              border: "none",
                              background: `${color}cc`,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              color: theme.background,
                              fontWeight: 600,
                              fontSize: "0.72rem",
                              cursor: "pointer",
                              transition: "filter 0.2s ease"
                            }}
                            onMouseEnter={(event) => {
                              event.currentTarget.style.filter = "brightness(1.1)";
                            }}
                            onMouseLeave={(event) => {
                              event.currentTarget.style.filter = "brightness(1)";
                            }}
                          >
                            {stripe.label}
                          </button>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
