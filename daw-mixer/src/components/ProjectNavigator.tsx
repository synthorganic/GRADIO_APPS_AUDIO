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
import type { Measure, SampleClip } from "../types";
import { useDemucsProcessing } from "../hooks/useDemucsProcessing";
import { theme } from "../theme";

interface ProjectNavigatorProps {
  selectedSampleId: string | null;
  onSelectSample: (id: string | null) => void;
}

const paletteOrder: Array<SampleClip["stems"][number]["type"]> = [
  "full",
  "vocals",
  "leads",
  "percussion",
  "kicks",
  "bass"
];

interface ContextMenuState {
  sampleId: string;
  x: number;
  y: number;
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

  useEffect(() => {
    const closeMenu = () => setContextMenu(null);
    window.addEventListener("click", closeMenu);
    return () => {
      window.removeEventListener("click", closeMenu);
      audioEngine.stop();
    };
  }, []);

  const addSamples = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      let indexOffset = 0;
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
          isFragment: false
        };
        dispatch({ type: "add-sample", projectId: currentProjectId, sample });
        void processSample(sample);
        indexOffset += 1;
      }
    },
    [currentProjectId, dispatch, processSample, project.samples.length]
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
    await audioEngine.play(sample, sample.measures);
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
    await audioEngine.playSegment(sample, measure.start, segmentDuration);
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

  return (
    <div
      style={{
        flex: 1,
        overflow: "hidden",
        padding: "18px 22px",
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
          padding: "18px",
          border: `1px dashed ${theme.button.outline}`,
          borderRadius: "14px",
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
        <p style={{ margin: "0 0 10px", fontWeight: 600, letterSpacing: "0.04em", color: theme.text }}>
          Drop audio or browse to start a technicolor session
        </p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          style={{
            border: `1px solid ${theme.button.outline}`,
            background: theme.button.primary,
            color: theme.button.primaryText,
            padding: "10px 20px",
            borderRadius: "999px",
            fontWeight: 600,
            cursor: "pointer",
            boxShadow: theme.cardGlow
          }}
        >
          Select audio
        </button>
        <p style={{ margin: "12px 0 0", fontSize: "0.85rem", color: theme.textMuted }}>
          Samples auto-stretch to {project.masterBpm} BPM. DEMUCS will unfold stems on demand.
        </p>
        {isProcessing && (
          <span
            style={{
              position: "absolute",
              top: "12px",
              right: "14px",
              fontSize: "0.75rem",
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
          fontSize: "0.95rem",
          letterSpacing: "0.08em",
          color: theme.text,
          textTransform: "uppercase"
        }}
      >
        Project navigator
      </h2>

      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: 0,
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          overflowY: "auto"
        }}
      >
        {groupedSamples.map((sample) => {
          const expanded = expandedSamples[sample.id] ?? false;
          const isSelected = selectedSampleId === sample.id;
          const paletteChip = paletteOrder
            .map((type) => palette[type])
            .filter((color): color is string => Boolean(color));
          return (
            <li
              key={sample.id}
              style={{
                background: isSelected ? theme.surfaceRaised : theme.surfaceOverlay,
                borderRadius: "14px",
                border: `1px solid ${isSelected ? theme.button.primary : theme.border}`,
                padding: "14px 16px",
                boxShadow: isSelected ? theme.cardGlow : "none",
                transition: "border 0.2s ease, box-shadow 0.2s ease"
              }}
              onContextMenu={(event) => onSampleContextMenu(event, sample.id)}
              onClick={() => onSelectSample(sample.id)}
              draggable
              onDragStart={(event) => {
                event.dataTransfer.effectAllowed = "copy";
                event.dataTransfer.setData("application/x-sample", sample.id);
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: "12px"
                }}
              >
                <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        toggleExpand(sample.id);
                      }}
                      style={{
                        width: "26px",
                        height: "26px",
                        borderRadius: "8px",
                        border: `1px solid ${theme.button.outline}`,
                        background: theme.button.base,
                        color: theme.text,
                        cursor: "pointer"
                      }}
                    >
                      {expanded ? "−" : "+"}
                    </button>
                    <strong style={{ fontSize: "0.95rem", color: theme.text }}>{sample.name}</strong>
                  </div>
                  <div
                    style={{
                      fontSize: "0.78rem",
                      color: theme.textMuted,
                      display: "flex",
                      gap: "10px"
                    }}
                  >
                    <span>{sample.bpm ? `${sample.bpm} BPM` : "Analyzing BPM"}</span>
                    <span>{sample.key ? `Key ${sample.key}` : "Detecting key"}</span>
                    <span>{sample.measures.length ? `${sample.measures.length} measures` : "Preparing measures"}</span>
                  </div>
                  {sample.retuneMap && (
                    <div style={{ fontSize: "0.72rem", color: theme.button.primary, opacity: 0.9 }}>
                      {sample.retuneMap.join(" · ")}
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      void toggleSamplePlayback(sample);
                    }}
                    style={{
                      width: "42px",
                      height: "42px",
                      borderRadius: "50%",
                      border: `1px solid ${theme.button.outline}`,
                      background: playingId === sample.id ? theme.button.primary : theme.button.base,
                      color: playingId === sample.id ? theme.button.primaryText : theme.text,
                      fontWeight: 700,
                      cursor: "pointer",
                      boxShadow: playingId === sample.id ? theme.cardGlow : "none"
                    }}
                  >
                    {playingId === sample.id ? "■" : "▶"}
                  </button>
                  <div style={{ display: "flex", gap: "6px" }}>
                    {paletteChip.map((color) => (
                      <span
                        key={`${sample.id}-${color}`}
                        style={{
                          width: "10px",
                          height: "10px",
                          borderRadius: "999px",
                          background: color,
                          boxShadow: "0 0 12px rgba(0,0,0,0.35)"
                        }}
                      ></span>
                    ))}
                  </div>
                </div>
              </div>

              {expanded && (
                <ul
                  style={{
                    listStyle: "none",
                    padding: "14px 0 0",
                    margin: 0,
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
                    const hasRetune = measure.tunedPitch && measure.detectedPitch && measure.tunedPitch !== measure.detectedPitch;
                    return (
                      <li
                        key={measure.id}
                        style={{
                          padding: "10px 12px",
                          borderRadius: "12px",
                          background: theme.surfaceRaised,
                          border: `1px solid ${theme.border}`,
                          display: "grid",
                          gridTemplateColumns: "auto 1fr auto",
                          gap: "10px",
                          alignItems: "center"
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
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            void playMeasure(sample, measure);
                          }}
                          style={{
                            width: "32px",
                            height: "32px",
                            borderRadius: "50%",
                            border: `1px solid ${theme.button.outline}`,
                            background: playingId === measure.id ? theme.button.primary : theme.button.base,
                            color: playingId === measure.id ? theme.button.primaryText : theme.text,
                            cursor: "pointer",
                            boxShadow: playingId === measure.id ? theme.cardGlow : "none"
                          }}
                        >
                          {playingId === measure.id ? "■" : "▶"}
                        </button>
                        <div
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: "4px",
                            color: theme.text
                          }}
                        >
                          <strong style={{ fontSize: "0.85rem" }}>Measure {index + 1}</strong>
                          <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>
                            {measure.start.toFixed(2)}s → {measure.end.toFixed(2)}s · {duration.toFixed(2)}s ·
                            {measure.beatCount} beats
                          </span>
                          <span style={{ fontSize: "0.72rem", color: theme.button.primary }}>
                            {measure.detectedPitch ?? "?"} → {tunedPitch}
                            {hasRetune ? " (re-keyed)" : ""}
                          </span>
                        </div>
                        <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
                          Energy {Math.round((measure.energy ?? 0) * 100)}%
                        </span>
                      </li>
                    );
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>

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
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
