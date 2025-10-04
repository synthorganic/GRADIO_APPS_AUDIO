
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { nanoid } from "nanoid";
import { useProjectStore } from "../state/ProjectStore";
import type {
  AutomationChannel,
  AutomationPoint,
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

type SnapResolution = "measure" | "half-measure" | "beat" | "half-beat";

const MEASURE_WIDTH = 84;
const CLIP_HEIGHT = 48;
const CHANNEL_HEIGHT = 72;
const MAX_CLONE_COUNT = 32;

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

export function Timeline({ project, selectedSampleId, onSelectSample }: TimelineProps) {
  const { dispatch, currentProjectId, lastControlTarget } = useProjectStore();
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
  const [openMidiChannelId, setOpenMidiChannelId] = useState<string | null>(null);

  const { processSample } = useDemucsProcessing((updated) => {
    dispatch({
      type: "update-sample",
      projectId: currentProjectId,
      sampleId: updated.id,
      sample: updated,
    });
  });

  const secondsPerMeasure = useMemo(
    () => measureDurationSeconds(project),
    [project.masterBpm],
  );

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

  const timelineWidth = useMemo(
    () => Math.max(totalDuration / secondsPerMeasure, 4) * MEASURE_WIDTH,
    [secondsPerMeasure, totalDuration],
  );

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
    const measures = Math.ceil(totalDuration / secondsPerMeasure);
    for (let measure = 0; measure < measures * divisor; measure += 1) {
      ticks.push({ position: measure / divisor, accent: measure % divisor });
    }
    return ticks;
  }, [secondsPerMeasure, snapResolution, totalDuration]);

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
    [secondsPerMeasure, snapResolution],
  );

  const pointerTimelinePosition = useCallback(
    (clientX: number) => {
      const container = scrollRef.current;
      if (!container) return 0;
      const bounds = container.getBoundingClientRect();
      const relativeX = clientX - bounds.left + container.scrollLeft;
      const measuresFromStart = Math.max(0, relativeX / MEASURE_WIDTH);
      return measuresFromStart * secondsPerMeasure;
    },
    [secondsPerMeasure],
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
        if (sampleData) {
          const sample = project.samples.find((item) => item.id === sampleData);
          if (!sample) return;
          const note: MidiNote = {
            id: nanoid(),
            start: dropPosition,
            length: sample.length,
            pitch: 60,
            velocity: 0.85,
            sampleId: sample.id,
          };
          dispatch({
            type: "update-channel",
            projectId: currentProjectId,
            channelId: channel.id,
            patch: {
              notes: [...channel.notes, note],
            },
          });
          void audioEngine.play(sample, sample.measures, { emitTimelineEvents: false });
        }
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
    [currentProjectId, dispatch, onSelectSample, processSample, project.samples, projectDropPosition],
  );

  const playClip = useCallback((sample: SampleClip) => {
    void audioEngine.play(sample, sample.measures, { emitTimelineEvents: false });
  }, []);

  const handleClipPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, sample: SampleClip) => {
      if (!sample.channelId) return;
      const container = scrollRef.current;
      if (!container) return;
      event.currentTarget.setPointerCapture(event.pointerId);
      dragState.current = {
        sampleId: sample.id,
        pointerId: event.pointerId,
        originX: event.clientX + container.scrollLeft,
        originY: event.clientY,
        originPosition: sample.position,
        channelId: sample.channelId,
      };
    },
    [],
  );

  const handleClipPointerMove = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>, sample: SampleClip) => {
      const state = dragState.current;
      if (!state || state.pointerId !== event.pointerId) return;
      const container = scrollRef.current;
      if (!container) return;
      const cursor = event.clientX + container.scrollLeft;
      const delta = cursor - state.originX;
      const measuresMoved = delta / MEASURE_WIDTH;
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
    },
    [currentProjectId, dispatch, secondsPerMeasure, snapResolution],
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
    },
    [currentProjectId, dispatch, project.channels],
  );

  const handleRemoveClip = useCallback(
    (sample: SampleClip) => {
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
    [currentProjectId, dispatch, effectsSampleId, onSelectSample, selectedSampleId],
  );

  const handleClipContextMenu = useCallback(
    (event: ReactMouseEvent, sample: SampleClip) => {
      event.preventDefault();
      event.stopPropagation();
      handleRemoveClip(sample);
    },
    [handleRemoveClip],
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
      notes: [],
    };
    dispatch({ type: "add-channel", projectId: currentProjectId, channel });
  };

  const renderAutomationLane = (channel: AutomationChannel) => {
    const sorted = ensurePointRange([...channel.points]).sort((a, b) => a.time - b.time);
    const values = sorted.map((point) => point.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = maxValue - minValue || 1;
    const timelineLength = Math.max(totalDuration, secondsPerMeasure);
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
    const timelineLength = Math.max(totalDuration, secondsPerMeasure);
    return (
      <div
        style={{
          position: "relative",
          height: `${CHANNEL_HEIGHT}px`,
          background: "#102f19",
          borderRadius: "8px",
          border: `1px solid ${theme.button.outline}55`,
          padding: "8px",
          cursor: "pointer",
        }}
        onClick={() => setOpenMidiChannelId(channel.id)}
      >
        <div style={{ position: "relative", height: "100%", width: `${timelineWidth}px` }}>
          {channel.notes.map((note) => {
            const left = (note.start / timelineLength) * timelineWidth;
            const width = Math.max((note.length / timelineLength) * timelineWidth, 8);
            return (
              <div
                key={note.id}
                style={{
                  position: "absolute",
                  top: `${16 + (note.pitch % 12) * 1.6}px`,
                  left: `${left}px`,
                  width: `${width}px`,
                  height: "12px",
                  borderRadius: "5px",
                  background: "#3cf47d",
                  boxShadow: "0 0 6px rgba(60, 244, 125, 0.4)",
                }}
              />
            );
          })}
        </div>
      </div>
    );
  };

  const midiChannel = project.channels.find(
    (channel): channel is MidiChannel => channel.id === openMidiChannelId && channel.type === "midi",
  );

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
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
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
                  width: `${MEASURE_WIDTH / SNAP_DIVISORS[snapResolution]}px`,
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
                        const clipWidth = Math.max(80, widthMeasures * MEASURE_WIDTH);
                        const left = (sample.position / secondsPerMeasure) * MEASURE_WIDTH;
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
                              background: clipGradient(sample.stems),
                              cursor: "grab",
                              display: "flex",
                              flexDirection: "column",
                              justifyContent: "space-between",
                              padding: "6px 8px",
                            }}
                            onPointerDown={(event) => handleClipPointerDown(event, sample)}
                            onPointerMove={(event) => handleClipPointerMove(event, sample)}
                            onPointerUp={(event) => handleClipPointerUp(event, sample)}
                            onClick={() => onSelectSample(sample.id)}
                            onContextMenu={(event) => handleClipContextMenu(event, sample)}
                          >
                            <div style={{ display: "flex", justifyContent: "space-between", gap: "6px" }}>
                              <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
                                <strong style={{ fontSize: "0.72rem" }}>{clipLabel}</strong>
                                <div
                                  style={{
                                    fontSize: "0.62rem",
                                    color: theme.textMuted,
                                    display: "flex",
                                    gap: "6px",
                                  }}
                                >
                                  <span>{sample.bpm ? `${sample.bpm} BPM` : "Analyzing"}</span>
                                  <span>{sample.key ? `Key ${sample.key}` : "Key TBD"}</span>
                                  <span>{sample.length.toFixed(2)}s</span>
                                </div>
                                <div style={{ display: "flex", gap: "3px" }}>
                                  {sample.stems.map((stem) => (
                                    <span
                                      key={`${sample.id}-${stem.id}-chip`}
                                      style={{
                                        width: "7px",
                                        height: "7px",
                                        borderRadius: "50%",
                                        background: stem.color,
                                        boxShadow: "0 0 6px rgba(0,0,0,0.35)",
                                      }}
                                    />
                                  ))}
                                </div>
                              </div>
                              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    playClip(sample);
                                  }}
                                  style={{
                                    padding: "3px 5px",
                                    borderRadius: "6px",
                                    border: `1px solid ${theme.button.outline}`,
                                    background: theme.surface,
                                    color: theme.text,
                                    fontSize: "0.62rem",
                                    cursor: "pointer",
                                  }}
                                >
                                  ▶
                                </button>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    duplicateClip(sample);
                                  }}
                                  style={{
                                    padding: "3px 5px",
                                    borderRadius: "6px",
                                    border: `1px solid ${theme.button.outline}`,
                                    background: theme.surface,
                                    color: theme.text,
                                    fontSize: "0.62rem",
                                    cursor: "pointer",
                                  }}
                                >
                                  ⧉
                                </button>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setEffectsSampleId(sample.id);
                                  }}
                                  style={{
                                    padding: "3px 5px",
                                    borderRadius: "6px",
                                    border: `1px solid ${theme.button.outline}`,
                                    background: theme.surface,
                                    color: theme.text,
                                    fontSize: "0.62rem",
                                    cursor: "pointer",
                                  }}
                                >
                                  FX
                                </button>
                              </div>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem" }}>
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleRemoveClip(sample);
                                }}
                                style={{
                                  border: "none",
                                  background: "transparent",
                                  color: theme.textMuted,
                                  cursor: "pointer",
                                }}
                              >
                                Remove
                              </button>
                              <span style={{ color: theme.textMuted }}>{channel.name}</span>
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
              left: `${200 + (playheadPosition / secondsPerMeasure) * MEASURE_WIDTH}px`,
              width: "2px",
              background: theme.button.primary,
              boxShadow: `0 0 12px ${theme.button.primary}`,
            }}
          />
        </div>
      </div>

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

      {midiChannel && (
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
          onClick={() => setOpenMidiChannelId(null)}
        >
          <div
            style={{
              width: "min(680px, 90vw)",
              background: theme.surfaceOverlay,
              borderRadius: "16px",
              border: `1px solid ${theme.button.outline}`,
              boxShadow: theme.cardGlow,
              padding: "24px",
              color: theme.text,
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3 style={{ margin: 0, fontSize: "1rem" }}>{midiChannel.name} piano roll</h3>
              <button
                type="button"
                onClick={() => setOpenMidiChannelId(null)}
                style={{
                  border: "none",
                  background: "transparent",
                  color: theme.text,
                  cursor: "pointer",
                  fontSize: "1.1rem",
                }}
              >
                ✕
              </button>
            </div>
            <p style={{ fontSize: "0.8rem", color: theme.textMuted }}>
              Drag samples into this lane to sketch note events. Notes preview the assigned sample and
              align to the global snap grid.
            </p>
            <div style={{ display: "grid", gap: "8px", marginTop: "16px" }}>
              {midiChannel.notes.length === 0 ? (
                <span style={{ color: theme.textMuted, fontSize: "0.75rem" }}>
                  No notes yet — drop a sample on the MIDI lane to add one.
                </span>
              ) : (
                midiChannel.notes
                  .slice()
                  .sort((a, b) => a.start - b.start)
                  .map((note) => {
                    const sample = note.sampleId
                      ? project.samples.find((item) => item.id === note.sampleId)
                      : null;
                    return (
                      <div
                        key={note.id}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          fontSize: "0.75rem",
                          padding: "6px 10px",
                          borderRadius: "8px",
                          background: theme.surface,
                          border: `1px solid ${theme.button.outline}33`,
                        }}
                      >
                        <span>
                          {sample ? sample.name : "Sampler"} · {note.start.toFixed(2)}s → {(
                            note.start + note.length
                          ).toFixed(2)}s
                        </span>
                        <span>Pitch {note.pitch}</span>
                      </div>
                    );
                  })
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
