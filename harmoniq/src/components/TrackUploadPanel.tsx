import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type DragEvent,
} from "react";
import { theme } from "@daw/theme";
import type { StemType } from "../types";
import { analyzeAudioFile } from "../lib/audioAnalysis";
import type { AudioAnalysisStem } from "../lib/audioAnalysis";

export interface AnalyzedStem {
  id: string;
  label: string;
  type: StemType;
}

export interface AnalyzedTrackSummary {
  id: string;
  name: string;
  bpm: number;
  scale: string;
  stems: AnalyzedStem[];
  origin: string;
  file: File;
  objectUrl: string;
  durationSeconds: number | null;
  analysisError: string | null;
}

interface TrackUploadPanelProps {
  onTracksAnalyzed: (tracks: AnalyzedTrackSummary[]) => void;
}

type PendingItem = {
  file: File;
  id: string;
};

const FALLBACK_STEMS: ReadonlyArray<{ type: StemType; label: string }> = [
  { type: "drums", label: "Drums" },
  { type: "synths", label: "Synths" },
  { type: "vocals", label: "Vocals" },
];

function hashString(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) & 0xffffffff;
  }
  return Math.abs(hash);
}

const MUSICAL_KEYS = [
  "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "11A", "12A",
  "1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B", "9B", "10B", "11B", "12B",
];

async function enumerateFromDataTransfer(items: DataTransferItemList): Promise<File[]> {
  const pending: Promise<File[]>[] = [];

  const visitEntry = (entry: FileSystemEntry | null): Promise<File[]> => {
    if (!entry) {
      return Promise.resolve([]);
    }
    if (entry.isFile) {
      return new Promise((resolve) => {
        (entry as FileSystemFileEntry).file((file) => resolve([file]));
      });
    }
    if (entry.isDirectory) {
      const directory = entry as FileSystemDirectoryEntry;
      return new Promise((resolve) => {
        const reader = directory.createReader();
        const accum: File[] = [];
        const iterate = () => {
          reader.readEntries((entries) => {
            if (!entries.length) {
              resolve(accum);
              return;
            }
            Promise.all(entries.map((child) => visitEntry(child))).then((chunks) => {
              chunks.forEach((chunk) => accum.push(...chunk));
              iterate();
            });
          });
        };
        iterate();
      });
    }
    return Promise.resolve([]);
  };

  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    const withEntry = item as DataTransferItem & {
      webkitGetAsEntry?: () => FileSystemEntry | null;
    };
    const entry = withEntry.webkitGetAsEntry?.();
    if (entry) {
      pending.push(visitEntry(entry));
    } else if (item.kind === "file") {
      const file = item.getAsFile();
      if (file) {
        pending.push(Promise.resolve([file]));
      }
    }
  }

  if (!pending.length) {
    return Array.from(items)
      .map((item) => item.getAsFile())
      .filter((file): file is File => Boolean(file));
  }

  const collected = await Promise.all(pending);
  return collected.flat();
}

function createPendingId(file: File) {
  if (typeof crypto?.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function TrackUploadPanel({ onTracksAnalyzed }: TrackUploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const objectUrlsRef = useRef<Map<string, string>>(new Map());

  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      objectUrlsRef.current.clear();
    };
  }, []);

  const borderStyle = isDragging
    ? "1px solid rgba(103, 255, 230, 0.85)"
    : "1px dashed rgba(120, 203, 220, 0.4)";

  const helperText = useMemo(() => {
    if (isAnalyzing) {
      return "Analyzing uploads for beat and scale...";
    }
    if (isDragging) {
      return "Drop folders or files to analyze";
    }
    return "Drag folders or files here, or browse to upload";
  }, [isAnalyzing, isDragging]);

  const analyzeFiles = useCallback(
    async (files: File[]) => {
      if (!files.length) return;
      setIsAnalyzing(true);
      const tracks: AnalyzedTrackSummary[] = [];
      const seenOrigins = new Set<string>();
      const queue: PendingItem[] = files.map((file) => ({ file, id: createPendingId(file) }));
      for (const item of queue) {
        const origin = `${item.file.webkitRelativePath || item.file.name}`;
        if (seenOrigins.has(origin)) {
          // Skip duplicates from nested directory traversal
          continue;
        }
        seenOrigins.add(origin);
        const objectUrl = URL.createObjectURL(item.file);
        objectUrlsRef.current.set(item.id, objectUrl);
        const baseKey = `${item.file.name}-${item.file.size}-${item.file.lastModified}`;
        const hash = hashString(baseKey);
        let analysisError: string | null = null;
        let bpm = 120;
        let scale = MUSICAL_KEYS[hash % MUSICAL_KEYS.length];
        let stems: AudioAnalysisStem[] = [];
        let durationSeconds: number | null = null;
        try {
          const analysis = await analyzeAudioFile(item.file);
          bpm = analysis.bpm;
          scale = analysis.camelotKey;
          stems = analysis.stems;
          durationSeconds = analysis.durationSeconds;
        } catch (error) {
          analysisError = error instanceof Error ? error.message : String(error);
          stems = FALLBACK_STEMS.map((stem) => ({ type: stem.type, label: stem.label }));
        }
        const resolvedStems: AnalyzedStem[] = stems.map((stem, index) => ({
          id: `${stem.type}-${hash % 10_000}-${index}`,
          label: stem.label,
          type: stem.type,
        }));
        tracks.push({
          id: item.id,
          name: item.file.name.replace(/\.[^/.]+$/, ""),
          bpm,
          scale,
          stems: resolvedStems,
          origin,
          file: item.file,
          objectUrl,
          durationSeconds,
          analysisError,
        });
      }
      onTracksAnalyzed(tracks);
      setIsAnalyzing(false);
    },
    [onTracksAnalyzed],
  );

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const usable = Array.isArray(files) ? files : Array.from(files);
      await analyzeFiles(usable.filter((file) => file.type.startsWith("audio") || file.type === ""));
    },
    [analyzeFiles],
  );

  const handleDrop = useCallback(
    async (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const items = event.dataTransfer.items;
      if (items && items.length) {
        try {
          const files = await enumerateFromDataTransfer(items);
          await analyzeFiles(files);
          return;
        } catch (error) {
          console.warn("Failed to enumerate dropped items", error);
        }
      }
      if (event.dataTransfer.files?.length) {
        await handleFiles(event.dataTransfer.files);
      }
    },
    [analyzeFiles, handleFiles],
  );

  const onBrowse = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <section
      onDragOver={(event) => {
        event.preventDefault();
        if (!isDragging) {
          setIsDragging(true);
        }
      }}
      onDragLeave={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node)) {
          return;
        }
        setIsDragging(false);
      }}
      onDrop={handleDrop}
      style={{
        borderRadius: "16px",
        border: borderStyle,
        padding: "24px",
        display: "grid",
        gap: "12px",
        background: "rgba(6, 24, 34, 0.78)",
        transition: "border 0.2s ease, background 0.2s ease",
      }}
    >
      <div style={{ display: "grid", gap: "6px" }}>
        <strong style={{ fontSize: "0.85rem" }}>Upload Tracks</strong>
        <span style={{ fontSize: "0.72rem", color: theme.textMuted }}>{helperText}</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <button
          type="button"
          onClick={onBrowse}
          style={{
            ...buttonStyles,
            background: theme.button.primary,
            color: theme.button.primaryText,
          }}
        >
          Browse Files
        </button>
        <span style={{ fontSize: "0.7rem", color: theme.textMuted }}>
          Supported: WAV, MP3, FLAC, AIFF
        </span>
      </div>
      {isAnalyzing ? (
        <div style={{ fontSize: "0.72rem", color: theme.button.primaryText }}>
          Performing stem separation and beat detection...
        </div>
      ) : null}
      <input
        ref={(node) => {
          fileInputRef.current = node;
          if (node) {
            node.setAttribute("webkitdirectory", "");
            node.setAttribute("directory", "");
          }
        }}
        type="file"
        accept="audio/*"
        multiple
        style={{ display: "none" }}
        onChange={(event) => {
          const { files } = event.currentTarget;
          if (files?.length) {
            handleFiles(files).finally(() => {
              event.currentTarget.value = "";
            });
          }
        }}
      />
    </section>
  );
}

const buttonStyles: CSSProperties = {
  padding: "10px 16px",
  borderRadius: "12px",
  border: "none",
  cursor: "pointer",
  fontSize: "0.75rem",
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  boxShadow: theme.shadow,
};
