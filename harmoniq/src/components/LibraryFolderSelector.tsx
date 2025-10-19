import { memo, useMemo } from "react";
import { theme } from "@daw/theme";
import { cardSurfaceStyle, toolbarButtonStyle } from "@daw/components/layout/styles";

interface LibraryFolderSelectorProps {
  folders: string[];
  value: string;
  onChange: (folder: string) => void;
}

const FALLBACK_FOLDER = "All Sessions";

export const LibraryFolderSelector = memo(function LibraryFolderSelector({
  folders,
  value,
  onChange,
}: LibraryFolderSelectorProps) {
  const options = useMemo(() => {
    const unique = new Map<string, string>();
    unique.set(FALLBACK_FOLDER, FALLBACK_FOLDER);
    folders.forEach((folder) => {
      if (folder.trim()) {
        unique.set(folder, folder);
      }
    });
    return Array.from(unique.values());
  }, [folders]);

  const handleSelect = (folder: string) => {
    if (folder !== value) {
      onChange(folder);
    }
  };

  return (
    <section
      style={{
        ...cardSurfaceStyle,
        padding: "16px 20px",
        display: "grid",
        gap: "12px",
      }}
    >
      <header style={{ display: "grid", gap: "4px" }}>
        <span
          style={{
            fontSize: "0.66rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: "rgba(120, 203, 220, 0.7)",
          }}
        >
          Library Folder
        </span>
        <h2
          style={{
            margin: 0,
            fontSize: "0.92rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
          }}
        >
          {value || FALLBACK_FOLDER}
        </h2>
        <p style={{ margin: 0, fontSize: "0.68rem", color: theme.textMuted }}>
          Filter saved sessions by folder to quickly recall curated loop sets.
        </p>
      </header>
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {options.map((folder) => {
          const isActive = folder === value || (!value && folder === FALLBACK_FOLDER);
          return (
            <button
              key={folder}
              type="button"
              onClick={() => handleSelect(folder)}
              style={{
                ...toolbarButtonStyle,
                padding: "8px 14px",
                borderRadius: "12px",
                borderColor: isActive ? theme.button.primary : "rgba(120, 203, 220, 0.3)",
                background: isActive ? theme.button.primary : "rgba(9, 30, 42, 0.75)",
                color: isActive ? theme.button.primaryText : theme.text,
                boxShadow: isActive ? "0 12px 32px rgba(3, 26, 34, 0.45)" : "none",
              }}
            >
              {folder}
            </button>
          );
        })}
      </div>
    </section>
  );
});
