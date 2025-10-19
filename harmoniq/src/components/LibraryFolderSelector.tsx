import { theme } from "@daw/theme";

interface LibraryFolderSelectorProps {
  folders: string[];
  value: string;
  onChange: (folder: string) => void;
}

export function LibraryFolderSelector({ folders, value, onChange }: LibraryFolderSelectorProps) {
  return (
    <section
      style={{
        width: "100%",
        borderRadius: "18px",
        border: "1px solid rgba(120, 203, 220, 0.28)",
        background: "rgba(6, 24, 34, 0.82)",
        padding: "18px 22px",
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: "16px",
        justifyContent: "space-between",
      }}
    >
      <div
        style={{
          fontSize: "0.75rem",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: theme.textMuted,
          display: "flex",
          flexDirection: "column",
          gap: "6px",
        }}
      >
        <span>Library Folders</span>
        <span style={{ fontSize: "0.68rem", color: theme.button.primaryText }}>
          Browsing: {value}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          gap: "12px",
          flexWrap: "wrap",
          justifyContent: "flex-end",
        }}
      >
        {folders.map((folder) => {
          const isActive = folder === value;
          return (
            <button
              key={folder}
              type="button"
              onClick={() => onChange(folder)}
              style={{
                padding: "10px 16px",
                borderRadius: "999px",
                border: `1px solid ${isActive ? theme.button.primary : "rgba(120, 203, 220, 0.24)"}`,
                background: isActive ? "rgba(21, 74, 98, 0.7)" : "rgba(6, 24, 34, 0.7)",
                color: isActive ? theme.button.primaryText : theme.textMuted,
                fontSize: "0.68rem",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              {folder}
            </button>
          );
        })}
      </div>
    </section>
  );
}
