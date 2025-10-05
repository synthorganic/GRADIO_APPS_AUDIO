import { useEffect, useMemo, useRef, useState } from "react";
import { theme } from "../theme";

type MenuKey = "File" | "Edit";

interface MenuItem {
  label: string;
  hint?: string;
  action?: () => void;
}

interface TopMenuProps {
  onOpenSettings: () => void;
}

export function TopMenu({ onOpenSettings }: TopMenuProps) {
  const [openMenu, setOpenMenu] = useState<MenuKey | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const menuConfig = useMemo<Record<MenuKey, MenuItem[]>>(
    () => ({
      File: [
        { label: "New Project" },
        { label: "Open Project…" },
        { label: "Save Project" },
        { label: "Export Mixdown…" }
      ],
      Edit: [
        { label: "Undo" },
        { label: "Redo" },
        { label: "Settings", hint: "Sample Rate, Playback Device, …", action: onOpenSettings }
      ]
    }),
    [onOpenSettings]
  );

  useEffect(() => {
    const handleClickAway = (event: MouseEvent) => {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(event.target as Node)) {
        setOpenMenu(null);
      }
    };

    document.addEventListener("mousedown", handleClickAway);
    return () => document.removeEventListener("mousedown", handleClickAway);
  }, []);

  const toggleMenu = (key: MenuKey) => {
    setOpenMenu((current) => (current === key ? null : key));
  };

  return (
    <nav
      ref={containerRef}
      aria-label="Application Menu"
      style={{ display: "flex", alignItems: "center", gap: "6px" }}
    >
      {(Object.keys(menuConfig) as MenuKey[]).map((key) => (
        <div key={key} style={{ position: "relative" }}>
          <button
            type="button"
            onClick={() => toggleMenu(key)}
            style={{
              border: `1px solid ${theme.button.outline}`,
              background: theme.surface,
              color: theme.text,
              borderRadius: "6px",
              padding: "6px 12px",
              fontSize: "0.75rem",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
              cursor: "pointer",
              minWidth: "72px"
            }}
          >
            {key}
          </button>
          {openMenu === key && (
            <div
              role="menu"
              style={{
                position: "absolute",
                top: "calc(100% + 4px)",
                left: 0,
                minWidth: "180px",
                padding: "6px 0",
                background: theme.surfaceRaised,
                border: `1px solid ${theme.border}`,
                borderRadius: "10px",
                boxShadow: theme.shadow,
                display: "flex",
                flexDirection: "column",
                gap: "2px",
                zIndex: 10
              }}
            >
              {menuConfig[key].map((item) => (
                <button
                  key={item.label}
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setOpenMenu(null);
                    item.action?.();
                  }}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    textAlign: "left",
                    gap: item.hint ? "2px" : 0,
                    padding: "8px 12px",
                    background: "transparent",
                    border: "none",
                    color: theme.text,
                    fontSize: "0.8rem",
                    letterSpacing: "0.02em",
                    cursor: "pointer"
                  }}
                >
                  <span>{item.label}</span>
                  {item.hint ? (
                    <span style={{ fontSize: "0.68rem", opacity: 0.65 }}>{item.hint}</span>
                  ) : null}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </nav>
  );
}
