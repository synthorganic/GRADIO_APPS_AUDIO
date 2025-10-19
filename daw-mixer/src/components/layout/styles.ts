import type { CSSProperties } from "react";
import { theme } from "../../theme";

export const toolbarButtonStyle: CSSProperties = {
  border: `1px solid ${theme.button.outline}`,
  background: theme.button.base,
  color: theme.text,
  borderRadius: "999px",
  padding: "4px 12px",
  fontSize: "0.7rem",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  cursor: "pointer"
};

export const toolbarButtonDisabledStyle: CSSProperties = {
  opacity: 0.4,
  cursor: "not-allowed"
};

export const cardSurfaceStyle: CSSProperties = {
  background: theme.surfaceOverlay,
  borderRadius: "12px",
  border: `1px solid ${theme.border}`,
  boxShadow: theme.shadow,
  color: theme.text
};

export const gridRows = (...values: Array<string | number>): string => values.join(" ");
