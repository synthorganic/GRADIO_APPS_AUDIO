export const theme = {
  background: "#03121a",
  surface: "#071c24",
  surfaceRaised: "#0b2733",
  surfaceOverlay: "#123543",
  border: "rgba(120, 203, 220, 0.36)",
  text: "#eef8fb",
  textMuted: "rgba(192, 222, 230, 0.7)",
  accentBeam: ["#f18bb4", "#f7b89c", "#f4e3a5", "#9fe0d6", "#6cc6d8", "#7e9ae6"],
  button: {
    base: "#0d2a33",
    hover: "#123540",
    active: "#17404d",
    primary: "#4cc7c2",
    primaryText: "#031014",
    disabled: "rgba(94, 140, 150, 0.45)",
    outline: "rgba(120, 203, 220, 0.45)"
  },
  shadow: "0 24px 60px -40px rgba(2, 14, 20, 0.88)",
  divider: "rgba(60, 108, 120, 0.45)",
  cardGlow: "0 12px 32px -22px rgba(5, 22, 30, 0.82)"
} as const;

export type Theme = typeof theme;
