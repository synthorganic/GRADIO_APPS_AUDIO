# Shared DAW Mixer Components

This directory contains UI primitives that are shared between the main mixer views
and any Harmoniq-specific panels. The goal is to keep timeline rendering and FX
rack orchestration consistent across experiences.

## `WaveformPreview`

**Location:** `./WaveformPreview.tsx`

- **Purpose:** Render a normalized audio waveform into an SVG footprint that can
  scale responsively inside clip tiles.
- **Props:**
  - `waveform: Float32Array` — The min/max envelope for a clip, normalized to the
    range `[0, 1]`. The preview component does not perform normalization itself.
  - `fillColor?: string` — Optional custom fill colour for the closed path.
  - `strokeColor?: string` — Optional outline colour; defaults are tuned for the
    dark mixer theme.
- **State expectations:** The caller owns waveform memoisation. `WaveformPreview`
  memoises its SVG path internally and only re-renders when the provided
  `Float32Array` reference changes.

## `FxRack`

**Location:** `./FxRack.tsx`

- **Purpose:** Present a consistent interface for loading and listing
  clip/master FX (e.g., WASM VSTs) across the mixer and downstream Harmoniq
  tools.
- **Props:**
  - `title: string` / `description: string` — Heading copy to contextualise the
    rack.
  - `targetName: string` — Human-readable routing target (e.g., sample name or
    `project.name`). An optional `targetLabel` overrides the default "Target"
    caption.
  - `plugins: FxPluginDescriptor[]` — Current rack entries. The descriptor
    includes `id`, `name`, `type`, `loadedAt`, and an optional `fileName`.
  - `onLoadPlugin: (file: File | null) => void` — Callback invoked when the user
    selects a file. The consumer is responsible for instantiating and storing the
    plugin metadata.
  - `emptyStateMessage: string` — Copy displayed when no plugins are loaded.
  - Optional overrides: `actionLabel` for the load button label and `accept` for
    the file input filter.
- **Helpers:**
  - `deriveFxPluginType(file: File): FxPluginType` — Utility to infer a plugin
    type (`"synth"`, `"effect"`, or `"utility"`) from a filename extension.
- **State expectations:** The component is intentionally stateless. Consumers
  own the plugin list and determine how file selection mutates their stores.
  Because the hidden file input resets its value after each selection, callers
  can safely push identical filenames without manual cleanup.

When introducing new shared views, prefer colocating UI logic here so that the
Harmoniq panels can reuse the same surface-level behaviour without duplicating
mixer internals.
