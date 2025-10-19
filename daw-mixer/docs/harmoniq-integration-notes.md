# Harmoniq Integration Notes

This document summarizes the reusable pieces inside the DAW mixer prototype that can be
leveraged while building Harmoniq's darker performance skin and live looping workflows.
It also calls out the largest functional gaps for an on-stage experience.

## Application Shell & Layout

- **AppShell layout (`src/App.tsx`)** – Composes the grid-based shell that divides the
  interface into the left rail (project navigation + live looper) and the main timeline.
  The shell also manages floating panels (Mixer, VST Rack, Sample Details) that can be
  styled independently from the base grid. Toolbar buttons reuse the shared
  `toolbarButtonStyle` constant for consistent pill-shaped controls.
- **Header composition** – The top bar combines `TopMenu`, the SONiQ logo asset, master
  transport via `MasterControls`, and quick links to floating panels. It uses theme-driven
  inline styles for borders, background, and spacing.
- **Left rail** – `ProjectNavigator` lists project materials while `LiveLooper` renders the
  live capture controls. Both use the same theme tokens for surfaces/borders to stay cohesive.
- **Main content** – `Timeline` consumes project state and emits sample selections back to
  the shell via callbacks. This main area is flex-driven and expects content stacked
  vertically with gaps.
- **Modal-style floating panels** – Mixer/VST/Sample panels share a full-screen overlay with
  a raised card container. Closing the overlay resets the `activePanel` state in `AppShell`.
- **Settings dialog** – `SettingsDialog` is mounted globally with boolean state in the shell
  and receives preference update callbacks.

These pieces are reusable: any new layout should preserve the `ProjectProvider` context and
can remap styles by swapping out theme tokens or overriding the inline style objects.

## State Management Overview

- **Single store (`src/state/ProjectStore.tsx`)** – The app relies on a `useReducer`
  powered context. `ProjectProvider` exposes the entire `ProjectState` plus a `dispatch`
  function and a `getPalette` helper for stem colors.
- **Initial project bootstrap** – `initialProject` generates one audio channel via
  `createAudioChannel` and seeds mastering defaults. Preferences default to the first stem
  engine and heuristics defined in `stem_engines`.
- **Actions** – Reducer branches cover sample CRUD, channel CRUD, mastering updates,
  scale updates, registering automation control changes, and preference updates. Channel
  additions normalize volume/pan defaults across audio, automation, and MIDI tracks.
- **Selection helpers** – Components such as `AppShell`, `MasterControls`, and
  `LiveLooper` pull project data via `useProjectStore()` and dispatch actions for updates.
- **Theme coupling** – The store imports `theme` to seed channel colors and mastering UI,
  so any redesign should either keep this dependency or replace the defaults when loading a
  project.

For Harmoniq, this centralized reducer is the main extension point for new live-looping,
automation, or stem-processing state.

## Audio Engine Capabilities (`src/lib/audioEngine.ts`)

- **Core playback** – `AudioEngine` wraps a singleton `AudioContext` with helpers for
  decoding buffers, caching, and connecting audio nodes through a master gain stage.
  Methods include `play`, `playSegment`, `playStem`, `playTimeline`, `triggerOneShot`,
  `stop`, and waveform peak extraction (`getWaveformPeaks`).
- **Loop support** – Playback options accept `loop` flags and timeline offsets. `play` and
  `playStem` configure loop start/end on the source node when `loop` is true.
- **Channel mixing** – `syncChannelMix` ingests the timeline's channel definitions and
  caches per-channel volume/pan. `playTimeline` uses those mixes when scheduling clips.
- **Stem processing chain** – `buildStemChain` constructs EQ/filter node chains per stem
  type (`vocals`, `leads`, `percussion`, `kicks`, `bass`) and routes them through the gain
  node when a stem is auditioned.
- **Timeline events** – The engine dispatches browser events (`audio-play`, `audio-stop`,
  `audio-tick`) and optional measure callbacks so UI components can react to transport state.
- **Buffer caching** – `decodeSample` memoizes decoded buffers per sample/stem identity to
  avoid redundant decoding during looped playback or repeated auditioning.

These utilities are ready to power Harmoniq's transport; extending them with overdub
recording, automation playback, or synchronized stem streaming will require new APIs.

## Live Looping Inventory & Gaps

- **`LiveLooper` component** – Presents a simple arm/record flow and loop length control.
  When armed it schedules a timeout (600 ms) that fabricates a `File` object and inserts a
  new `SampleClip` into the project store. All measurements derive from `project.masterBpm`.
- **Project store expectations** – Samples inserted by the looper mark `isLooping: true`,
  set `isInTimeline: true`, and attach default track effects via `createDefaultTrackEffects`.
  No additional state slice tracks live-take history or overdub layers.

**Missing features for stage use**

1. Real audio capture – The looper currently fakes recorded audio instead of pulling data
   from `MediaStream` inputs or the Web Audio graph. No waveform data is generated.
2. Downbeat synchronization – `setTimeout` does not align to `audioEngine` transport
   events or tempo grid; there's no quantization to ensure the take lands on the next bar.
3. Stem routing – New samples land on the first audio channel without stem classification,
   so there is no automatic assignment to vocal/percussion/etc. chains for live stems.
4. Automation hooks – Project store actions do not persist automation recordings or
   parameter locks during looping. `register-control-change` only tracks the last touched
   control.
5. Live overdub layers – There is no notion of loop banks/scenes, undo/redo of takes, or
   muting individual layers once captured.
6. Engine synchronization – `LiveLooper` never interacts with `audioEngine`; therefore
   recordings do not start/stop with actual playback and transport state is ignored.

These gaps outline the roadmap for integrating real-time stems and automation capture in
Harmoniq.

## Styling Primitives & Theming

- **Global CSS (`src/styles.css`)** – Defines the base font stack (Montserrat), background
  color, and basic resets. A single bespoke class `.timeline-zoom-slider` styles the range
  input with custom thumb/track colors (`#4cc7c2`) and focus states.
- **Theme tokens (`src/theme.ts`)** – Exports a typed theme object covering:
  - Layer backgrounds (`background`, `surface`, `surfaceRaised`, `surfaceOverlay`)
  - Text colors (`text`, `textMuted`)
  - Border/divider colors and shadow presets
  - Button states (`base`, `hover`, `active`, `primary`, `primaryText`, `disabled`, `outline`)
  - Accent palette (`accentBeam`) used for channel coloring
  - `cardGlow` used for emphasis shadows

To produce Harmoniq's darker performance skin, you can swap these theme values (and the
`.timeline-zoom-slider` custom property) without rewriting component logic. Inline styles
in components reference the theme tokens directly, so providing an alternate `theme`
export or runtime override will propagate the new palette across the UI.
