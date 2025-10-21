# Harmoniq Workstation Prototype

Harmoniq is a browser-based loop performance surface that shares theme, layout, and state patterns with the `daw-mixer` reference experience in this repository. The app persists loop libraries locally so Camelot-key track selection stays hydrated and keeps uploaded tracks ready for deck assignment. The lower library strip now focuses purely on the upload panel and analyzed track list—automation envelopes, active deck dashboards, and saved loop folders have been retired to keep the workflow lean.

Real-time automation inside Harmoniq is intentionally minimal: holding the **Play** button performs a smooth fade-in and holding **Stop** performs the matching fade-out. Any deeper envelope editing or playback automation remains in the `daw-mixer` project.

## Audio engine integration highlights

- **Deck EQ cuts** now drive three overlapping filters inside the `HarmoniqAudioBridge`, so the Hi/Mid/Low Cut buttons reshape the signal instead of toggling UI state.
- **Stem triggers** map to bridge stem profiles that rebalance the deck’s multiband split, muting and boosting regions to spotlight drums, synths, or vocals without reloading audio files.
- **Loop capture** routes the recorder strip through ScriptProcessor nodes to print bar- or half-bar loops, reloading the rendered WAV back onto the focused deck and persisting it to the loop library.
- **Track analysis** performs DSP-based tempo, Camelot key, and stem energy detection during upload, ensuring the library metadata mirrors the actual audio instead of placeholder hashes.

## Getting started

1. **Install dependencies**
   ```bash
   cd harmoniq
   npm install
   ```
2. **Run the development server**
   ```bash
   npm run dev
   ```
   Vite will print a local URL—open it in your browser to explore the surface.
3. **Build for production**
   ```bash
   npm run build
   ```
4. **Preview the production bundle**
   ```bash
   npm run preview
   ```

## State persistence model

- The `LoopLibraryProvider` in `src/state/LoopLibraryStore.tsx` mirrors the reducer/context structure used by `daw-mixer`’s `ProjectStore`. It loads from `localStorage` on boot, hydrates `Float32Array` waveforms, and writes back on every state transition with graceful error handling.
- The provider exposes `exportToFile`/`importFromFile` helpers for engineering workflows. When the File System Access API is available the user gets a native save dialog; otherwise a JSON download is triggered. The exported payload mirrors the stored reducer state so it can be checked into source control or transferred between machines when needed.
- Track uploads are merged into the local library as they are analyzed so every Camelot key remains populated without juggling additional management panels.

## Migrating shared code

The Harmoniq build reuses UI primitives and theming from `daw-mixer` through the `@daw` path alias defined in `tsconfig.json`:

```
"paths": {
  "@daw": ["../daw-mixer/src"],
  "@daw/*": ["../daw-mixer/src/*"]
}
```

When migrating shared modules from `daw-mixer` into a standalone package:

1. Extract the target module(s) into a shared workspace directory (e.g., `packages/ui-kit`) and publish them with a proper `package.json`.
2. Update Harmoniq’s `tsconfig.json` to point the `@daw` alias at the new workspace path, or replace the alias with a standard npm package reference.
3. Run `npm run build` to confirm the TypeScript project resolves the new module location.
4. Repeat the process for any audio DSP helpers that graduate from notebooks into reusable utilities—keep the API surface aligned with `daw-mixer` so the reducer patterns stay interchangeable.

## Testing strategy

Harmoniq currently relies on repository-wide harnesses:

- **Audio engine regression tests** live in `../tests/` (e.g., `test_master_simple.py`, `test_multiband.py`). Run them with `pytest` before shipping changes that touch shared DSP code:
  ```bash
  pytest
  ```
- **Front-end unit and interaction tests** should be authored with [Vitest](https://vitest.dev/) using the existing Vite toolchain. Add a `"test": "vitest"` npm script and install `vitest`/`@testing-library/react` to exercise component render flows (Deck matrix focus logic, loop scheduling, key-sorted track selection, etc.).
- **Integration smoke tests** can mount the Harmoniq bundle inside Playwright or Cypress if you need end-to-end verification. Reuse the reducer fixtures from `LoopLibraryStore` to seed predictable scenarios.

Document newly added tests alongside the components they target so that QA can map coverage quickly. The shared reducer design makes it straightforward to port existing `ProjectStore` test cases into Harmoniq by swapping the action payloads.

## Packaging as a PyWebview executable

The repository includes a helper script, `build_harmoniq_pywebview.bat`, that
produces a Windows executable bundling the Harmoniq production build inside a
PyWebview shell.

1. Install Node.js (18+) and Python 3.10+ on the build machine.
2. Run the batch file from the repository root using an elevated developer
   command prompt:

   ```bat
   build_harmoniq_pywebview.bat
   ```

The script installs npm dependencies, compiles the Vite bundle, provisions a
local Python virtual environment, and invokes the `exe_compiler` helper to run
PyInstaller. The final executable is placed in
`build\\harmoniq_pywebview\\compiler\\dist\\Harmoniq.exe`.
