import { useCallback, useEffect, useMemo, useState } from "react";
import { ProjectProvider, useProjectStore, type PreferencesUpdate } from "./state/ProjectStore";
import { HarmoniqProvider } from "./harmoniq/state";
import { ProjectNavigator } from "./components/ProjectNavigator";
import { Timeline } from "./components/Timeline";
import { MasterControls } from "./components/MasterControls";
import { LiveLooper } from "./components/LiveLooper";
import { SampleDetailPanel } from "./components/SampleDetailPanel";
import { VstRack } from "./components/VstRack";
import type { SampleClip } from "./types";
import { theme } from "./theme";
import { MixerPanel } from "./components/MixerPanel";
import { audioEngine } from "./lib/audioEngine";
import { TopMenu } from "./components/TopMenu";
import soniqLogo from "./assets/soniq-logo.svg";
import { SettingsDialog } from "./components/SettingsDialog";
import { DeckPanel, type DeckId } from "./components/harmoniq/DeckPanel";
import { CrossFader } from "./components/harmoniq/CrossFader";
import { HarmonicWheel } from "./components/harmoniq/HarmonicWheel";
import { LoopLibrary } from "./components/harmoniq/LoopLibrary";
import {
  gridRows,
  toolbarButtonDisabledStyle,
  toolbarButtonStyle
} from "./components/layout/styles";

type FloatingPanel = "mixer" | "vst" | "sample";

function AppShell() {
  const { currentProjectId, projects, dispatch, preferences } = useProjectStore();
  const project = projects[currentProjectId];
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [activePanel, setActivePanel] = useState<FloatingPanel | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [deckAssignments, setDeckAssignments] = useState<Record<DeckId, string | null>>({
    A: null,
    B: null
  });
  const [crossfaderPosition, setCrossfaderPosition] = useState(0.5);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const selectedSample = useMemo<SampleClip | null>(() => {
    if (!selectedSampleId) return null;
    return project.samples.find((sample) => sample.id === selectedSampleId) ?? null;
  }, [selectedSampleId, project.samples]);

  const resolveDeckSample = useCallback(
    (deck: DeckId) => {
      const sampleId = deckAssignments[deck];
      if (!sampleId) return null;
      return project.samples.find((sample) => sample.id === sampleId) ?? null;
    },
    [deckAssignments, project.samples]
  );

  useEffect(() => {
    audioEngine.syncChannelMix(project.channels);
  }, [project.channels]);

  const handleUpdatePreferences = useCallback(
    (update: PreferencesUpdate) => dispatch({ type: "update-preferences", payload: update }),
    [dispatch],
  );

  const panelTitle: Record<FloatingPanel, string> = {
    mixer: "Mixer",
    vst: "VST Rack",
    sample: "Sample Details"
  };

  const handleAssignDeck = useCallback(
    (deck: DeckId) => {
      if (!selectedSample) return;
      setDeckAssignments((previous) => ({ ...previous, [deck]: selectedSample.id }));
    },
    [selectedSample]
  );

  const handleFocusDeck = useCallback(
    (deck: DeckId) => {
      const sampleId = deckAssignments[deck];
      setSelectedSampleId(sampleId ?? null);
    },
    [deckAssignments]
  );

  const handleAssignFromLibrary = useCallback((deck: DeckId, loopId: string) => {
    setDeckAssignments((previous) => ({ ...previous, [deck]: loopId }));
    setSelectedSampleId(loopId);
  }, []);

  const handlePreviewLoop = useCallback((loopId: string) => {
    setSelectedSampleId(loopId);
  }, []);

  const renderPanelContent = () => {
    if (!activePanel) return null;
    if (activePanel === "mixer") return <MixerPanel project={project} />;
    if (activePanel === "vst") return <VstRack project={project} targetSample={selectedSample} />;
    return <SampleDetailPanel sample={selectedSample} />;
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "240px 1fr",
        gridTemplateRows: "auto 1fr",
        minHeight: "100vh",
        background: theme.background,
        color: theme.text,
        position: "relative",
        overflow: "hidden"
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(80% 120% at 80% 0%, rgba(122, 116, 255, 0.16), transparent 55%)",
          mixBlendMode: "screen"
        }}
      />
      <header
        style={{
          gridColumn: "1 / span 2",
          padding: "12px 18px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "16px",
          borderBottom: `1px solid ${theme.border}`,
          background: theme.surfaceRaised,
          boxShadow: theme.shadow,
          position: "relative",
          zIndex: 2
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "18px" }}>
          <TopMenu onOpenSettings={() => setIsSettingsOpen(true)} />
          <img
            src={soniqLogo}
            alt="SONiQ"
            style={{ height: "32px", objectFit: "contain", pointerEvents: "none" }}
          />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <MasterControls project={project} />
          <div style={{ display: "flex", gap: "6px" }}>
            <button type="button" onClick={() => setActivePanel("mixer")} style={toolbarButtonStyle}>
              Mixer
            </button>
            <button type="button" onClick={() => setActivePanel("vst")} style={toolbarButtonStyle}>
              VST
            </button>
            <button
              type="button"
              disabled={!selectedSample}
              onClick={() => selectedSample && setActivePanel("sample")}
              style={{
                ...toolbarButtonStyle,
                ...(selectedSample ? {} : toolbarButtonDisabledStyle)
              }}
            >
              Sample
            </button>
          </div>
        </div>
      </header>

      <aside
        style={{
          gridRow: "2 / span 1",
          background: theme.surface,
          borderRight: `1px solid ${theme.border}`,
          display: "flex",
          flexDirection: "column",
          paddingBottom: "12px",
          position: "relative",
          zIndex: 1
        }}
      >
        <ProjectNavigator onSelectSample={setSelectedSampleId} selectedSampleId={selectedSampleId} />
        <LiveLooper project={project} />
      </aside>

      <main
        style={{
          gridColumn: "2 / span 1",
          padding: "14px 18px",
          display: "grid",
          gridTemplateRows: gridRows("minmax(0, 1fr)", "auto", "minmax(0, 1fr)"),
          gap: "12px",
          position: "relative",
          zIndex: 1
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2.6fr) minmax(0, 1fr)",
            gap: "12px",
            alignItems: "stretch"
          }}
        >
          <DeckPanel
            deckId="A"
            assignedSample={resolveDeckSample("A")}
            onRequestAssign={handleAssignDeck}
            onFocusDeck={handleFocusDeck}
            canAssign={Boolean(selectedSample)}
          />
          <Timeline
            project={project}
            onSelectSample={setSelectedSampleId}
            selectedSampleId={selectedSampleId}
          />
          <DeckPanel
            deckId="B"
            assignedSample={resolveDeckSample("B")}
            onRequestAssign={handleAssignDeck}
            onFocusDeck={handleFocusDeck}
            canAssign={Boolean(selectedSample)}
          />
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)",
            gap: "12px"
          }}
        >
          <CrossFader
            value={crossfaderPosition}
            onChange={setCrossfaderPosition}
            onCueDeck={handleFocusDeck}
          />
          <HarmonicWheel selectedKey={selectedKey} onSelectKey={setSelectedKey} />
        </div>
        <LoopLibrary
          loops={project.samples}
          onPreviewLoop={handlePreviewLoop}
          onAssignToDeck={handleAssignFromLibrary}
        />
      </main>

      {activePanel && (
        <div
          role="dialog"
          aria-modal="true"
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(7, 10, 18, 0.78)",
            backdropFilter: "blur(6px)",
            display: "flex",
            justifyContent: "center",
            alignItems: "flex-start",
            padding: "64px 24px 24px",
            zIndex: 20
          }}
          onClick={() => setActivePanel(null)}
        >
          <div
            style={{
              width: "min(960px, 96vw)",
              maxHeight: "80vh",
              overflow: "hidden",
              background: theme.surfaceRaised,
              borderRadius: "16px",
              border: `1px solid ${theme.border}`,
              boxShadow: theme.shadow,
              display: "flex",
              flexDirection: "column"
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "10px 16px",
                borderBottom: `1px solid ${theme.border}`,
                background: theme.surface
              }}
            >
              <strong style={{ fontSize: "0.85rem", letterSpacing: "0.05em" }}>
                {panelTitle[activePanel]}
              </strong>
              <button
                type="button"
                onClick={() => setActivePanel(null)}
                style={{
                  border: `1px solid ${theme.button.outline}`,
                  background: theme.button.base,
                  color: theme.text,
                  borderRadius: "999px",
                  padding: "4px 10px",
                  fontSize: "0.7rem",
                  cursor: "pointer"
                }}
              >
                Close
              </button>
            </div>
            <div style={{ padding: "14px 16px", overflowY: "auto" }}>{renderPanelContent()}</div>
          </div>
        </div>
      )}

      <SettingsDialog
        isOpen={isSettingsOpen}
        preferences={preferences}
        onClose={() => setIsSettingsOpen(false)}
        onUpdate={handleUpdatePreferences}
      />
    </div>
  );
}

export default function App() {
  return (
    <ProjectProvider>
      <HarmoniqProvider>
        <AppShell />
      </HarmoniqProvider>
    </ProjectProvider>
  );
}
