import { useMemo, useState } from "react";
import { ProjectProvider, useProjectStore } from "./state/ProjectStore";
import { ProjectNavigator } from "./components/ProjectNavigator";
import { Timeline } from "./components/Timeline";
import { TransportControls } from "./components/TransportControls";
import { MasterControls } from "./components/MasterControls";
import { LiveLooper } from "./components/LiveLooper";
import { MasteringPanel } from "./components/MasteringPanel";
import { SampleDetailPanel } from "./components/SampleDetailPanel";
import { VstRack } from "./components/VstRack";
import type { SampleClip } from "./types";
import { theme } from "./theme";

function AppShell() {
  const { currentProjectId, projects } = useProjectStore();
  const project = projects[currentProjectId];
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);

  const selectedSample = useMemo<SampleClip | null>(() => {
    if (!selectedSampleId) return null;
    return project.samples.find((sample) => sample.id === selectedSampleId) ?? null;
  }, [selectedSampleId, project.samples]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "320px 1fr 360px",
        gridTemplateRows: "auto 1fr auto",
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
          gridColumn: "1 / span 3",
          padding: "18px 24px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: `1px solid ${theme.border}`,
          background: theme.surfaceRaised,
          boxShadow: theme.shadow,
          position: "relative",
          zIndex: 1
        }}
      >
        <h1
          style={{
            margin: 0,
            fontSize: "1.75rem",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: theme.text
          }}
        >
          Chromatic Collage Lab
        </h1>
        <MasterControls project={project} />
      </header>

      <aside
        style={{
          gridRow: "2 / span 2",
          background: theme.surface,
          borderRight: `1px solid ${theme.border}`,
          display: "flex",
          flexDirection: "column",
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
          padding: "18px 24px",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
          position: "relative",
          zIndex: 1
        }}
      >
        <Timeline project={project} onSelectSample={setSelectedSampleId} selectedSampleId={selectedSampleId} />
        <TransportControls project={project} />
      </main>

      <aside
        style={{
          gridColumn: "3 / span 1",
          gridRow: "2 / span 2",
          padding: "18px 22px",
          background: theme.surface,
          borderLeft: `1px solid ${theme.border}`,
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          overflowY: "auto",
          position: "relative",
          zIndex: 1
        }}
      >
        <MasteringPanel project={project} />
        <VstRack project={project} targetSample={selectedSample} />
        <SampleDetailPanel sample={selectedSample} />
      </aside>
    </div>
  );
}

export default function App() {
  return (
    <ProjectProvider>
      <AppShell />
    </ProjectProvider>
  );
}
