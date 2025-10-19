import { useState } from "react";
import { nanoid } from "nanoid";
import type { Project, SampleClip } from "../types";
import {
  FxRack,
  type FxPluginDescriptor,
  deriveFxPluginType,
} from "../shared/FxRack";

interface VstRackProps {
  project: Project;
  targetSample?: SampleClip | null;
}

export function VstRack({ project, targetSample }: VstRackProps) {
  const [plugins, setPlugins] = useState<FxPluginDescriptor[]>([]);

  const addPlugin = (file: File | null) => {
    if (!file) return;
    const plugin: FxPluginDescriptor = {
      id: nanoid(),
      name: file.name.replace(/\.[^.]+$/, ""),
      type: deriveFxPluginType(file),
      loadedAt: new Date(),
      fileName: file.name,
    };
    setPlugins((prev) => [...prev, plugin]);
  };

  return (
    <FxRack
      title="VST Rack"
      description="Drop WebAssembly VSTs or effect presets to extend your collage toolkit."
      targetName={targetSample ? targetSample.name : project.name}
      actionLabel="Load VST"
      plugins={plugins}
      emptyStateMessage="No VSTs loaded yet. Import compatible WASM or preset files to chain them after the mastering rack."
      onLoadPlugin={addPlugin}
    />
  );
}
