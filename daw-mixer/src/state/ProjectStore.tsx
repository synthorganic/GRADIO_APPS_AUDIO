import React, { createContext, useContext, useMemo, useReducer } from "react";
import { nanoid } from "nanoid";
import type {
  AudioChannel,
  AutomationChannel,
  MasteringSettings,
  MidiChannel,
  Project,
  SampleClip,
  StemInfo,
  TimelineChannel,
} from "../types";
import { theme } from "../theme";

export type ProjectAction =
  | { type: "add-sample"; projectId: string; sample: SampleClip }
  | { type: "update-sample"; projectId: string; sampleId: string; sample: Partial<SampleClip> }
  | { type: "remove-sample"; projectId: string; sampleId: string }
  | { type: "add-project"; project: Project }
  | { type: "update-mastering"; projectId: string; payload: Partial<MasteringSettings> }
  | { type: "set-project"; project: Project }
  | { type: "add-channel"; projectId: string; channel: TimelineChannel }
  | {
      type: "update-channel";
      projectId: string;
      channelId: string;
      patch:
        | Partial<AudioChannel>
        | Partial<AutomationChannel>
        | Partial<MidiChannel>;
    }
  | { type: "set-scale"; projectId: string; scale: string }
  | { type: "register-control-change"; target: AutomationTarget | null };

export interface AutomationTarget {
  id: string;
  label: string;
  value: number;
  unit?: string;
}

interface ProjectState {
  currentProjectId: string;
  projects: Record<string, Project>;
  lastControlTarget: AutomationTarget | null;
}

const rainbowPalette: Record<StemInfo["type"], string> = {
  full: theme.accentBeam[0],
  vocals: theme.accentBeam[1],
  leads: theme.accentBeam[2],
  percussion: theme.accentBeam[3],
  kicks: theme.accentBeam[4],
  bass: theme.accentBeam[5]
};

function createAudioChannel(name: string): AudioChannel {
  return {
    id: nanoid(),
    name,
    type: "audio",
    color: theme.surface,
    isFxEnabled: true,
    volume: 0.85,
    pan: 0,
  };
}

const initialProject: Project = {
  id: nanoid(),
  name: "Chromatic Collage",
  masterBpm: 120,
  scale: "C Major",
  samples: [],
  channels: [createAudioChannel("Channel 1")],
  mastering: {
    widenStereo: 0.35,
    glueCompression: 0.45,
    spectralTilt: 0.1,
    limiterCeiling: -0.3,
    tapeSaturation: 0.2
  }
};

const initialState: ProjectState = {
  currentProjectId: initialProject.id,
  projects: {
    [initialProject.id]: initialProject
  },
  lastControlTarget: null
};

function reducer(state: ProjectState, action: ProjectAction): ProjectState {
  switch (action.type) {
    case "add-project": {
      return {
        currentProjectId: action.project.id,
        projects: {
          ...state.projects,
          [action.project.id]: action.project
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "set-project": {
      return {
        currentProjectId: action.project.id,
        projects: {
          ...state.projects,
          [action.project.id]: action.project
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "add-sample": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            samples: [...project.samples, action.sample]
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "update-sample": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      const updatedSamples = project.samples.map((sample) =>
        sample.id === action.sampleId ? { ...sample, ...action.sample } : sample
      );
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            samples: updatedSamples
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "remove-sample": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            samples: project.samples.filter((sample) => sample.id !== action.sampleId)
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "update-mastering": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            mastering: {
              ...project.mastering,
              ...action.payload
            }
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "add-channel": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      let normalizedChannel: TimelineChannel;
      if (action.channel.type === "audio") {
        normalizedChannel = {
          ...action.channel,
          volume: action.channel.volume ?? 0.85,
          pan: action.channel.pan ?? 0
        };
      } else if (action.channel.type === "automation") {
        normalizedChannel = {
          ...action.channel,
          volume: action.channel.volume ?? 0.85,
          pan: action.channel.pan ?? 0
        };
      } else {
        const midiChannel = action.channel as MidiChannel;
        normalizedChannel = {
          ...midiChannel,
          blocks: midiChannel.blocks ?? [],
          blockSizeMeasures: midiChannel.blockSizeMeasures ?? 1,
          volume: midiChannel.volume ?? 0.85,
          pan: midiChannel.pan ?? 0
        };
      }
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            channels: [...project.channels, normalizedChannel]
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "set-scale": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            scale: action.scale
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "update-channel": {
      const project = state.projects[action.projectId];
      if (!project) return state;
      return {
        ...state,
        projects: {
          ...state.projects,
          [project.id]: {
            ...project,
            channels: project.channels.map((channel) => {
              if (channel.id !== action.channelId) return channel;
              if (channel.type === "audio") {
                return { ...channel, ...(action.patch as Partial<AudioChannel>) };
              }
              if (channel.type === "automation") {
                return { ...channel, ...(action.patch as Partial<AutomationChannel>) };
              }
              return { ...channel, ...(action.patch as Partial<MidiChannel>) };
            })
          }
        },
        lastControlTarget: state.lastControlTarget
      };
    }
    case "register-control-change": {
      return {
        ...state,
        lastControlTarget: action.target
      };
    }
    default:
      return state;
  }
}

interface ProjectContextValue extends ProjectState {
  dispatch: React.Dispatch<ProjectAction>;
  getPalette: () => typeof rainbowPalette;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const value = useMemo(
    () => ({
      ...state,
      dispatch,
      getPalette: () => rainbowPalette
    }),
    [state]
  );

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export function useProjectStore() {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error("ProjectStore must be used inside ProjectProvider");
  return ctx;
}
