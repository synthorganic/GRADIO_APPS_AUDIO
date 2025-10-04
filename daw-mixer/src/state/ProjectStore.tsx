import React, { createContext, useContext, useMemo, useReducer } from "react";
import { nanoid } from "nanoid";
import type { MasteringSettings, Project, SampleClip, StemInfo } from "../types";
import { theme } from "../theme";

export type ProjectAction =
  | { type: "add-sample"; projectId: string; sample: SampleClip }
  | { type: "update-sample"; projectId: string; sampleId: string; sample: Partial<SampleClip> }
  | { type: "remove-sample"; projectId: string; sampleId: string }
  | { type: "add-project"; project: Project }
  | { type: "update-mastering"; projectId: string; payload: Partial<MasteringSettings> }
  | { type: "set-project"; project: Project };

interface ProjectState {
  currentProjectId: string;
  projects: Record<string, Project>;
}

const rainbowPalette: Record<StemInfo["type"], string> = {
  full: theme.accentBeam[0],
  vocals: theme.accentBeam[1],
  leads: theme.accentBeam[2],
  percussion: theme.accentBeam[3],
  kicks: theme.accentBeam[4],
  bass: theme.accentBeam[5]
};

const initialProject: Project = {
  id: nanoid(),
  name: "Chromatic Collage",
  masterBpm: 120,
  samples: [],
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
  }
};

function reducer(state: ProjectState, action: ProjectAction): ProjectState {
  switch (action.type) {
    case "add-project": {
      return {
        currentProjectId: action.project.id,
        projects: {
          ...state.projects,
          [action.project.id]: action.project
        }
      };
    }
    case "set-project": {
      return {
        currentProjectId: action.project.id,
        projects: {
          ...state.projects,
          [action.project.id]: action.project
        }
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
        }
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
        }
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
        }
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
        }
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
