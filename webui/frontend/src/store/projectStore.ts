import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ProjectStore {
  projectDir: string;
  inputSubdir: string;
  modelVersion: "v1" | "v2";
  setProjectDir: (dir: string) => void;
  setInputSubdir: (sub: string) => void;
  setModelVersion: (v: "v1" | "v2") => void;
}

export const useProjectStore = create<ProjectStore>()(
  persist(
    (set) => ({
      projectDir: "e:\\u torrentz\\MangaTranslator FUll STACK\\Manga Translator",
      inputSubdir: "input",
      modelVersion: "v2",
      setProjectDir: (dir) => set({ projectDir: dir }),
      setInputSubdir: (sub) => set({ inputSubdir: sub }),
      setModelVersion: (v) => set({ modelVersion: v }),
    }),
    { name: "manga-translator-project" }
  )
);
