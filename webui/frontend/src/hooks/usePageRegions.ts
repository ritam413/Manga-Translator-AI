"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { TextRegion, BBox } from "@/lib/types";
import { savePageRegions } from "@/lib/api";

const MAX_STACK_DEPTH = 50;

export function usePageRegions(initialRegions: TextRegion[] = []) {
  const [regions, setRegions] = useState<TextRegion[]>(initialRegions);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);

  // Undo/redo stacks stored in refs to avoid extra re-renders
  const undoStack = useRef<TextRegion[][]>([]);
  const redoStack = useRef<TextRegion[][]>([]);
  const savedState = useRef<TextRegion[]>(initialRegions);

  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [isDirty, setIsDirty] = useState(false);

  // Helper to deep clone regions
  const cloneRegions = (regs: TextRegion[]): TextRegion[] => {
    return JSON.parse(JSON.stringify(regs));
  };

  // Check if two region lists are equal
  const areRegionsEqual = (a: TextRegion[], b: TextRegion[]): boolean => {
    return JSON.stringify(a) === JSON.stringify(b);
  };

  const updateStatus = useCallback(() => {
    setCanUndo(undoStack.current.length > 0);
    setCanRedo(redoStack.current.length > 0);
    setIsDirty(!areRegionsEqual(regions, savedState.current));
  }, [regions]);

  useEffect(() => {
    updateStatus();
  }, [regions, updateStatus]);

  // Reset hook when page changes
  const reset = useCallback((newRegions: TextRegion[]) => {
    const cloned = cloneRegions(newRegions);
    setRegions(cloned);
    setSelectedRegionId(null);
    undoStack.current = [];
    redoStack.current = [];
    savedState.current = cloned;
    setCanUndo(false);
    setCanRedo(false);
    setIsDirty(false);
  }, []);

  const pushToUndo = useCallback((current: TextRegion[]) => {
    undoStack.current.push(cloneRegions(current));
    if (undoStack.current.length > MAX_STACK_DEPTH) {
      undoStack.current.shift();
    }
    redoStack.current = []; // Clear redo stack on new action
  }, []);

  const addRegion = useCallback((box: BBox) => {
    setRegions((prev) => {
      pushToUndo(prev);

      // Generate new sequential id rXXXX
      const maxNum = prev.reduce((max, r) => {
        const match = r.id.match(/^r(\d+)/);
        if (match) {
          const num = parseInt(match[1], 10);
          return num > max ? num : max;
        }
        return max;
      }, -1);
      const newId = `r${String(maxNum + 1).padStart(4, "0")}`;

      const newRegion: TextRegion = {
        id: newId,
        box,
        raw_text: "",
        translated_text: "",
        ocr_confidence: 1.0,
        is_vertical: false,
        needs_review: false,
        region_index: prev.length,
        speaker_id: "",
      };

      const next = [...prev, newRegion];
      setSelectedRegionId(newId);
      return next;
    });
  }, [pushToUndo]);

  const updateRegionBox = useCallback((id: string, box: BBox) => {
    setRegions((prev) => {
      const idx = prev.findIndex((r) => r.id === id);
      if (idx === -1) return prev;

      // Only push to undo if the box coordinates actually changed
      const oldBox = prev[idx].box;
      if (
        oldBox.x_min === box.x_min &&
        oldBox.y_min === box.y_min &&
        oldBox.x_max === box.x_max &&
        oldBox.y_max === box.y_max
      ) {
        return prev;
      }

      pushToUndo(prev);
      const next = cloneRegions(prev);
      next[idx].box = box;
      return next;
    });
  }, [pushToUndo]);

  const updateRegionText = useCallback((id: string, translatedText: string) => {
    setRegions((prev) => {
      const idx = prev.findIndex((r) => r.id === id);
      if (idx === -1) return prev;

      if (prev[idx].translated_text === translatedText) return prev;

      pushToUndo(prev);
      const next = cloneRegions(prev);
      next[idx].translated_text = translatedText;
      return next;
    });
  }, [pushToUndo]);

  const toggleNeedsReview = useCallback((id: string) => {
    setRegions((prev) => {
      const idx = prev.findIndex((r) => r.id === id);
      if (idx === -1) return prev;

      pushToUndo(prev);
      const next = cloneRegions(prev);
      next[idx].needs_review = !next[idx].needs_review;
      return next;
    });
  }, [pushToUndo]);

  const deleteRegion = useCallback((id: string) => {
    setRegions((prev) => {
      const idx = prev.findIndex((r) => r.id === id);
      if (idx === -1) return prev;

      pushToUndo(prev);
      const next = prev.filter((r) => r.id !== id);
      if (selectedRegionId === id) {
        setSelectedRegionId(null);
      }
      return next;
    });
  }, [selectedRegionId, pushToUndo]);

  const undo = useCallback(() => {
    if (undoStack.current.length === 0) return;
    const prevSnapshot = undoStack.current.pop()!;
    redoStack.current.push(cloneRegions(regions));
    setRegions(prevSnapshot);
  }, [regions]);

  const redo = useCallback(() => {
    if (redoStack.current.length === 0) return;
    const nextSnapshot = redoStack.current.pop()!;
    undoStack.current.push(cloneRegions(regions));
    setRegions(nextSnapshot);
  }, [regions]);

  const save = useCallback(async (projectDir: string, filename: string) => {
    try {
      const res = await savePageRegions(projectDir, filename, regions);
      if (res.success) {
        savedState.current = cloneRegions(regions);
        undoStack.current = [];
        redoStack.current = [];
        updateStatus();
        return true;
      }
      return false;
    } catch (e) {
      console.error("Failed to save regions", e);
      return false;
    }
  }, [regions, updateStatus]);

  return {
    regions,
    selectedRegionId,
    setSelectedRegionId,
    isDirty,
    canUndo,
    canRedo,
    reset,
    addRegion,
    updateRegionBox,
    updateRegionText,
    toggleNeedsReview,
    deleteRegion,
    undo,
    redo,
    save,
  };
}
