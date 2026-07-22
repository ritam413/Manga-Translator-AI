"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { TextRegion, BBox } from "@/lib/types";
import { ZoomPanViewport, ZoomPanRef } from "./ZoomPanViewport";
import { imageUrl } from "@/lib/api";

interface CanvasEditorProps {
  projectDir: string;
  inputSubdir: string;
  filename: string;
  imageWidth: number;
  imageHeight: number;
  regions: TextRegion[];
  selectedRegionId: string | null;
  setSelectedRegionId: (id: string | null) => void;
  addRegion: (box: BBox) => void;
  updateRegionBox: (id: string, box: BBox) => void;
  deleteRegion: (id: string) => void;
  drawMode: boolean;
  setDrawMode: (mode: boolean) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  save: (projectDir: string, filename: string) => Promise<boolean>;
  isDirty: boolean;
}

export default function CanvasEditor({
  projectDir,
  inputSubdir,
  filename,
  imageWidth = 800,
  imageHeight = 1200,
  regions,
  selectedRegionId,
  setSelectedRegionId,
  addRegion,
  updateRegionBox,
  deleteRegion,
  drawMode,
  setDrawMode,
  undo,
  redo,
  canUndo,
  canRedo,
  save,
  isDirty,
}: CanvasEditorProps) {
  const [zoomLevel, setZoomLevel] = useState(1);
  const zoomPanRef = useRef<ZoomPanRef>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Output image cache-busting timestamp state
  const [outputImageTimestamp, setOutputImageTimestamp] = useState(Date.now());
  const [isSaving, setIsSaving] = useState(false);

  // Track if spacebar is held down
  const spacePressed = useRef(false);

  // Dragging states
  const [dragAction, setDragAction] = useState<{
    type: "move" | "resize" | "draw";
    regionId?: string;
    handle?: "tl" | "tr" | "bl" | "br";
    startBox?: BBox;
    startX: number; // in image pixels
    startY: number; // in image pixels
  } | null>(null);

  const [tempDrawBox, setTempDrawBox] = useState<BBox | null>(null);

  const inputImgUrl = imageUrl(projectDir, "input", filename, inputSubdir);
  const outputImgUrl = imageUrl(projectDir, "output", filename, inputSubdir);

  // Map screen coordinates (clientX/Y) to local image coordinates
  const getLocalCoords = (clientX: number, clientY: number): { x: number; y: number } => {
    if (!svgRef.current) return { x: 0, y: 0 };
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * imageWidth;
    const y = ((clientY - rect.top) / rect.height) * imageHeight;
    return {
      x: Math.max(0, Math.min(imageWidth, x)),
      y: Math.max(0, Math.min(imageHeight, y)),
    };
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    // Only allow left click (0)
    if (e.button !== 0) return;

    // If spacebar is held down, allow standard panning behavior to take over (bubble up)
    if (spacePressed.current) return;

    const target = e.target as SVGElement;
    const regionId = target.getAttribute("data-region-id");
    const handleType = target.getAttribute("data-handle-type"); // "tl", "tr", "bl", "br"

    const coords = getLocalCoords(e.clientX, e.clientY);

    // 1. Draw Mode starts drawing (only draw if drawMode is explicitly enabled)
    if (drawMode) {
      setDragAction({
        type: "draw",
        startX: coords.x,
        startY: coords.y,
      });
      setTempDrawBox({
        x_min: coords.x,
        y_min: coords.y,
        x_max: coords.x,
        y_max: coords.y,
      });
      setSelectedRegionId(null);
      e.preventDefault();
      return;
    }

    // 2. Corner handle resize clicked
    if (handleType && regionId) {
      const region = regions.find((r) => r.id === regionId);
      if (region) {
        setDragAction({
          type: "resize",
          regionId,
          handle: handleType as "tl" | "tr" | "bl" | "br",
          startBox: { ...region.box },
          startX: coords.x,
          startY: coords.y,
        });
        setSelectedRegionId(regionId);
      }
      e.stopPropagation();
      e.preventDefault();
      return;
    }

    // 3. Move region box clicked
    if (regionId) {
      const region = regions.find((r) => r.id === regionId);
      if (region) {
        setDragAction({
          type: "move",
          regionId,
          startBox: { ...region.box },
          startX: coords.x,
          startY: coords.y,
        });
        setSelectedRegionId(regionId);
      }
      e.stopPropagation();
      e.preventDefault();
      return;
    }
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!dragAction) return;

    const coords = getLocalCoords(e.clientX, e.clientY);
    const dx = coords.x - dragAction.startX;
    const dy = coords.y - dragAction.startY;

    if (dragAction.type === "draw" && tempDrawBox) {
      setTempDrawBox({
        x_min: Math.min(dragAction.startX, coords.x),
        y_min: Math.min(dragAction.startY, coords.y),
        x_max: Math.max(dragAction.startX, coords.x),
        y_max: Math.max(dragAction.startY, coords.y),
      });
    } else if (dragAction.type === "move" && dragAction.regionId && dragAction.startBox) {
      const box = dragAction.startBox;
      const w = box.x_max - box.x_min;
      const h = box.y_max - box.y_min;

      let x_min = box.x_min + dx;
      let y_min = box.y_min + dy;

      // Clamp coordinates to image boundaries
      if (x_min < 0) x_min = 0;
      if (y_min < 0) y_min = 0;
      if (x_min + w > imageWidth) x_min = imageWidth - w;
      if (y_min + h > imageHeight) y_min = imageHeight - h;

      updateRegionBox(dragAction.regionId, {
        x_min: Math.round(x_min),
        y_min: Math.round(y_min),
        x_max: Math.round(x_min + w),
        y_max: Math.round(y_min + h),
      });
    } else if (dragAction.type === "resize" && dragAction.regionId && dragAction.startBox && dragAction.handle) {
      const box = { ...dragAction.startBox };

      if (dragAction.handle === "tl") {
        box.x_min = Math.min(dragAction.startBox.x_max - 10, Math.max(0, box.x_min + dx));
        box.y_min = Math.min(dragAction.startBox.y_max - 10, Math.max(0, box.y_min + dy));
      } else if (dragAction.handle === "tr") {
        box.x_max = Math.max(dragAction.startBox.x_min + 10, Math.min(imageWidth, box.x_max + dx));
        box.y_min = Math.min(dragAction.startBox.y_max - 10, Math.max(0, box.y_min + dy));
      } else if (dragAction.handle === "bl") {
        box.x_min = Math.min(dragAction.startBox.x_max - 10, Math.max(0, box.x_min + dx));
        box.y_max = Math.max(dragAction.startBox.y_min + 10, Math.min(imageHeight, box.y_max + dy));
      } else if (dragAction.handle === "br") {
        box.x_max = Math.max(dragAction.startBox.x_min + 10, Math.min(imageWidth, box.x_max + dx));
        box.y_max = Math.max(dragAction.startBox.y_min + 10, Math.min(imageHeight, box.y_max + dy));
      }

      updateRegionBox(dragAction.regionId, {
        x_min: Math.round(box.x_min),
        y_min: Math.round(box.y_min),
        x_max: Math.round(box.x_max),
        y_max: Math.round(box.y_max),
      });
    }
  };

  const handlePointerUp = () => {
    if (!dragAction) return;

    if (dragAction.type === "draw" && tempDrawBox) {
      const w = tempDrawBox.x_max - tempDrawBox.x_min;
      const h = tempDrawBox.y_max - tempDrawBox.y_min;
      // Accept minimum box dimensions
      if (w > 10 && h > 10) {
        addRegion({
          x_min: Math.round(tempDrawBox.x_min),
          y_min: Math.round(tempDrawBox.y_min),
          x_max: Math.round(tempDrawBox.x_max),
          y_max: Math.round(tempDrawBox.y_max),
        });
      }
      setTempDrawBox(null);
      setDrawMode(false); // disable draw mode after drawing
    }

    setDragAction(null);
  };

  const handleSave = useCallback(async () => {
    if (isSaving) return;
    setIsSaving(true);
    try {
      const success = await save(projectDir, filename);
      if (success) {
        setOutputImageTimestamp(Date.now());
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  }, [save, projectDir, filename, isSaving]);

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        spacePressed.current = true;
      } else if (e.key === "Escape") {
        setSelectedRegionId(null);
        setDrawMode(false);
      } else if (e.key === "Delete" || e.key === "Backspace") {
        if (selectedRegionId) {
          if (confirm(`Delete speech bubble region ${selectedRegionId}?`)) {
            deleteRegion(selectedRegionId);
          }
        }
      } else if (e.ctrlKey && e.key.toLowerCase() === "z") {
        if (e.shiftKey) {
          e.preventDefault();
          redo();
        } else {
          e.preventDefault();
          undo();
        }
      } else if (e.ctrlKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        handleSave();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        spacePressed.current = false;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [selectedRegionId, undo, redo, handleSave, deleteRegion, setSelectedRegionId, setDrawMode]);

  return (
    <div className="flex-1 relative flex flex-col overflow-hidden clay-panel glass-backdrop">
      {/* Canvas Toolbelt */}
      <div className="h-12 flex items-center justify-between px-4 flex-shrink-0 z-10 select-none">
        <div className="flex items-center gap-2">
          {/* Cursor / Select mode */}
          <button
            onClick={() => setDrawMode(false)}
            className={`px-3 py-1 text-xs font-semibold flex items-center gap-1 transition-all ${
              !drawMode ? "clay-btn-active" : "clay-btn"
            }`}
          >
            <span>✥</span> Select
          </button>
          {/* Draw box mode */}
          <button
            onClick={() => setDrawMode(true)}
            className={`px-3 py-1 text-xs font-semibold flex items-center gap-1 transition-all ${
              drawMode ? "clay-btn-active" : "clay-btn"
            }`}
          >
            <span>✚</span> Draw Box
          </button>
        </div>

        {/* Undo/Redo & Save Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={undo}
            disabled={!canUndo}
            className="w-7 h-7 flex items-center justify-center clay-btn disabled:opacity-20 disabled:pointer-events-none text-xs transition-all"
            title="Undo (Ctrl+Z)"
          >
            ↩
          </button>
          <button
            onClick={redo}
            disabled={!canRedo}
            className="w-7 h-7 flex items-center justify-center clay-btn disabled:opacity-20 disabled:pointer-events-none text-xs transition-all"
            title="Redo (Ctrl+Shift+Z)"
          >
            ↪
          </button>
          <div className="w-px h-5 bg-white/10 mx-1" />
          <button
            onClick={handleSave}
            disabled={!isDirty || isSaving}
            className={`px-4 py-1 text-xs font-semibold flex items-center gap-1 transition-all ${
              isDirty && !isSaving
                ? "clay-btn-primary cursor-pointer"
                : "clay-well text-white/30 cursor-default"
            }`}
          >
            {isSaving ? "⏳ Saving..." : "💾 Save"}
          </button>
        </div>

        {/* Float Zoom controls */}
        <div className="flex items-center gap-0.5 clay-well p-0.5 rounded-xl text-xs">
          <button
            onClick={() => zoomPanRef.current?.zoomOut()}
            className="w-6 h-6 flex items-center justify-center hover:bg-white/5 rounded cursor-pointer text-gray-300"
          >
            -
          </button>
          <button
            onClick={() => zoomPanRef.current?.reset()}
            className="px-2 py-0.5 font-mono hover:bg-white/5 rounded cursor-pointer text-[10px] text-gray-300"
          >
            {Math.round(zoomLevel * 100)}%
          </button>
          <button
            onClick={() => zoomPanRef.current?.zoomIn()}
            className="w-6 h-6 flex items-center justify-center hover:bg-white/5 rounded cursor-pointer text-gray-300"
          >
            +
          </button>
        </div>
      </div>

      {/* Viewport Workspace */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center clay-well m-2 rounded-2xl">
        <ZoomPanViewport
          ref={zoomPanRef}
          zoomLevel={zoomLevel}
          setZoomLevel={setZoomLevel}
          resetKey={`editor-${filename}`}
        >
          <div className="flex gap-4 items-center select-none">
            {/* Left panel: Original (Input) image + interactive SVG layer */}
            <div
              className="relative select-none rounded-lg overflow-hidden border border-white/10 shadow-2xl bg-[#0f1215]"
              style={{
                aspectRatio: `${imageWidth} / ${imageHeight}`,
                height: "75vh",
                maxHeight: "75vh",
                maxWidth: "40vw",
                pointerEvents: "auto",
              }}
            >
              {/* Badge: Original */}
              <div className="absolute top-3 left-3 px-2 py-0.5 bg-black/60 text-xs rounded border border-white/10 z-10 text-white select-none">
                Original
              </div>

              {/* Background Manga Image */}
              <img
                src={inputImgUrl}
                alt="Original"
                className="w-full h-full object-contain"
                draggable={false}
              />

              {/* Interactive SVG Editing Layer */}
              <svg
                ref={svgRef}
                viewBox={`0 0 ${imageWidth} ${imageHeight}`}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerUp}
                className="absolute inset-0 select-none cursor-crosshair"
                style={{
                  width: "100%",
                  height: "100%",
                  pointerEvents: "auto",
                  cursor: drawMode ? "crosshair" : "default",
                }}
              >
                {/* Existing Bounding Boxes */}
                {regions.map((region) => {
                  const box = region.box;
                  const isSelected = region.id === selectedRegionId;
                  const strokeColor = isSelected
                    ? "#4ade80"
                    : region.needs_review
                    ? "#fbbf24"
                    : "rgba(74, 222, 128, 0.4)";
                  const fillColor = isSelected
                    ? "rgba(74, 222, 128, 0.12)"
                    : region.needs_review
                    ? "rgba(251, 191, 36, 0.04)"
                    : "rgba(74, 222, 128, 0.02)";
                  const strokeWidth = isSelected ? 3 : 1.5;

                  return (
                    <g key={region.id} className="group select-none">
                      {/* Rectangle outline + hit area */}
                      <rect
                        x={box.x_min}
                        y={box.y_min}
                        width={box.x_max - box.x_min}
                        height={box.y_max - box.y_min}
                        data-region-id={region.id}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        rx="4"
                        className="transition-colors duration-150"
                        style={{
                          cursor: drawMode ? "crosshair" : "move",
                        }}
                      />

                      {/* Region ID badge drawn in top-left */}
                      <g transform={`translate(${box.x_min}, ${box.y_min - 12})`} className="pointer-events-none">
                        <rect
                          x="0"
                          y="0"
                          width={region.id.length * 6 + 10}
                          height="12"
                          rx="2"
                          fill={isSelected ? "#4ade80" : "#0f1215"}
                          opacity="0.85"
                        />
                        <text
                          x="5"
                          y="9"
                          fill={isSelected ? "#000" : "#a3e4a3"}
                          fontSize="8px"
                          fontWeight="700"
                          fontFamily="monospace"
                        >
                          {region.id}
                        </text>
                      </g>

                      {/* Bounding box Resize handles (only when selected) */}
                      {isSelected && (
                        <>
                          {/* Top-Left */}
                          <circle
                            cx={box.x_min}
                            cy={box.y_min}
                            r="6"
                            fill="#4ade80"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            data-region-id={region.id}
                            data-handle-type="tl"
                            className="cursor-nwse-resize hover:scale-125 transition-transform"
                          />
                          {/* Top-Right */}
                          <circle
                            cx={box.x_max}
                            cy={box.y_min}
                            r="6"
                            fill="#4ade80"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            data-region-id={region.id}
                            data-handle-type="tr"
                            className="cursor-nesw-resize hover:scale-125 transition-transform"
                          />
                          {/* Bottom-Left */}
                          <circle
                            cx={box.x_min}
                            cy={box.y_max}
                            r="6"
                            fill="#4ade80"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            data-region-id={region.id}
                            data-handle-type="bl"
                            className="cursor-nesw-resize hover:scale-125 transition-transform"
                          />
                          {/* Bottom-Right */}
                          <circle
                            cx={box.x_max}
                            cy={box.y_max}
                            r="6"
                            fill="#4ade80"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            data-region-id={region.id}
                            data-handle-type="br"
                            className="cursor-nwse-resize hover:scale-125 transition-transform"
                          />
                        </>
                      )}
                    </g>
                  );
                })}

                {/* Temporary drag-to-draw rectangle preview */}
                {tempDrawBox && (
                  <rect
                    x={tempDrawBox.x_min}
                    y={tempDrawBox.y_min}
                    width={tempDrawBox.x_max - tempDrawBox.x_min}
                    height={tempDrawBox.y_max - tempDrawBox.y_min}
                    fill="rgba(74, 222, 128, 0.15)"
                    stroke="#4ade80"
                    strokeWidth="2"
                    strokeDasharray="4 3"
                    rx="4"
                  />
                )}
              </svg>
            </div>

            {/* Right panel: Translated (Output) image */}
            <div
              className="relative select-none rounded-lg overflow-hidden border border-[#4ade80]/15 shadow-2xl bg-[#0f1215]"
              style={{
                aspectRatio: `${imageWidth} / ${imageHeight}`,
                height: "75vh",
                maxWidth: "40vw",
                pointerEvents: "none",
              }}
            >
              {/* Badge: Translated */}
              <div className="absolute top-3 left-3 px-2 py-0.5 bg-black/60 text-xs rounded border border-white/10 z-10 text-white select-none">
                Translated
              </div>

              {/* Background Translated Image */}
              <img
                src={`${outputImgUrl}&t=${outputImageTimestamp}`}
                alt="Translated"
                className="w-full h-full object-contain"
                onError={(e) => {
                  (e.target as HTMLElement).style.display = "none";
                }}
                draggable={false}
              />
            </div>
          </div>
        </ZoomPanViewport>
      </div>
    </div>
  );
}
