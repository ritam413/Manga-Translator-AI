"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { imageUrl } from "@/lib/api";
import { PageStatus } from "@/lib/types";
import { ZoomPanViewport, ZoomPanRef } from "./ZoomPanViewport";

// Import img-comparison-slider web component (client-side only)
import "img-comparison-slider";

interface ImageViewerProps {
  projectDir: string;
  inputSubdir: string;
  pages: PageStatus[];
  initialIndex: number;
  onClose: () => void;
}

// ── Zoom Controls Panel ─────────────────────────────────────────────────────

function ZoomControls({
  zoomLevel,
  zoomIn,
  zoomOut,
  reset,
}: {
  zoomLevel: number;
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
}) {
  return (
    <div className="flex bg-white/5 rounded-lg border border-white/10 p-0.5 mr-2">
      <button
        onClick={zoomOut}
        className="w-8 h-8 flex items-center justify-center hover:bg-white/5 rounded cursor-pointer"
        title="Zoom Out"
      >
        -
      </button>
      <button
        onClick={reset}
        className="px-3 py-[10px] text-xs font-mono hover:bg-white/5 rounded cursor-pointer"
        title="Reset Zoom"
        style={{ padding: "10px 12px" }}
      >
        {Math.round(zoomLevel * 100)}%
      </button>
      <button
        onClick={zoomIn}
        className="w-8 h-8 flex items-center justify-center hover:bg-white/5 rounded cursor-pointer"
        title="Zoom In"
      >
        +
      </button>
    </div>
  );
}

// ── Bounding Box Overlay ────────────────────────────────────────────────────

function BoundingBoxOverlay({
  pageDetails,
  onRegionHover,
}: {
  pageDetails: any;
  onRegionHover: (region: any | null) => void;
}) {
  if (!pageDetails?.text_regions) return null;

  return (
    <div className="absolute inset-0 pointer-events-none">
      {pageDetails.text_regions.map((region: any) => {
        const width = pageDetails.width || 1;
        const height = pageDetails.height || 1;
        const left = (region.box.x_min / width) * 100;
        const top = (region.box.y_min / height) * 100;
        const w = ((region.box.x_max - region.box.x_min) / width) * 100;
        const h = ((region.box.y_max - region.box.y_min) / height) * 100;

        return (
          <div
            key={region.id}
            onMouseEnter={() => onRegionHover(region)}
            onMouseLeave={() => onRegionHover(null)}
            className={`absolute border rounded pointer-events-auto cursor-pointer transition-all ${
              region.needs_review
                ? "border-amber/60 bg-amber/5 hover:border-amber hover:bg-amber/15"
                : "border-[#4ade80]/40 bg-[#4ade80]/5 hover:border-[#4ade80] hover:bg-[#4ade80]/15"
            }`}
            style={{
              left: `${left}%`,
              top: `${top}%`,
              width: `${w}%`,
              height: `${h}%`,
            }}
          />
        );
      })}
    </div>
  );
}

// ── Main ImageViewer ────────────────────────────────────────────────────────

export default function ImageViewer({
  projectDir,
  inputSubdir,
  pages,
  initialIndex,
  onClose,
}: ImageViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [viewMode, setViewMode] = useState<"input" | "output" | "side-by-side" | "swipe">("swipe");
  const [showBoxes, setShowBoxes] = useState(true);
  const [activeRegion, setActiveRegion] = useState<any | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const zoomPanRef = useRef<ZoomPanRef>(null);
  const sliderRef = useRef<any>(null);

  const handleSliderPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation(); // prevent panning

    const handleEl = e.currentTarget;
    handleEl.setPointerCapture(e.pointerId);

    const updateValue = (clientX: number) => {
      const slider = sliderRef.current;
      if (!slider) return;
      const rect = slider.getBoundingClientRect();
      if (rect.width === 0) return;
      const percentage = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      slider.value = percentage;
    };

    updateValue(e.clientX);

    const onPointerMove = (moveEvent: PointerEvent) => {
      moveEvent.stopPropagation();
      updateValue(moveEvent.clientX);
    };

    const onPointerUp = (upEvent: PointerEvent) => {
      upEvent.stopPropagation();
      try {
        handleEl.releasePointerCapture(upEvent.pointerId);
      } catch (err) {}
      handleEl.removeEventListener("pointermove", onPointerMove);
      handleEl.removeEventListener("pointerup", onPointerUp);
      handleEl.removeEventListener("pointercancel", onPointerUp);
    };

    handleEl.addEventListener("pointermove", onPointerMove);
    handleEl.addEventListener("pointerup", onPointerUp);
    handleEl.addEventListener("pointercancel", onPointerUp);
  };

  // Full page details (for bounding boxes) loaded from manifest engine data
  const [pageDetails, setPageDetails] = useState<any | null>(null);

  const currentPage = pages[currentIndex];

  // Fetch page details for bounding boxes
  useEffect(() => {
    if (!currentPage) return;
    setPageDetails(null);
    setActiveRegion(null);

    const fetchPageDetail = async () => {
      try {
        const res = await fetch(
          `/api/project/page-detail?project_dir=${encodeURIComponent(projectDir)}&filename=${encodeURIComponent(currentPage.filename)}`
        );
        if (res.ok) {
          const detail = await res.json();
          setPageDetails(detail);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchPageDetail();
  }, [currentPage, currentIndex, projectDir]);

  // Key navigation
  const handlePrev = useCallback(() => {
    if (currentIndex > 0) setCurrentIndex(currentIndex - 1);
  }, [currentIndex]);

  const handleNext = useCallback(() => {
    if (currentIndex < pages.length - 1) setCurrentIndex(currentIndex + 1);
  }, [currentIndex, pages.length]);

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

      if (e.key === "ArrowLeft") handlePrev();
      else if (e.key === "ArrowRight") handleNext();
      else if (e.key === "Escape") onClose();
      else if (e.key === "=" || e.key === "+") {
        zoomPanRef.current?.zoomIn();
      } else if (e.key === "-") {
        zoomPanRef.current?.zoomOut();
      } else if (e.key === "0") {
        zoomPanRef.current?.reset();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handlePrev, handleNext, onClose]);

  const zoomIn = () => zoomPanRef.current?.zoomIn();
  const zoomOut = () => zoomPanRef.current?.zoomOut();
  const reset = () => zoomPanRef.current?.reset();

  // Build Image URLs
  const inputImgUrl = currentPage ? imageUrl(projectDir, "input", currentPage.filename, inputSubdir) : "";
  const outputImgUrl = currentPage ? imageUrl(projectDir, "output", currentPage.filename, inputSubdir) : "";

  // View mode buttons config
  const viewModes = [
    { key: "input" as const, label: "Original (JA)" },
    { key: "output" as const, label: "Translated (EN)" },
    { key: "side-by-side" as const, label: "Side by Side" },
    { key: "swipe" as const, label: "Interactive Swipe" },
  ];

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-[#070a0d]/95 backdrop-blur-md text-white select-none transition-all duration-300">
      {/* Top Header Controls */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-white/5 bg-[#0a0c0f]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <button
            onClick={onClose}
            className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 hover:text-red transition-all cursor-pointer"
            title="Close (Esc)"
          >
            ✕
          </button>
          <div>
            <div className="text-sm font-semibold flex items-center gap-2">
              <span className="font-mono text-xs px-2 py-0.5 bg-white/5 rounded border border-white/10">
                PAGE {currentIndex + 1} / {pages.length}
              </span>
              <span className="font-mono text-[#a3e4a3]">{currentPage?.filename}</span>
            </div>
            <div className="text-[10px] text-white/40 flex items-center gap-2 mt-0.5">
              <span>Regions: {currentPage?.region_count ?? 0}</span>
              {currentPage?.needs_review_count > 0 && (
                <span className="text-amber">● {currentPage.needs_review_count} needs review</span>
              )}
            </div>
          </div>
        </div>

        {/* View Mode Selectors */}
        <div className="flex gap-1.5 bg-white/5 p-1 rounded-lg border border-white/10 text-xs">
          {viewModes.map((mode) => (
            <button
              key={mode.key}
              onClick={() => setViewMode(mode.key)}
              className={`px-4 py-[10px] rounded-md transition-all cursor-pointer ${
                viewMode === mode.key
                  ? "bg-[#4ade80] text-black font-semibold"
                  : "text-white/70 hover:text-white"
              }`}
              style={{ padding: "8px 16px" }}
            >
              {mode.label}
            </button>
          ))}
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {/* Zoom controls are rendered inside TransformWrapper via ZoomControls component */}
          <div className="flex bg-white/5 rounded-lg border border-white/10 p-0.5 mr-2">
            <span className="px-3 py-[10px] text-xs font-mono text-white/50" style={{ padding: "10px 12px" }}>
              {Math.round(zoomLevel * 100)}%
            </span>
          </div>

          <button
            onClick={() => setShowBoxes(!showBoxes)}
            className={`px-4 py-[10px] rounded-lg border text-xs transition-all cursor-pointer flex items-center gap-1.5 ${
              showBoxes
                ? "bg-[#4ade80]/10 border-[#4ade80] text-[#4ade80]"
                : "border-white/10 hover:border-white/30 text-white/60 hover:text-white"
            }`}
            style={{ padding: "10px 16px" }}
          >
            <span>⬡</span> Bounding Boxes
          </button>
        </div>
      </div>

      {/* Main View Area */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center">
        {/* Navigation Arrows */}
        <button
          onClick={handlePrev}
          disabled={currentIndex === 0}
          className="absolute left-6 z-10 w-12 h-12 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/70 border border-white/10 hover:border-white/30 text-white/80 disabled:opacity-20 disabled:pointer-events-none transition-all cursor-pointer"
        >
          ◀
        </button>
        <button
          onClick={handleNext}
          disabled={currentIndex === pages.length - 1}
          className="absolute right-6 z-10 w-12 h-12 flex items-center justify-center rounded-full bg-black/40 hover:bg-black/70 border border-white/10 hover:border-white/30 text-white/80 disabled:opacity-20 disabled:pointer-events-none transition-all cursor-pointer"
        >
          ▶
        </button>

        {/* Zoom controls floating panel */}
        <div className="absolute top-20 right-6 z-20">
          <ZoomControls
            zoomLevel={zoomLevel}
            zoomIn={zoomIn}
            zoomOut={zoomOut}
            reset={reset}
          />
        </div>

        {/* ─── 1. Single Image Mode (Original or Translated) ─── */}
        {(viewMode === "input" || viewMode === "output") && (
          <ZoomPanViewport
            ref={zoomPanRef}
            zoomLevel={zoomLevel}
            setZoomLevel={setZoomLevel}
            resetKey={`${viewMode}-${currentIndex}`}
          >
            <div className="relative group">
              <img
                src={viewMode === "input" ? inputImgUrl : outputImgUrl}
                alt="Manga page"
                className="max-h-[80vh] max-w-[80vw] object-contain rounded-lg border border-white/10 shadow-2xl"
                draggable={false}
              />
              {/* Overlay Bounding Boxes */}
              {showBoxes && (
                <BoundingBoxOverlay
                  pageDetails={pageDetails}
                  onRegionHover={setActiveRegion}
                />
              )}
            </div>
          </ZoomPanViewport>
        )}

        {/* ─── 2. Side-by-Side Mode ─── */}
        {viewMode === "side-by-side" && (
          <ZoomPanViewport
            ref={zoomPanRef}
            zoomLevel={zoomLevel}
            setZoomLevel={setZoomLevel}
            resetKey={`sidebyside-${currentIndex}`}
            minScale={0.3}
          >
            <div className="flex gap-4 items-center">
              <div className="relative">
                <div className="absolute top-3 left-3 px-2 py-0.5 bg-black/60 text-xs rounded border border-white/10 z-10">
                  Original
                </div>
                <img
                  src={inputImgUrl}
                  alt="Original"
                  className="max-h-[75vh] max-w-[40vw] object-contain rounded-lg border border-white/10 shadow-2xl"
                  draggable={false}
                />
              </div>
              <div className="relative">
                <div className="absolute top-3 left-3 px-2 py-0.5 bg-black/60 text-xs rounded border border-white/10 z-10">
                  Translated
                </div>
                <img
                  src={outputImgUrl}
                  alt="Translated"
                  className="max-h-[75vh] max-w-[40vw] object-contain rounded-lg border border-white/10 shadow-2xl"
                  draggable={false}
                />
              </div>
            </div>
          </ZoomPanViewport>
        )}

        {/* ─── 3. Interactive Swipe Mode (Web Component) ─── */}
        {viewMode === "swipe" && (
          <ZoomPanViewport
            ref={zoomPanRef}
            zoomLevel={zoomLevel}
            setZoomLevel={setZoomLevel}
            resetKey={`swipe-${currentIndex}`}
          >
            <div className="relative">
              {/* @ts-expect-error — img-comparison-slider is a web component */}
              <img-comparison-slider
                ref={sliderRef}
                style={{
                  "--divider-color": "#4ade80",
                  "--divider-width": "4px",
                  pointerEvents: "none",
                } as React.CSSProperties}
                className="rounded-lg border border-white/10 shadow-2xl"
              >
                <img
                  slot="first"
                  src={inputImgUrl}
                  alt="Original"
                  className="max-h-[80vh] max-w-[80vw] object-contain"
                  draggable={false}
                />
                <img
                  slot="second"
                  src={outputImgUrl}
                  alt="Translated"
                  className="max-h-[80vh] max-w-[80vw] object-contain"
                  draggable={false}
                />

                {/* Custom Thick Handle Slot (has pointer-events: auto) */}
                <div
                  slot="handle"
                  style={{ pointerEvents: "auto" }}
                  onPointerDown={handleSliderPointerDown}
                  className="w-12 h-full bg-[#4ade80]/15 hover:bg-[#4ade80]/30 backdrop-blur-xs border-x-2 border-[#4ade80] flex items-center justify-center cursor-ew-resize transition-colors duration-200 z-30"
                >
                  <div className="w-2 h-16 bg-[#4ade80] rounded-full shadow-[0_0_12px_rgba(74,222,128,0.8)]" />
                </div>
              {/* @ts-expect-error */}
              </img-comparison-slider>

              {/* Bounding Box Overlays on top */}
              {showBoxes && (
                <BoundingBoxOverlay
                  pageDetails={pageDetails}
                  onRegionHover={setActiveRegion}
                />
              )}
            </div>
          </ZoomPanViewport>
        )}
      </div>

      {/* Region details floating panel (active region tooltip) */}
      {activeRegion && (
        <div className="absolute bottom-6 left-6 z-20 max-w-sm p-4 bg-[#0a0c0f]/90 border border-white/10 rounded-lg shadow-2xl backdrop-blur-md animate-fade-in pointer-events-none">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">
              Region {activeRegion.id}
            </span>
            <span className={`badge ${activeRegion.needs_review ? "badge-amber" : "badge-green"}`}>
              {activeRegion.needs_review ? "Needs Review" : "OK"}
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <div>
              <span className="text-white/40 block mb-0.5">Original (JA):</span>
              <p className="font-sans text-[#a3e4a3] bg-white/5 p-1.5 rounded">
                {activeRegion.raw_text || "(empty)"}
              </p>
            </div>
            <div>
              <span className="text-white/40 block mb-0.5">Translation (EN):</span>
              <p className="font-sans text-white bg-white/5 p-1.5 rounded font-medium">
                {activeRegion.translated_text || "(empty)"}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
