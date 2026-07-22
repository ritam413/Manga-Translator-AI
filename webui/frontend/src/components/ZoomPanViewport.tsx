"use client";

import { useEffect, useRef, useImperativeHandle, forwardRef, useState } from "react";
import gsap from "gsap";

export interface ZoomPanRef {
  zoomIn: () => void;
  zoomOut: () => void;
  reset: () => void;
}

interface ZoomPanViewportProps {
  children: React.ReactNode;
  zoomLevel: number;
  setZoomLevel: (scale: number) => void;
  resetKey: string;
  className?: string;
  minScale?: number;
  maxScale?: number;
}

export const ZoomPanViewport = forwardRef<ZoomPanRef, ZoomPanViewportProps>(
  (
    {
      children,
      zoomLevel,
      setZoomLevel,
      resetKey,
      className = "",
      minScale = 0.3,
      maxScale = 4,
    },
    ref
  ) => {
    const viewportRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);

    // Keep actual transform state in refs to avoid React state update lag during animation/interaction
    const transform = useRef({ scale: 1, x: 0, y: 0 });
    const isDragging = useRef(false);
    const startMouse = useRef({ x: 0, y: 0 });
    const startTransform = useRef({ x: 0, y: 0 });

    const spacePressed = useRef(false);
    const [isSpaceActive, setIsSpaceActive] = useState(false);

    // Track Space bar for panning activation
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
          // Prevent page scroll
          e.preventDefault();
          if (!spacePressed.current) {
            spacePressed.current = true;
            setIsSpaceActive(true);
          }
        }
      };

      const handleKeyUp = (e: KeyboardEvent) => {
        if (e.code === "Space") {
          spacePressed.current = false;
          setIsSpaceActive(false);
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      window.addEventListener("keyup", handleKeyUp);
      return () => {
        window.removeEventListener("keydown", handleKeyDown);
        window.removeEventListener("keyup", handleKeyUp);
      };
    }, []);

    // Perform zoom towards a specific coordinate (clientX, clientY)
    const animateZoom = (targetScale: number, clientX?: number, clientY?: number) => {
      const viewport = viewportRef.current;
      const content = contentRef.current;
      if (!viewport || !content) return;

      const rect = viewport.getBoundingClientRect();
      const cx = rect.width / 2;
      const cy = rect.height / 2;

      // Default target coordinates to viewport center
      const mouseX = clientX !== undefined ? clientX - rect.left : cx;
      const mouseY = clientY !== undefined ? clientY - rect.top : cy;

      const currentScale = transform.current.scale;
      const newScale = Math.max(minScale, Math.min(maxScale, targetScale));

      if (newScale === currentScale && clientX !== undefined) return;

      // Zoom towards mouse cursor math (with transform-origin: center)
      const localX = (mouseX - cx - transform.current.x) / currentScale;
      const localY = (mouseY - cy - transform.current.y) / currentScale;

      const newX = mouseX - cx - localX * newScale;
      const newY = mouseY - cy - localY * newScale;

      transform.current.scale = newScale;
      transform.current.x = newX;
      transform.current.y = newY;

      setZoomLevel(newScale);

      gsap.to(content, {
        scale: newScale,
        x: newX,
        y: newY,
        duration: 0.18,
        ease: "power2.out",
        overwrite: "auto",
        force3D: false,
      });
    };

    // Public API for parent component
    useImperativeHandle(ref, () => ({
      zoomIn: () => {
        animateZoom(transform.current.scale * 1.3);
      },
      zoomOut: () => {
        animateZoom(transform.current.scale / 1.3);
      },
      reset: () => {
        transform.current.scale = 1;
        transform.current.x = 0;
        transform.current.y = 0;
        setZoomLevel(1);

        gsap.to(contentRef.current, {
          scale: 1,
          x: 0,
          y: 0,
          duration: 0.25,
          ease: "power2.out",
          overwrite: "auto",
          force3D: false,
        });
      },
    }));

    // Reset transform when page or view mode changes
    useEffect(() => {
      transform.current.scale = 1;
      transform.current.x = 0;
      transform.current.y = 0;
      setZoomLevel(1);

      if (contentRef.current) {
        gsap.set(contentRef.current, {
          scale: 1,
          x: 0,
          y: 0,
          force3D: false,
        });
      }
    }, [resetKey, setZoomLevel]);

    const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
      e.preventDefault();
      const zoomFactor = 1.12;
      const isZoomIn = e.deltaY < 0;
      const targetScale = isZoomIn
        ? transform.current.scale * zoomFactor
        : transform.current.scale / zoomFactor;

      animateZoom(targetScale, e.clientX, e.clientY);
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
      // Left click (0) or middle click (1) only
      if (e.button !== 0 && e.button !== 1) return;

      const target = e.target as HTMLElement;
      // Prevent drag initiation on controls or buttons
      if (
        target.closest("button") ||
        target.closest("input") ||
        target.closest("select") ||
        target.closest("a")
      ) {
        return;
      }

      // Check if clicking inside img-comparison-slider
      const isComparisonSlider = target.closest("img-comparison-slider");
      // If clicking comparison slider, only allow panning if spacebar is held down
      if (isComparisonSlider && !spacePressed.current) {
        return;
      }

      isDragging.current = true;
      startMouse.current = { x: e.clientX, y: e.clientY };
      startTransform.current = { x: transform.current.x, y: transform.current.y };
      
      if (viewportRef.current) {
        viewportRef.current.style.cursor = "grabbing";
      }
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
      if (!isDragging.current) return;

      const deltaX = e.clientX - startMouse.current.x;
      const deltaY = e.clientY - startMouse.current.y;

      const newX = startTransform.current.x + deltaX;
      const newY = startTransform.current.y + deltaY;

      transform.current.x = newX;
      transform.current.y = newY;

      gsap.to(contentRef.current, {
        x: newX,
        y: newY,
        duration: 0.05, // slightly smoothed drag for premium momentum feel
        ease: "power1.out",
        overwrite: "auto",
        force3D: false,
      });
    };

    const handleMouseUp = () => {
      if (!isDragging.current) return;
      isDragging.current = false;
      if (viewportRef.current) {
        viewportRef.current.style.cursor = spacePressed.current ? "grab" : "default";
      }
    };

    const handleDoubleClick = (e: React.MouseEvent<HTMLDivElement>) => {
      const target = e.target as HTMLElement;
      if (target.closest("img-comparison-slider")) return; // ignore double click on slider
      
      // If zoomed in, reset. Otherwise, zoom in to cursor
      if (transform.current.scale > 1.05) {
        transform.current.scale = 1;
        transform.current.x = 0;
        transform.current.y = 0;
        setZoomLevel(1);

        gsap.to(contentRef.current, {
          scale: 1,
          x: 0,
          y: 0,
          duration: 0.25,
          ease: "power2.out",
          overwrite: "auto",
          force3D: false,
        });
      } else {
        animateZoom(2, e.clientX, e.clientY);
      }
    };

    return (
      <div
        ref={viewportRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        className={`w-full h-full overflow-hidden select-none relative ${className}`}
        style={{
          cursor: isSpaceActive ? "grab" : "default",
        }}
      >
        <div
          ref={contentRef}
          className="w-full h-full flex items-center justify-center"
          style={{
            transformOrigin: "center center",
          }}
        >
          {children}
        </div>
      </div>
    );
  }
);

ZoomPanViewport.displayName = "ZoomPanViewport";
