# WebUI Performance Optimization Implementation Plan

This document details the step-by-step implementation of the Next.js, modern React, and GSAP optimizations for the Manga Translator WebUI, as outlined in `improvement_webui.md`.

## User Review Required

> [!IMPORTANT]
> Please review the new dependencies that will be added. Let me know if you have any constraints regarding bundle size or if you prefer a different UI package for any specific component.
> Also, confirm if you want me to start scaffolding these changes immediately or if we should tackle them one component at a time.

## Proposed Changes

### Dependencies

#### [MODIFY] package.json
Add the following high-performance libraries:
- `react-zoom-pan-pinch` (for GPU-accelerated zooming)
- `gsap` (for hardware-accelerated animations and staggered entries)
- `react-img-comparison-slider` (for the before/after swipe view)
- `react-virtuoso` (for SSE log virtualization and Grid view)
- `sharp` (for Next.js image optimization and blur placeholder generation)

Remove the following legacy dependencies (if present):
- `lodash-es` (replaced by native React `useDeferredValue`)

---

### Core Components

#### [MODIFY] WebUI Main Viewer (Panning & Zooming)
- Wrap the main image viewing area with `TransformWrapper` and `TransformComponent` from `react-zoom-pan-pinch`.
- Implement `next/dynamic` (`ssr: false`) to ensure this interactive component only loads on the client side, keeping the initial HTML payload minimal.
- Wire up GSAP `quickSetter()` if any custom follow-cursor or floating UI elements are tied to the pan state.

#### [MODIFY] Before/After Swipe Component
- Replace the current custom `clip-path` state logic with `<ImgComparisonSlider>` from `react-img-comparison-slider`.
- Ensure both the "before" (raw Japanese) and "after" (inpainted/typeset English) images are served via Next.js `<Image />` tags.
- Implement server-side generated `blurDataURL` for instant placeholder rendering (`placeholder="blur"`).

#### [MODIFY] Pipeline Logs View (SSE Streaming)
- Replace the raw `.map()` log rendering with `Virtuoso` from `react-virtuoso`.
- Configure `Virtuoso` to automatically follow the bottom of the list as new logs stream in.
- Move the `EventSource` message parsing into a separate Web Worker file.
- Wrap the React state dispatch for appending logs in React 18's `startTransition` to prevent UI freezing during massive log dumps.

#### [NEW] Thumbnail Grid View
- Create a new component `ChapterGridView.tsx`.
- Implement a responsive CSS Grid: `grid-template-columns: repeat(auto-fill, minmax(200px, 1fr))`.
- Map over page manifests and render Next.js `<Image />` elements for each page with aggressive downscaling (`sizes="200px"`).
- Implement a `useEffect` hook with GSAP: `gsap.from(".thumbnail-card", { y: 30, opacity: 0, stagger: 0.05, duration: 0.4, ease: "power2.out" })` to animate the grid when opened.
- Wrap the grid in `VirtuosoGrid` if the page count exceeds 50 to maintain a flat DOM node count.

---

## Verification Plan

### Automated Tests
- Run `npm run build` to ensure Next.js RSC and Client Component boundaries are correctly respected (e.g., ensuring `useTransition` and GSAP are only used in `"use client"` files).
- Run existing unit/integration tests to ensure no core pipeline behavior is broken.

### Manual Verification
- **Performance Profiling:** Use Chrome DevTools (Performance tab) to verify that panning, zooming, and swiping maintain a steady 60fps and that rendering is offloaded to the GPU (no "Layout Thrashing" warnings).
- **Log Streaming:** Run a full 10-stage pipeline execution on a large chapter and ensure the UI remains fully responsive while logs stream.
- **Grid View:** Open a chapter with 40+ pages and visually verify the GSAP staggered animation and that memory usage remains stable via virtualization.
