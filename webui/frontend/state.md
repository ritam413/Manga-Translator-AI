# Frontend — Project State

## Current Status
- **Phase:** Performance Optimization (Complete) ✅
- **Last Updated:** 2026-06-29

---

## ✅ Implemented

### WebUI Performance Optimizations (improvement_webui.md)
- [x] **GPU-Accelerated Pan/Zoom**: Replaced manual react-state panning/zooming with `react-zoom-pan-pinch` utilizing CSS `matrix3d` GPU-accelerated transforms.
- [x] **Web Component Swipe Slider**: Replaced custom `clipPath` state updates with native Web Component `<img-comparison-slider>` for zero React render overhead during swiping.
- [x] **Log Virtualization**: Implemented `react-virtuoso` `<Virtuoso>` list virtualization for pipeline console, keeping visible DOM node count flat at ~30. Wrapped `setLogs` in `startTransition` to prioritize user interactions. Capped history at 2000 lines.
- [x] **Optimized React Hooks**: Created `useWindowSize` hook utilizing `useSyncExternalStore` for tear-free, resize-triggered UI adjustments with zero infinite render risks.
- [x] **Thumbnail Grid View**: Created `ChapterGridView` featuring inline toggle between table list and grid cards, GSAP entry staggering for standard projects, and `VirtuosoGrid` for huge datasets (>100 pages).
- [x] **Dynamic Component Imports**: Wrapped `ImageViewer` and `ChapterGridView` imports with Next.js dynamic imports (`ssr: false`) to keep initial JS payloads lean.

### App Shell
- [x] Root layout with Inter/Outfit font stack
- [x] Full design system in `globals.css` (dark theme, neon green accent `#4ade80`)
- [x] Fixed sidebar with navigation (Dashboard, Pipeline, Pages, Settings)
- [x] Fixed header with page title and project path badge
- [x] Next.js API rewrite proxy to FastAPI backend (`localhost:8000`)

### Dashboard (`/`)
- [x] Project info fetching from `/api/project/info`
- [x] 4 stat cards (Total Pages, Phase A Done, Phase B Done, Needs Review)
- [x] Phase A / Phase B progress bars
- [x] Manifest status panel
- [x] Input/Output directory info cards with file counts and sizes
- [x] Scrollable pages table with phase badges
- [x] Click-to-open ImageViewer from table rows
- [x] Plan/Re-Plan button to trigger manifest creation

### Pipeline Runner (`/pipeline`)
- [x] Visual 5-stage pipeline with animated stage icons (done ✓, active pulse, idle)
- [x] Phase A / Phase B progress bars
- [x] Run mode selector (All Stages / Single / Range)
- [x] Stage dropdowns for single and range modes
- [x] Model version selector (v1 / v2)
- [x] Page limit input
- [x] Force re-run checkbox
- [x] Live SSE log streaming via POST fetch + ReadableStream
- [x] Auto-scrolling log console with color-coded output
- [x] Active stage detection from log content
- [x] Run/Stop toggle button
- [x] Status panel with run state badge

### ImageViewer Lightbox
- [x] Full-screen overlay with backdrop blur
- [x] 4 view modes: Original (JA), Translated (EN), Side-by-Side, Interactive Swipe
- [x] Mouse drag panning with pan axis lock (Free / Horizontal / Vertical)
- [x] Scroll wheel zooming (0.5x – 4x)
- [x] Zoom controls (+, −, reset with percentage display)
- [x] Interactive swipe slider with drag handle
- [x] Bounding box overlays from manifest page-detail API
- [x] Region hover tooltips (Japanese raw text + English translation)
- [x] Keyboard navigation (← → arrows, Esc to close)
- [x] Draggable pan handle (✥)
- [x] Navigation arrows (◀ ▶) with disabled state at boundaries

### Pages List (`/pages`)
- [x] Full pages table with phase A/B badges, region count, review flags
- [x] Filter buttons (All / Done / Pending / Review)
- [x] Filename search input
- [x] Result count display
- [x] Click-to-open ImageViewer

### Settings (`/settings`)
- [x] Project directory input (persisted to localStorage)
- [x] Input subdirectory input
- [x] Model version selector (v1 / v2)
- [x] Backend connection info display

### State Management
- [x] Zustand store with localStorage persistence
- [x] Stores: projectDir, inputSubdir, modelVersion

### API Layer
- [x] `fetchProjectInfo()` — project stats
- [x] `fetchStages()` — pipeline stage definitions
- [x] `fetchPipelineStatus()` — phase completion snapshot
- [x] `planProject()` — trigger manifest creation
- [x] `streamExecute()` — SSE pipeline execution with cleanup function
- [x] `fetchImageList()` — image directory listing
- [x] `imageUrl()` / `thumbUrl()` — URL builders for `<img>` src

---

## 📋 Future / Not Yet Implemented

- [ ] **GSAP page transition animations** — Smooth route transitions between Dashboard → Pipeline → Pages
- [ ] **Image prefetching** — Use `useRouter().prefetch()` to preload next/previous manga pages in ImageViewer
- [ ] **Blur placeholder images** — Generate `blurDataURL` on the backend for instant image preview loading
- [ ] **Dark/Light mode toggle** — Design system supports it via CSS variables, but no toggle UI exists
- [ ] **Responsive mobile layout** — Current layout assumes desktop (fixed sidebar, large tables)
- [ ] **Region editing** — Click a bounding box to edit `translated_text` inline and save back to manifest
- [ ] **Export UI** — Trigger CBZ/ZIP export from the frontend
- [ ] **Progress notifications** — Browser notifications when pipeline stages complete
- [ ] **Multi-project switcher** — Dropdown to switch between different manga project directories

