# Frontend — Context & Architecture

## What Is This?

A **Next.js 16 App Router** single-page application that provides a premium dark-themed dashboard for the Manga Translator pipeline. It communicates with the FastAPI backend (port 8000) via a Next.js rewrite proxy — all `/api/*` requests are transparently forwarded to `localhost:8000/api/*`.

## How the Code Is Implemented

```
frontend/src/
├── app/
│   ├── layout.tsx           ← Root layout (HTML shell, font imports)
│   ├── globals.css          ← Full design system: tokens, components, animations (~540 lines)
│   ├── page.tsx             ← Dashboard page: stats, progress, pages table, ImageViewer launcher
│   ├── pipeline/
│   │   └── page.tsx         ← Pipeline Runner: stage visualization, run controls, SSE log console
│   ├── pages/
│   │   └── page.tsx         ← Pages list: filterable table, search, ImageViewer launcher
│   └── settings/
│       └── page.tsx         ← Settings: project dir, input subdir, model version
├── components/
│   ├── Header.tsx           ← Fixed top header with page title and project path badge
│   ├── Sidebar.tsx          ← Fixed left sidebar with nav links and backend status
│   ├── ImageViewer.tsx      ← Full-screen lightbox overlay (pan, zoom, swipe, bounding boxes)
│   └── ui/                  ← shadcn/UI components (auto-generated)
├── lib/
│   ├── api.ts               ← All fetch functions + SSE streaming helper + image URL builders
│   ├── types.ts             ← TypeScript interfaces matching backend Pydantic models
│   └── utils.ts             ← cn() helper for Tailwind class merging
└── store/
    └── projectStore.ts      ← Zustand store: projectDir, inputSubdir, modelVersion (persisted to localStorage)
```

## Core Functionalities

### 1. Dashboard (`app/page.tsx`)
- Fetches project info from `/api/project/info` on mount
- Displays 4 stat cards (total pages, phase A/B done, needs review)
- Progress bars for Phase A and Phase B
- Manifest status panel, directory info cards
- Clickable pages table → opens ImageViewer

### 2. Pipeline Runner (`app/pipeline/page.tsx`)
- Visual 5-stage pipeline with animated stage nodes (done/active/idle states)
- Run controls: All Stages / Single / Range mode selectors
- Model version selector (v1/v2), page limit, force re-run checkbox
- **SSE log streaming** via `fetch()` + `ReadableStream` (not EventSource, because we need POST)
- Auto-detects active stage from log lines via regex matching
- Log console with color-coded output (green=success, amber=warning, red=error)

### 3. ImageViewer Lightbox (`components/ImageViewer.tsx`)
- Full-screen overlay (`fixed inset-0 z-50`)
- **4 view modes:** Original (JA), Translated (EN), Side-by-Side, Interactive Swipe
- Pan/zoom via React state (currently — will be replaced with `react-zoom-pan-pinch`)
- Swipe via `clipPath` polygon manipulation (currently — will be replaced with `react-img-comparison-slider`)
- Bounding box overlays from manifest page detail API
- Region hover tooltips showing raw Japanese + English translation
- Keyboard navigation (←→ arrows, Esc to close)
- Pan axis lock (Free / Horizontal / Vertical)

### 4. Pages List (`app/pages/page.tsx`)
- Fetches all pages, supports filter (All / Done / Pending / Review) and filename search
- Full data table with phase status badges
- Click any row → opens ImageViewer

### 5. Settings (`app/settings/page.tsx`)
- Edits Zustand store values (auto-persisted to `localStorage`)
- Project directory path, input subdirectory, model version

### 6. State Management (`store/projectStore.ts`)
- Zustand with `persist` middleware → `localStorage` key: `manga-translator-project`
- Stores: `projectDir`, `inputSubdir`, `modelVersion`
- Default project dir: `e:\u torrentz\Manga Translator`

### 7. API Proxy (`next.config.ts`)
- All `/api/*` routes are rewritten to `http://localhost:8000/api/*`
- This means the frontend never makes cross-origin requests — CORS is handled transparently

## What to Check Before Editing

1. **`globals.css`** — Contains the entire design system (CSS variables, card styles, badges, buttons, progress bars, sidebar, header, tables, animations). Check here first for any visual changes.
2. **`lib/types.ts`** — All TypeScript interfaces must match the backend Pydantic models. If you add a field to a backend response, add it here too.
3. **`lib/api.ts`** — All API calls are centralized here. The `BASE` constant is `/api` (proxied). The `streamExecute()` function uses fetch+ReadableStream for POST SSE.
4. **`store/projectStore.ts`** — The default `projectDir` is hardcoded. Changing the storage key name will reset all users' saved settings.
5. **`next.config.ts`** — The rewrite rule proxies to `localhost:8000`. If the backend port changes, update here.
6. **`ImageViewer.tsx`** — The heaviest component (578 lines). Any interaction changes (pan, zoom, swipe) happen here.
7. **`"use client"` directives** — Every page and component uses `"use client"` because they all use React hooks. If you add a Server Component, do NOT import client components directly.
