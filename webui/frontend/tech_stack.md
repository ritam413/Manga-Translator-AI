# Frontend — Tech Stack

## Core Framework

| Technology | Version | Purpose |
|---|---|---|
| **Next.js** | 16.2.9 | React framework with App Router, file-based routing, API rewrites |
| **React** | 19.2.4 | UI library — hooks-based component architecture |
| **TypeScript** | ^5 | Type safety across all components, API calls, and store |

### Next.js Features Used
- **App Router** — `app/` directory with `layout.tsx` + `page.tsx` convention
- **`next/link`** — client-side navigation with prefetching (Sidebar nav)
- **`next/navigation`** — `usePathname()` for active nav highlighting
- **`next.config.ts` rewrites** — proxies `/api/*` to FastAPI backend at `localhost:8000`
- **`"use client"` directive** — all pages are Client Components (hooks-heavy)

### React Features Used
- **Hooks**: `useState`, `useEffect`, `useCallback`, `useRef`
- **Event handling**: `MouseEvent`, `WheelEvent`, `KeyboardEvent` for ImageViewer interactions

---

## Styling

| Technology | Version | Purpose |
|---|---|---|
| **Tailwind CSS** | ^4 | Utility-first CSS framework for layout and spacing |
| **@tailwindcss/postcss** | ^4 | PostCSS integration for Tailwind processing |
| **tw-animate-css** | ^1.4.0 | Tailwind animation utilities |
| **Custom CSS** (`globals.css`) | — | Full design system: CSS variables, card/badge/button components, animations |

### Tailwind Features Used
- Utility classes for flexbox, grid, spacing, borders, opacity, hover states
- `@import "tailwindcss"` — Tailwind v4 CSS-first configuration
- `@import "tw-animate-css"` — animation utility classes
- `@import "shadcn/tailwind.css"` — shadcn component styles
- `@custom-variant dark` — dark mode variant
- `@theme inline` — CSS variable mapping for shadcn compatibility

### Custom Design System (`globals.css`)
- **CSS Variables**: `--bg-base`, `--accent`, `--text-primary`, etc. — dark theme with neon green accent
- **Component Classes**: `.card`, `.badge`, `.btn`, `.progress-track`, `.log-console`, `.data-table`
- **Animations**: `@keyframes fade-in`, `@keyframes shimmer` (skeleton loading), `@keyframes pulse-glow`, `@keyframes spin`

---

## UI Components

| Technology | Version | Purpose |
|---|---|---|
| **shadcn/UI** | ^4.12.0 | Pre-built accessible component primitives |
| **Radix UI** | ^1.6.0 | Headless UI primitives (foundation for shadcn) |
| **Lucide React** | ^1.22.0 | Icon library |
| **class-variance-authority** | ^0.7.1 | Variant-based component styling (used by shadcn) |
| **clsx** | ^2.1.1 | Conditional class name joining |
| **tailwind-merge** | ^3.6.0 | Intelligent Tailwind class deduplication (used in `cn()` helper) |

### shadcn Features Used
- Component primitives in `components/ui/` directory
- `cn()` utility combining `clsx` + `tailwind-merge` for class merging

---

## State Management

| Technology | Version | Purpose |
|---|---|---|
| **Zustand** | ^5.0.14 | Lightweight global state store |

### Zustand Features Used
- `create()` — store creation with typed interface
- `persist()` middleware — auto-saves to `localStorage` under key `manga-translator-project`
- Stores: `projectDir`, `inputSubdir`, `modelVersion` with setter functions

---

## Data Fetching

| Technology | Purpose |
|---|---|
| **fetch API** | All REST calls to backend via `/api/*` proxy |
| **ReadableStream** | SSE streaming for pipeline execution logs (POST-based, not EventSource) |

### API Architecture
- All API functions centralized in `lib/api.ts`
- Image URLs built via `imageUrl()` / `thumbUrl()` helper functions
- SSE streaming via `fetch()` + `reader.read()` loop with `TextDecoder`

---

## Build & Dev Tools

| Technology | Version | Purpose |
|---|---|---|
| **ESLint** | ^9 | Code linting |
| **eslint-config-next** | 16.2.9 | Next.js-specific lint rules |
| **PostCSS** | — | CSS processing pipeline (Tailwind integration) |
