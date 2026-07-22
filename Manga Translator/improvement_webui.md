# Manga Translator WebUI — Next.js & Modern React Performance Optimization Plan

This document outlines high-impact architectural changes and library recommendations to resolve the jittery interface and achieve smooth 60fps (or 120fps on ProMotion displays) rendering in the WebUI. It aggressively leverages Next.js App Router superpowers, React 18 concurrent features, and cutting-edge libraries to ensure the application is ultra-lightweight and blindingly fast.

---

## 1. High-Performance Panning & Zooming (GPU Accelerated)

### The Issue
Currently, panning updates React state variables (`pan.x`, `pan.y`, `zoom`) on every pixel of mouse movement, triggering hundreds of complete React virtual DOM reconciliation cycles per second. This causes layout thrashing and severe lag.

### The Next.js / Modern React Solution
**Library:** `react-zoom-pan-pinch` combined with **GSAP (GreenSock Animation Platform)**.
**Next.js Superpower:** **React Server Components (RSC) & Dynamic Imports**

- **Implementation:** 
  - Use `react-zoom-pan-pinch` as it delegates transformations entirely to CSS matrix3d transforms, bypassing the main thread layout engine.
  - Wrap the zoom viewer in `next/dynamic` (`ssr: false`) to ensure this heavy interactive module is lazy-loaded only when the user navigates to the viewer, keeping the initial JavaScript bundle microscopic.
  - If custom gesture tracking or advanced layout animations are preferred, use **GSAP (specifically `GSAP Draggable` and `gsap.quickSetter()`)**. GSAP is extremely lightweight, heavily optimized, and bypasses the React render loop by directly mutating CSS transforms with hardware acceleration, guaranteeing buttery smooth 120fps interactions even with complex timelines.

---

## 2. Snappy Interactive Swipe (Clip-Path Refactoring)

### The Issue
Dragging the swipe bar continuously updates `swipePercent` React state, re-clipping and re-rendering two full-sized image components simultaneously.

### The Next.js / Modern React Solution
**Library:** `react-img-comparison-slider` or **GSAP Draggable** for custom implementations.
**Next.js Superpower:** **Next.js `<Image />` with AVIF & Edge Caching**

- **Implementation:** 
  - Offload the visual slider entirely to `react-img-comparison-slider`, a highly optimized Web Component that uses CSS Grid and avoids the React render loop altogether.
  - If building a custom slider, use **GSAP Draggable** tied to a `clip-path` animation. GSAP can handle the complex interpolation of dragging mechanics effortlessly without a single React re-render.
  - Feed the slider with **Next.js `<Image />` components**. Next.js will automatically detect WebP/AVIF support and serve compressed, cached images from the edge. 
  - **Advanced UX:** Generate `blurDataURL` hashes on the server (using `sharp`) so the Next.js `<Image placeholder="blur" />` instantly shows a blurred preview while the high-res slider images load, eliminating layout shift (CLS).

---

## 3. SSE Log Rendering Optimization (Pipeline Runner)

### The Issue
Streaming log lines from the subprocess forces Next.js to render a rapidly growing list of DOM elements. The browser repaints the entire container on every new log event, choking the CPU.

### The Next.js / Modern React Solution
**Library:** `react-virtuoso` & Web Workers.
**Next.js Superpower:** **React 18 `useTransition` & Server-Sent Events via App Router**

- **Implementation:**
  - **Virtualization:** Use `react-virtuoso` to render only the ~20 log lines currently visible in the DOM. This keeps DOM nodes flat (O(1) memory) regardless of whether there are 10 or 100,000 logs.
  - **Thread Offloading:** Push the websocket/SSE parsing logic into a Web Worker. The main thread only receives pre-formatted JSON arrays.
  - **Non-Blocking UI:** Wrap the log state updates in React 18's `startTransition`. This tells React that log updates are "background tasks," ensuring that if the user clicks a button or swipes while logs are streaming, the user interaction is processed instantly without waiting for the log render.

---

## 4. Image Loading & Network Transfer Optimizations

### The Issue
Loading raw multi-megabyte PNG files directly over the proxy connection blocks the main thread during image decoding.

### The Next.js / Modern React Solution
**Next.js Superpower:** **Native Image Optimization API & Route Prefetching**

- **Implementation:**
  - Completely remove any custom `/api/images/thumb` endpoints. Rely natively on Next.js Image Optimization. By defining `sizes="(max-width: 768px) 100vw, 50vw"`, Next.js will automatically generate, cache, and serve perfectly sized images for the user's specific screen resolution.
  - **Instant Navigation:** Use Next.js `<Link prefetch={true}>` or the `useRouter().prefetch()` API to silently download the Next/Previous manga pages in the background. When the user hits the arrow key, the next page will render in 0ms from the local memory cache.

---

## 5. Event Throttling (Zero-Dependency Approach)

### The Issue
High-frequency events like window resizing or mouse panning fire faster than the screen's refresh rate, causing event queues to back up.

### The Next.js / Modern React Solution
**Library:** None (Native React)
**Next.js Superpower:** **React 18 `useSyncExternalStore` & `useDeferredValue`**

- **Implementation:**
  - Instead of pulling in `lodash-es` for throttling, we will use modern React primitives.
  - Use `useSyncExternalStore` to cleanly subscribe to window dimensions without causing tearing during concurrent renders.
  - Use `useDeferredValue` for any heavy derived state. If the user is dragging a slider quickly, React will automatically "throttle" the expensive updates to keep the animation running at 60fps, and only compute the final state when the CPU has free time. This provides intelligent, hardware-aware throttling without needing manual timeout delays.

---

## 6. Project/Chapter Thumbnail Grid View

### The Feature
Users need a bird's-eye view of all pages in a chapter or project to quickly navigate, check translation progress, or jump to specific pages without linearly clicking "Next".

### The Next.js / Modern React Solution
**Library:** Native CSS Grid + **GSAP** + (Optional: `react-virtuoso` for virtualization).
**Next.js Superpower:** **Next.js `<Image />` automatic thumbnailing**

- **Implementation:**
  - **CSS Grid Layout:** Utilize native CSS Grid (`grid-template-columns: repeat(auto-fill, minmax(200px, 1fr))`) to create a highly responsive thumbnail gallery that scales instantly across device sizes.
  - **Next.js Thumbnailing:** Use Next.js `<Image />` with aggressive downscaling sizes (e.g., `sizes="200px"`) and slightly lowered `quality` (e.g., `quality={60}`). Next.js will generate and cache these micro-thumbnails automatically, allowing 50+ thumbnails to load instantly without freezing the browser or maxing out network bandwidth.
  - **Premium GSAP Entrance:** When the grid mounts, use GSAP's stagger feature (`gsap.from(..., { stagger: 0.05, opacity: 0, y: 20 })`) to create a beautiful, staggered cascading fade-up animation for the thumbnails, giving the application a highly polished, premium feel.
  - **Grid Virtualization:** If the project contains hundreds of pages (e.g., full volumes), utilize `react-virtuoso`'s `VirtuosoGrid` component. This ensures only the thumbnails currently visible in the viewport exist in the DOM, keeping memory footprint drastically low even for massive projects.
