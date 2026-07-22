# Project State Reference
## Current Status
- **Current Active Stage:** Decoupled Backend-Engine Docker Integration Complete ✅
- **Overall Completion:** 100% (with fully containerized services)

## Completed Stages
- Stage 1: Environment & Dependencies ✅
- Stage 2: Core Data Models ✅
- Stage 3: Manifest Engine ✅
- Stage 4: Quality Control & Heuristics ✅
- Stage 5: VRAM Worker & Stage Controller ✅
- Stage 6: Phase A: Text Detection Layer ✅
- Stage 7: Phase A: Cross-Check OCR Engine ✅
- Stage 8: Phase A: Translation Subsystem ✅
- Stage 9: Phase B: Inpainting Engine ✅
- Stage 10: Phase B: Typesetting & CLI Integration ✅
- Stage 11: Decoupled Backend-Engine Docker Integration ✅

## System Dependencies & Environment
- Python version target: 3.10+ (Using 3.10.6)
- Virtual Env: Created (`venv`). Packages installation managed by user.

## Architectural Decisions
- Stage-Batching architecture enforced for VRAM footprint management.
- Double-stage CLI execution layout: `plan` and `execute`.

## The 10-Stage Roadmap
1. Environment & Dependencies (Completed) ✅
2. Core Data Models (Completed) ✅
3. Manifest Engine (Completed) ✅
4. Quality Control & Heuristics (Completed) ✅
5. VRAM Worker & Stage Controller (Completed) ✅
6. Phase A: Text Detection Layer (Completed) ✅
7. Phase A: Cross-Check OCR Engine (Completed) ✅
8. Phase A: Translation Subsystem (Completed) ✅
9. Phase B: Inpainting Engine (Completed) ✅
10. Phase B: Typesetting & CLI Integration (Completed) ✅

## Phase Synchronization Ledger
- **Stage 1 (Environment Setup):** Completed base directory structure, created `requirements.txt` with pinned dependency versions, and initialized python virtual environment.
- **Stage 2 (Core Data Models):** Defined dataclasses (`BoundingBox`, `TextRegion`, `PageManifest`, `ProjectManifest`) in `core/models.py` with full JSON serialization logic.
- **Stage 3 (Manifest Engine):** Implemented the `ManifestEngine` class in `core/manifest.py` to handle loading/saving of project manifests, scanning image files in directory with lazy-loading dimensions (Pillow), CRUD operations on text regions/pages, and crash-safe backup creation. Verified with comprehensive unit tests under `tests/test_manifest.py`.
- **Stage 4 (Quality Control & Heuristics):** Implemented the `quality/` module with six stateless heuristic filters (`BoundingBoxSizeFilter`, `AspectRatioFilter`, `OcrConfidenceFilter`, `TextDensityFilter`, `DuplicateRegionFilter`, `GarbageTextFilter`) in `quality/heuristics.py`, and the orchestrating `QualityChecker` class in `quality/checker.py` which applies the pipeline to PageManifest objects, removes rejected regions in-place, propagates `needs_review` flags, and emits `CheckReport` / `RegionCheckResult` data structures. Verified with 51 unit tests in `tests/test_quality.py` (all passing).
- **Stage 5 (VRAM Worker & Stage Controller):** Created the abstract `BaseWorker` and context-managing `VramManager` (enforcing load/unload pairings and clean VRAM cache flushing) in `workers/`, and implemented the `StageController` in `stages/` to handle iterative page processing, phase completion flag marking, quality checker integration, page-wise crash-safe disk writes, and fault-tolerant batch runs. Verified with mock-based unit tests in `tests/test_stage5.py`.
- **Stage 6 (Phase A: Text Detection Layer):** Implemented `TextDetectorWorker` in `workers/text_detector.py` subclassing `BaseWorker`. Uses `easyocr.Reader(['ja', 'en'], gpu=True)` to detect text bounding boxes; converts EasyOCR quad format `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]` to `BoundingBox(x_min, y_min, x_max, y_max)`; generates sequential region IDs (`r0000`, `r0001`, …); populates `page.text_regions` (replacing any prior data). Integrated via `StageController(..., marks_phase="a")`. Exported from `workers/__init__.py`. Verified with 31 mock-based unit tests in `tests/test_stage6.py` (all passing; total suite 100 tests).
- **Stage 7 (Phase A: Cross-Check OCR Engine):** Implemented the `OcrWorker` in `workers/ocr_worker.py` subclassing `BaseWorker`. Uses `manga-ocr` to re-OCR text regions detected in Stage 6. Crops images per region, overwrites `raw_text` with high-quality transcriptions, and sets `ocr_confidence` sentinel to 1.0. Zero-area boxes and exceptions are handled gracefully. Verified with 12 mock-based unit tests in `tests/test_stage7.py` (all 112 tests passing).
- **Stage 8 (Phase A: Translation Subsystem):** Implemented `TranslationWorker` in `workers/translator.py` subclassing `BaseWorker`. Translates `region.raw_text` (Japanese from MangaOCR) → `region.translated_text` (English) using a dual-backend strategy: argostranslate (offline neural MT, primary) with deep_translator GoogleTranslator as online fallback. Per-region failures are caught and logged; both-fail regions get `translated_text=raw_text` and `needs_review=True`. Integrated via `StageController(..., marks_phase="a")` — this stage marks `phase_a_completed=True`, completing Phase A. Exported from `workers/__init__.py`. Verified with 32 mock-based unit tests in `tests/test_stage8.py` (all 144 tests passing).
- **Stage 9 (Phase B: Inpainting Engine):** Implemented `InpainterWorker` in `workers/inpainter.py` subclassing `BaseWorker`. Erases original Japanese text from page images using OpenCV's `cv2.INPAINT_TELEA`. Handles missing images gracefully and catches per-region drawing exceptions. Saved output crash-safely via temp files. Tested without requiring a GPU using `unittest.mock`. Exported from `workers/__init__.py`. Integrated via `StageController(..., marks_phase="b")`. All tests passing.
- **Stage 10 (Phase B: Typesetting & CLI Integration):** ✅ **COMPLETE + IMPROVED** Implemented `TypesetterWorker` in `workers/typesetter.py`. **Post-stage improvement (2026-06-22):** Rewrote to produce professional manga-scanlation quality output: Bangers font (bundled at `assets/fonts/Bangers-Regular.ttf`), ALL CAPS text, white stroke outline (3 px) around black text for readability, binary-search auto-sizing, character-level word-split fallback. Tests updated to reflect multi-call draw.text behavior (outline strokes). 29/29 tests passing. Font falls back to Impact → Arial Bold → Comic Sans Bold → PIL default.
- **ML Finetuning Pipeline & Integration (Phases 1-10):** ✅ **COMPLETE** Designed, implemented, and integrated a complete Machine Learning training and inference integration layer for the pipeline workers. Features include:
  - **Phase A upgrades (Stage 7 & 8):** Translation Memory lookup (exact/fuzzy matching) bypassing the MT engine, preceding Japanese story context prepending (up to 5 sentences), and Japanese OCR token spelling correction via a shared phonetic normalization layer.
  - **Phase B upgrades (Stage 9 & 10):** Contour-based Japanese ink mask extraction for non-destructive inpainting with heuristic confidence fallback, and Layout Random Forest regressor dynamic fitting upper bounds for spacing, padding, and font size predictions.
  - **Validation:** Added a 64-test integration suite under `tests/integration/` verifying full backward compatibility, with all 330 tests passing cleanly.

## Post-Stage 10 Improvements
- Added support for custom font rendering via `PIL.ImageDraw`.
- Integrated automated font-fallback logic.
- Implemented character-level wrapping to prevent horizontal text clipping in narrow balloons.

## CLI Usage Guide

The pipeline is now fully automated via `main.py`. All stage orchestration is handled through a double-verb interface:

### 1. Plan Phase: Create Project Manifest

```bash
python main.py plan --project-dir /path/to/project [--input-subdir input]
```

Scans the input directory for image files and creates `project_manifest.json`. Run this first to initialize a project.

### 2. Execute Phase: Run Pipeline Stages

**List available stages:**
```bash
python main.py stages
```

**Run a single stage:**
```bash
python main.py execute --stage translate
python main.py execute --stage inpaint
python main.py execute --stage typeset
```

**Run a range of stages:**
```bash
python main.py execute --from ocr --to typeset
python main.py execute --from detect-text --to translate
```

**Run entire pipeline:**
```bash
python main.py execute all
```

**Force re-run (skip already-complete pages):**
```bash
python main.py execute --stage typeset --force
```

### Stage Names (for CLI use):
- `detect-text` - Text detection via EasyOCR (Phase A)
- `ocr` - OCR via MangaOCR (Phase A)
- `translate` - Translation to English (Phase A)
- `inpaint` - Remove original text (Phase B)
- `typeset` - Render English text (Phase B)

### Project Structure:
```
project/
├── input/                    # Source manga pages (input)
├── output/                   # Inpainted + typeset pages (output)
├── project_manifest.json     # Pipeline metadata
└── project_manifest.json.bak # Backup (auto-created)
```

The manifest stores all page metadata and tracks completion state (`phase_a_completed`, `phase_b_completed`). Already-complete pages are skipped on re-runs unless `--force` is used.

### 3. ML Finetuning Pipeline Orchestration

To run the entire machine learning finetuning pipeline (including resolution alignment, dataset extraction, layout model training, and Translation Memory compilation) for a specific model version:

```bash
python scripts/run_finetuning.py --model-version v2 --raw-dir training_data/raw --eng-dir training_data/en --reset-checkpoint
```

Options:
- `--model-version {v1,v2}`: Version of dataset/models to target (default: `v2`).
- `--reset-checkpoint`: Reset previous extraction checkpoints.
- `--no-gpu`: Run easyocr on CPU.
- `--no-manga-ocr`: Skip manga-ocr Japanese transcriptions for debug runs.
- `--tune`: Run hyperparameter tuning (CV) for the layout random forest regressor.

---

### 4. Interactive Web UI Dashboard & Runner
A premium standalone web application has been built outside the main engine directory to wrap the pipeline CLI, manifest engine, and image visualizer.

**Paths:**
- **Backend API Directory:** `e:\u torrentz\webui\backend`
- **Frontend React/Next.js Directory:** `e:\u torrentz\webui\frontend`

**Backend (FastAPI):**
- Configured with CORS middleware to allow requests from the Next.js development port (3000).
- Exposes API endpoints for project scan statistics (`/api/project/info`), single-page manifest details (`/api/project/page-detail`), and pipeline stage execution.
- Real-time logging console utilizes Server-Sent Events (SSE) at `/api/pipeline/execute/stream` to stream `main.py execute` logs line-by-line using asynchronous subprocess communication.
- Implements image serving and thumbnail scaling (`routers/images.py`) with Pillow fallback for client-side loading optimization.

**Frontend (Next.js 16 + TypeScript + Tailwind CSS):**
- Rebuilt with a premium hybrid **Liquid Claymorphism** visual style: deep background base (`#0f111a`) with floating radial cyan (`#06b6d4`) and purple (`#a855f7`) gradient backdrops, translucent glassmorphic sidebars and header (`rgba(28,30,42,0.6)` + blur-3xl), and 3D claymorphic card panels (`#1e2230`) and tactile buttons (utilizing custom raised and sunken shadow properties instead of flat color fills).
- Zustand store with state persistence in `localStorage` for project paths and configurations.
- Integrated **ImageViewer Lightbox** overlay featuring:
  - **Multi-view Toggles:** Switch between Original (JA), Translated (EN), Side-by-Side, and Interactive Swipe views.
  - **Interactive Swipe Slider:** Drag a central green vertical handle to swipe original text into translated text in real-time.
  - **GPU-Accelerated Viewport:** Drag-panning and mouse scroll zooming bind to CSS transforms to run at 60fps.
  - **Pan Axis Lock:** Cycle between Free Pan, Horizontal Only, and Vertical Only constraints to prevent wobbly diagonal movement.
  - **Draggable Handle (✥):** Drag-handle icon in the top-left of the image overlay to slide the viewport cleanly without mouse conflict.
  - **Bounding Box Tooltips:** Overlays detection regions from the manifest detail API, showing raw Japanese and translated English texts on region hover.
  - **Keyboard Navigation:** Left/Right arrows page flipping and Escape-key closure.
  - **High-Performance Optimizations (Completed 2026-06-29):**
    - GPU-accelerated panning and zooming utilizing `react-zoom-pan-pinch` for zero-render, fluid interaction.
    - Zero-overhead swipe slider using `<img-comparison-slider>` Web Component.
    - Virtualized console logs via `react-virtuoso` with updates scheduled via `startTransition`.
    - SSR-optimized window resize detection using `useSyncExternalStore`.
    - Visual thumbnail Grid View (with GSAP entrance stagger and `VirtuosoGrid` fallback) toggling dynamically with Table View.
    - Next.js dynamic code splitting for lightbox/grid dependencies.

## Web UI Zoom + Bounding Box Editor Dashboard (v2) (Completed 2026-07-03) ✅
Implemented a professional interactive bounding box editor, manifest backup systems, and layout model training console:
- **Interactive SVG Canvas Workspace (`CanvasEditor.tsx`):** Displays page images wrapped inside `ZoomPanViewport` with pointer-based drawing of new boxes, dragging/moving, and corner-handle resizing mapped to image coordinates via an aspect-ratio-scaled SVG overlay.
- **Undo/Redo State Management (`usePageRegions.ts`):** Hook handles active page regions, client-side undo/redo stacks (depth 50), and dirty state tracking.
- **Sidebar Region Editor & Backups:** Implemented a "Region" sidebar tab for deleting regions, viewing Japanese OCR texts, editing English translations, and flagging reviews. Added one-click restoring from backups in the "Dirs" tab.
- **FastAPI Backend Integration (`project.py` & `training.py`):** Added endpoint to estimate font sizes using `TypesetterWorker` code on region saves, prune manifest backups (max 10), clear training datasets, and stream model training logs via Server-Sent Events (SSE). *Optimized (2026-07-03):* Enabled font caching in `TypesetterWorker` to speed up binary search, and modified save endpoint to run inpaint/typeset subprocesses in the background asynchronously, reducing API response time from ~14 seconds to under 0.1 seconds and resolving connection hang-ups.
- **CLI Subcommand `finetune` (`main.py`):** Added CLI option to combine manual corrections with layout datasets and retrain the Random Forest layout regressor.
- **Launcher & Automated Tests:** Created `webui/start.ps1` and verification test suite `tests/test_webui_features.py` (100% passing).

## Decoupled Backend & Engine Docker Integration (v3) (Completed 2026-07-03) ✅
Implemented a robust, platform-independent connection between the backend and translation engine using Docker:
- **FastAPI REST Decoupling:** Replaced direct engine Python module imports (`core.*`, `workers.*`) inside `webui/backend` with REST API communication to a standalone FastAPI server (`engine_api.py` on port `8001`) running inside the PyTorch engine container.
- **Subprocess Queue & Serialization:** Added an asynchronous execution task queue (`queue_manager.py`) in the backend to serialize all heavy engine runs, preventing VRAM overflow.
- **WebSocket Logs & Cancellation:** Implemented bidirectional WebSocket logging (`ws_manager.py`) to stream real-time execution logs and allow immediate pipeline cancellation.
- **Single-Backup Manifest Saving:** Modified the backend save API to write JSON manifest updates to `project_manifest.json.tmp` and atomically replace `project_manifest.json`, keeping only a single `project_manifest.json.bak` file.
- **Dockerization:** Containerized all three services (Next.js frontend, FastAPI backend, PyTorch engine) with shared volume mapping to `/projects` and GPU device reservations.