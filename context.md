# Manga Translator Full Stack — System Context

## Project Purpose
Manga Translator Full Stack is an end-to-end, multi-tier manga translation and typesetting platform. It automates Japanese-to-English translation workflows: detecting speech bubbles and vertical text, transcribing via specialized OCR, translating contextual dialog, inpainting backgrounds to scrub Japanese text, and rendering high-quality formatted English text in speech balloons. It provides both an autonomous CLI batch pipeline and an interactive Next.js web application with a live visual editor.

## Architecture & System Overview
The application consists of three decoupled communicating services:

1. **Translation Engine (`Manga Translator/` — Port 8001)**
   - Core ML processing pipeline executing modular stages:
     - `detect-text`: Text detection & bounding box discovery via EasyOCR.
     - `ocr`: Japanese text transcription via MangaOCR.
     - `translate`: Local LLM batch translation via Ollama (`qwen2.5-3b-instruct-abliterated`), cross-page rolling dialogue buffer, Translation Memory glossary injection, ArgosTranslate (offline), and GoogleTranslator fallback.
     - `inpaint`: Background reconstruction using OpenCV / inpainting methods.
     - `typeset`: Font sizing, wrapping, alignment, and rendering English text into balloons.
   - Exposes REST endpoints via FastAPI in `engine_api.py`.
   - Manifest-driven orchestrator in `main.py` enabling resumable, stage-by-stage or batch operations.

2. **Backend API Gateway (`webui/backend/` — Port 8000)**
   - FastAPI gateway server managing project manifests, file uploads, translation queues, and live execution.
   - Proxies engine requests, monitors status, serves raw and processed image assets, and handles WebSocket updates.
   - Structured around modular routers (`manifest.py`, `images.py`, `engine.py`, `settings.py`).

3. **Frontend Web UI (`webui/frontend/` — Port 3000)**
   - Next.js 15+ (React 19, TypeScript, Tailwind CSS) dashboard.
   - Provides split-pane comparison view, interactive balloon editor (canvas/SVG overlay), batch queue control, settings manager, and translation comparison.

```
Browser (localhost:3000)
        │
        ▼
   Frontend (Next.js)  ──proxy /api/*──►  Backend (FastAPI :8000)
                                                │
                                                ▼ REST
                                         Engine API (:8001)
                                                │
                                                ▼
                                    main.py pipeline (CLI)
```

## Important Directories & Modules
- `/Manga Translator`:
  - `core/`: Core abstractions, config, schemas, pipeline state management.
  - `workers/`: Specific processing workers (text detector, OCR reader, translator, inpainter, typesetter).
  - `stages/`: Manifest stage definitions and pipeline orchestration.
  - `cli/`: Command-line interface subcommands.
  - `docs/`: Comprehensive phased design and architecture documents.
  - `tests/`: Automated unit and integration test suite.
  - `engine_api.py`: FastAPI server wrapper around the engine pipeline.
  - `main.py`: CLI pipeline driver.
- `/webui/backend`:
  - `routers/`: API route controllers for projects, manifest, engine interaction.
  - `utils/`: File management, manifest parser, image utilities.
  - `main.py`: FastAPI application entry point.
- `/webui/frontend`:
  - `src/app/`: Next.js App Router pages and layout.
  - `src/components/`: UI components (editor, canvas, inspector, batch queue).
- `/repomix.config.json`: Configuration for Repomix codebase packaging.

## Technology Stack
- **Languages**: Python 3.10+, TypeScript, JavaScript.
- **Backend Frameworks**: FastAPI, Uvicorn, Pydantic.
- **Frontend Frameworks**: Next.js, React, Tailwind CSS, Lucide icons.
- **AI/ML & Vision**: Ollama (`qwen2.5-3b-instruct-abliterated`), EasyOCR, MangaOCR, PyTorch, OpenCV, ArgosTranslate, Transformers.
- **Packaging/Tooling**: Repomix, npm, PowerShell scripts (`start.ps1`).

## Data Flow
1. User creates or uploads a project containing manga image files (`input/`).
2. `main.py plan` or backend API analyzes images and initializes `project_manifest.json`.
3. Engine pipeline executes stages sequentially:
   - Detection finds bounding boxes -> records coordinates to manifest.
   - OCR reads Japanese characters -> records text to manifest.
   - Translation converts Japanese to English -> records translations to manifest.
   - Inpainting generates clean text-free background images (`output/clean/`).
   - Typesetting renders English text on clean images -> final rendered pages (`output/rendered/`).
4. Users review and manually refine translations and bounding boxes in the Frontend visual editor; changes persist to `project_manifest.json`.

## Coding Conventions & Constraints
- Manifest file (`project_manifest.json`) is the single source of truth for page states, blocks, and translations.
- Large binary artifacts (model checkpoints, `.pkl`, `.bin`, `.pt`, `.jsonl` datasets, `.cache`) must remain excluded from git and repository packages.
- Platform is developed for Windows PowerShell environments with optional Docker support.
