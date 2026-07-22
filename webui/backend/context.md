# Backend — Context & Architecture

## What Is This?

A **FastAPI bridge server** that sits between the Next.js frontend and the Manga Translator Python pipeline engine. It does NOT contain any AI/ML logic — it delegates all heavy work to the pipeline via subprocess calls to `main.py`.

## How the Code Is Implemented

```
backend/
├── main.py              ← FastAPI app entry point, CORS config, router registration
├── requirements.txt     ← Python deps (fastapi, uvicorn, pillow, python-multipart)
└── routers/
    ├── __init__.py      ← re-exports router modules
    ├── project.py       ← Project manifest reading & directory scanning
    ├── pipeline.py      ← Pipeline execution (subprocess) & SSE streaming
    └── images.py        ← Image file serving & Pillow thumbnail scaling
```

### Entry Point: `main.py`
- Creates the FastAPI app with lifespan handler
- Adds CORS middleware allowing `localhost:3000` (Next.js dev server)
- Mounts three routers under `/api/project`, `/api/pipeline`, `/api/images`
- Health check at `GET /api/health`

### Router: `routers/project.py`
- **`GET /api/project/info`** — Reads `project_manifest.json` from disk, scans `input/` and `output/` directories, returns page counts, phase completion stats, review flags, and directory sizes. This is the primary data source for the Dashboard.
- **`GET /api/project/stages`** — Returns the static 5-stage pipeline definition (detect-text → ocr → translate → inpaint → typeset).
- **`GET /api/project/page-detail`** — Returns the full manifest entry for a single page including all `text_regions` with bounding boxes. Used by the ImageViewer for bounding box overlays.

### Router: `routers/pipeline.py`
- **`POST /api/pipeline/plan`** — Runs `main.py plan` synchronously via `subprocess.run()`. Returns stdout/stderr/returncode.
- **`POST /api/pipeline/execute/stream`** — Runs `main.py execute` asynchronously via `asyncio.create_subprocess_exec()`. Streams stdout line-by-line as **Server-Sent Events** (`text/event-stream`). Emits `event: start`, `data: <line>`, `event: done`, or `event: error`.
- **`GET /api/pipeline/status`** — Quick manifest snapshot: total pages, phase_a/b completion counts and percentages.
- Uses the venv Python at `ENGINE_DIR/venv/Scripts/python.exe` to invoke `main.py`.

### Router: `routers/images.py`
- **`GET /api/images/list`** — Lists sorted image filenames from `input/` or `output/` directory.
- **`GET /api/images/file`** — Serves raw image bytes as `FileResponse` with 1-hour cache header.
- **`GET /api/images/thumb`** — Serves a Pillow-downscaled JPEG thumbnail (max 400px wide, quality 82, 24-hour cache). Falls back to serving the full image if Pillow is not available.
- All paths are sanitized against path traversal attacks via `_safe_image_path()`.

## Core Functionalities

| Functionality | Where | Notes |
|---|---|---|
| Read project manifest | `project.py` → `_load_manifest()` | Reads `project_manifest.json` from disk each request (no caching) |
| Directory scanning | `project.py` → `_scan_dir()` | Counts images, sums file sizes |
| Subprocess execution | `pipeline.py` → `_build_execute_cmd()` | Builds CLI args for `main.py execute` |
| SSE streaming | `pipeline.py` → `event_generator()` | Async generator yielding `data: ` lines |
| Image serving | `images.py` → `get_image()` / `get_thumbnail()` | FileResponse with cache headers |
| CORS | `main.py` | Allows `localhost:3000` |

## What to Check Before Editing

1. **ENGINE_DIR** in `pipeline.py` (line 21) — hardcoded path to the pipeline engine. If the project moves, update this.
2. **CORS origins** in `main.py` (line 30) — only `localhost:3000` is allowed. Add more origins if deploying elsewhere.
3. **Manifest schema** — `project.py` and `pipeline.py` both read `project_manifest.json`. If the schema changes in `core/models.py`, update the field access here.
4. **Venv path** — `pipeline.py` expects `venv/Scripts/python.exe` (Windows). Change for Linux (`venv/bin/python`).
5. **Image extensions** — both `project.py` and `images.py` define `IMAGE_EXTENSIONS` independently. Keep them in sync.

## How to Run

```bash
cd e:\u torrentz\webui\backend
.venv\Scripts\activate        # or: python -m venv .venv && pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs at: `http://localhost:8000/docs`
