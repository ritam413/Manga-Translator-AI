# Manga Translator Full Stack — Getting Started

A full-stack manga translation system with three services:

| Service | Directory | Port | Role |
|---------|-----------|------|------|
| **Translation Engine** | `Manga Translator/` | **8001** | ML pipeline (OCR, translation, inpainting, typesetting) |
| **Backend API** | `webui/backend/` | **8000** | FastAPI gateway (manifests, images, WebSockets, queue) |
| **Frontend** | `webui/frontend/` | **3000** | Next.js dashboard & editor |

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

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** | Engine targets 3.10.6 |
| **Node.js 18+** | For the Next.js frontend |
| **NVIDIA GPU + CUDA** | Recommended for EasyOCR / MangaOCR (CPU fallback possible) |
| **Windows PowerShell** | Project includes `start.ps1` for one-click launch |
| **Docker + NVIDIA Container Toolkit** | Optional, for containerized deployment |

---

## One-Time Setup

### 1. Translation Engine (Python)

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\Manga Translator"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pipeline dependencies
pip install -r requirements.txt
pip install -r requirements-cli.txt

# Optional: ML finetuning tools
pip install -r requirements-finetuning.txt

# Engine API server dependencies (FastAPI + uvicorn)
pip install fastapi uvicorn pydantic httpx
```

### 2. Backend API (Python)

The backend has its own lightweight venv (separate from the engine):

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui\backend"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** `webui/start.ps1` uses the **engine** venv Python for all three services. That works if engine deps include FastAPI/uvicorn. For isolation, start the backend manually with its own `.venv` (see below).

### 3. Frontend (Node.js)

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui\frontend"

npm install
```

---

## Quick Start (All Three Services)

From the `webui` folder:

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui"
.\start.ps1
```

This script:

1. Starts the **Engine API** on port **8001** (if not already running)
2. Starts the **Backend API** on port **8000**
3. Starts the **Next.js frontend** on port **3000**
4. Opens `http://localhost:3000` in your browser

---

## Manual Start (Three Terminals)

Use this when you want separate logs or the backend's own venv.

### Terminal 1 — Translation Engine (port 8001)

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\Manga Translator"
.\venv\Scripts\Activate.ps1
python engine_api.py
```

Health check: http://localhost:8001/api/engine/health

**CLI-only (no web UI):**

```powershell
# Step 1: Create project manifest
python main.py plan --project-dir "C:\path\to\your\manga\project"

# Step 2: Run full pipeline
python main.py execute all --project-dir "C:\path\to\your\manga\project"
```

**Pipeline stages:**

| Stage ID | Name | Description |
|----------|------|-------------|
| `detect-text` | Text Detection | EasyOCR bounding boxes |
| `ocr` | OCR | MangaOCR Japanese transcription |
| `translate` | Translation | ArgosTranslate (offline) or Google (online) |
| `inpaint` | Inpainting | Remove original text (OpenCV) |
| `typeset` | Typesetting | Render English text on pages |

List stages: `python main.py stages`

Useful flags: `--force`, `--limit N`, `--pages page01.png page02.png`, `--model-version v2`

---

### Terminal 2 — Backend API (port 8000)

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui\backend"
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check: http://localhost:8000/api/health  
API docs: http://localhost:8000/docs

**Environment variables (optional):**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGINE_URL` | `http://localhost:8001` | Engine REST API URL |
| `PROJECTS_DIR` | *(hardcoded paths in code)* | Base directory for manga projects (Docker) |

The backend talks to the engine over REST (`/api/engine/plan`, `/api/engine/execute`, `/api/engine/fit-font`, `/api/engine/train`) and uses a task queue to serialize heavy runs.

---

### Terminal 3 — Frontend (port 3000)

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui\frontend"
npm run dev
```

Open: http://localhost:3000

The frontend proxies `/api/*` to the backend via `next.config.ts`:

```typescript
// /api/* → http://localhost:8000/api/*
```

**Production build:**

```powershell
npm run build
npm run start
```

---

## Docker Start (Alternative)

Requires Docker with NVIDIA GPU support.

```powershell
cd "E:\u torrentz\MangaTranslator FUll STACK\webui"
docker compose up --build
```

| Service | Port | Image |
|---------|------|-------|
| frontend | 3000 | Node 18 Alpine |
| backend | 8000 | Python 3.10 slim |
| engine | 8001 | PyTorch 2.1 + CUDA 11.8 |

**Shared volume:** `E:/u torrentz/manga-projects` → `/projects` in backend and engine containers.

Update the volume path in `webui/docker-compose.yml` if your projects live elsewhere.

---

## Manga Project Folder Layout

Before translating, prepare a project directory:

```
your_manga_project/
├── input/                    # Raw Japanese page images (.png, .jpg, etc.)
├── output/                   # Generated after inpaint/typeset (auto-created)
├── project_manifest.json     # Created by `plan` command
└── project_manifest.json.bak # Auto backup on saves
```

In the Web UI **Settings** page, set **Project Directory** to this folder (default in code may differ).

---

## Service URLs Summary

| URL | Service |
|-----|---------|
| http://localhost:3000 | Frontend dashboard |
| http://localhost:8000/docs | Backend Swagger API |
| http://localhost:8001/api/engine/health | Engine health |

---

## Startup Order

1. **Engine** (8001) — must be up before backend pipeline calls
2. **Backend** (8000) — must be up before frontend API calls
3. **Frontend** (3000)

`start.ps1` handles this order and skips services that are already running.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Engine won't start | Activate venv; run `pip install -r requirements.txt`; check CUDA/GPU drivers |
| Backend can't reach engine | Confirm engine on 8001; set `ENGINE_URL=http://localhost:8001` |
| Frontend API errors | Confirm backend on 8000; check `next.config.ts` rewrite target |
| CORS errors | Backend allows `localhost:3000`; use frontend proxy (`/api/...`), not direct `:8000` calls |
| GPU OOM during pipeline | Run one stage at a time: `python main.py execute --stage ocr` |
| Wrong project path in UI | Settings → update Project Directory (stored in `localStorage`) |

---

## Repository Structure

```
MangaTranslator FUll STACK/
├── Manga Translator/          # Translation engine + CLI
│   ├── main.py                # CLI entry (plan / execute)
│   ├── engine_api.py          # REST API wrapper (port 8001)
│   ├── workers/               # Pipeline workers (OCR, translate, etc.)
│   ├── core/                  # Data models & manifest engine
│   ├── docs/                  # Architecture & implementation docs
│   ├── requirements.txt
│   └── venv/
│
└── webui/
    ├── start.ps1              # One-click launcher (Windows)
    ├── docker-compose.yml     # Docker orchestration
    ├── backend/               # FastAPI gateway (port 8000)
    │   ├── main.py
    │   ├── routers/
    │   └── requirements.txt
    └── frontend/              # Next.js app (port 3000)
        ├── src/
        └── package.json
```

---

## Further Reading

- **Engine CLI & finetuning:** `Manga Translator/README.md`
- **Engine architecture:** `Manga Translator/docs/project_context.md`
- **Docker integration plan:** `Manga Translator/docs/implementation_plan_backend_integration.md`
- **Backend API:** `webui/backend/context.md`
- **Frontend architecture:** `webui/frontend/context.md`
