# Backend — Tech Stack

## Core Framework

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Runtime |
| **FastAPI** | latest | REST API framework — async request handling, auto-generated OpenAPI docs |
| **Uvicorn** | latest | ASGI server — runs the FastAPI app with hot-reload in dev |

### FastAPI Features Used
- `APIRouter` — modular router registration (`project`, `pipeline`, `images`)
- `Query(...)` — typed query parameter validation with defaults
- `BaseModel` (Pydantic) — request/response schema validation (`PlanRequest`, `ExecuteRequest`, `ProjectInfo`, `PageStatus`)
- `CORSMiddleware` — cross-origin request handling for Next.js dev server
- `StreamingResponse` — Server-Sent Events (SSE) for real-time pipeline log streaming
- `FileResponse` / `Response` — binary image serving with cache headers
- `HTTPException` — structured error responses with status codes
- `@asynccontextmanager` lifespan — app startup/shutdown hooks

## Image Processing

| Technology | Version | Purpose |
|---|---|---|
| **Pillow (PIL)** | 10.2.0 | Server-side thumbnail generation — downscales manga pages to max 400px wide JPEG |

### Pillow Features Used
- `Image.open()` / `Image.resize()` with `LANCZOS` resampling
- `Image.save()` to in-memory `BytesIO` buffer (JPEG, quality 82, optimized)
- `Image.convert("RGB")` — ensures consistent color mode before JPEG encoding

## Subprocess Integration

| Technology | Purpose |
|---|---|
| **asyncio.create_subprocess_exec** | Non-blocking execution of `main.py execute` — streams stdout line-by-line |
| **subprocess.run** | Blocking execution of `main.py plan` — returns full stdout/stderr |

### How It Works
- Backend invokes the pipeline engine's `main.py` using the project venv's Python interpreter
- Builds CLI argument arrays dynamically from the `ExecuteRequest` Pydantic model
- SSE streaming reads async stdout line-by-line, yields as `data: <line>\n\n` events

## Data Format

| Technology | Purpose |
|---|---|
| **JSON** | Reads `project_manifest.json` directly from disk (no ORM, no database) |

## No Database
This backend is **stateless** — all state lives in the filesystem (`project_manifest.json`, image files). There is no database, no Redis, no session management.
