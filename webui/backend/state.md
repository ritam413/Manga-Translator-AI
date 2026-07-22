# Backend — Project State

## Current Status
- **Phase:** Stable / Feature-complete for v1
- **Last Updated:** 2026-06-29

---

## ✅ Implemented

### Core Server (`main.py`)
- [x] FastAPI app with lifespan hooks
- [x] CORS middleware for `localhost:3000`
- [x] Health check endpoint (`/api/health`)
- [x] Router registration (project, pipeline, images)

### Project Router (`routers/project.py`)
- [x] `GET /api/project/info` — full project stats, directory scan, page list
- [x] `GET /api/project/stages` — static 5-stage pipeline definition
- [x] `GET /api/project/page-detail` — single page manifest with bounding boxes and text regions

### Pipeline Router (`routers/pipeline.py`)
- [x] `POST /api/pipeline/plan` — synchronous manifest creation
- [x] `POST /api/pipeline/execute/stream` — SSE streaming of pipeline execution logs
- [x] `GET /api/pipeline/status` — phase completion snapshot
- [x] Dynamic CLI argument building from request body
- [x] Venv Python detection with fallback to `sys.executable`

### Images Router (`routers/images.py`)
- [x] `GET /api/images/list` — sorted image listing for input/output dirs
- [x] `GET /api/images/file` — raw image serving with cache headers
- [x] `GET /api/images/thumb` — Pillow-downscaled thumbnail serving (max 400px, JPEG q82)
- [x] Path traversal protection
- [x] Pillow fallback (serves original if Pillow unavailable)

---

## 🔄 Currently Working On

### WebUI Performance Optimization (improvement_webui.md)
No backend changes required for the current optimization plan. All changes are frontend-only:
- Pan/zoom → `react-zoom-pan-pinch` (client-side)
- Swipe slider → `react-img-comparison-slider` (client-side)
- Log virtualization → `react-virtuoso` (client-side)
- Thumbnail grid → Uses existing `/api/images/thumb` endpoint (already implemented)

---

## 📋 Future / Not Yet Implemented

- [ ] **WebSocket for logs** — Replace SSE with WebSocket for bidirectional communication (e.g., sending cancel signals)
- [ ] **Manifest caching** — Currently reads `project_manifest.json` from disk on every request. Could add in-memory caching with file-watcher invalidation
- [ ] **Image format conversion** — Serve WebP/AVIF from the thumbnail endpoint for smaller payloads
- [ ] **Pipeline cancellation** — Endpoint to kill a running subprocess (currently frontend just closes the stream, but the subprocess continues)
- [ ] **Multi-project support** — Current architecture already supports any project dir via query params, but no project listing/discovery endpoint exists
- [ ] **Authentication** — No auth layer exists. Not needed for local development but required for any remote deployment
