# Manga Translator Pipeline — Agent Context & Stage 9 Implementation Guide

> [!IMPORTANT]
> **Backend Docker Integration:** Complete ✅
> Decoupled backend-engine REST APIs, process queues, bidirectional WebSockets, single backup manifest schemas, and Docker Compose configurations are fully implemented.

> **This document is self-contained.** An AI agent can read only this file and
> have everything needed to implement Stage 9 without reading any other file.
> It contains the full project architecture, every existing API surface, all
> design decisions, and a precise specification for what to build next.
>
> **[2026-06-22 UPDATE]** Pipeline is fully complete (Stages 1-10). The most
> recent change was a post-stage improvement to Stage 10's TypesetterWorker:
> - **Font**: Bangers (bundled at `assets/fonts/Bangers-Regular.ttf`) — the
>   standard manga scanlation bold comic font matching reference output images.
> - **ALL CAPS**: All translated text is uppercased automatically.
> - **White stroke outline** (3 px): Rendered around black text for readability
>   on any background (white speech bubbles or full-colour panels).
> - **Auto-fit font sizing**: Binary-search finds the largest size that fills
>   each balloon. Character-level splitting handles single long words.
> - **Test suite**: 185/185 tests passing, 0 regressions.
> - **To run typeset**: `python main.py execute --stage typeset [--force]`


---

## 1. Project Overview

A **10-stage Python pipeline** that translates Japanese manga pages into English.
The pipeline is broken into two high-level phases:

- **Phase A** — Read manga: detect text bubbles → OCR → translate ✅ **COMPLETE**
- **Phase B** — Write manga: inpaint original text → typeset English

All stages share a common manifest file (`project_manifest.json`) which acts as
the single source of truth for every page and its text regions.

### Guiding Architecture Decisions (NEVER violate these)

| Decision | Detail |
|---|---|
| **Stage-Batched execution** | Each heavy AI model (EasyOCR, MangaOCR, Argos) is loaded, used, then explicitly unloaded before the next model runs. No two heavy models coexist in VRAM. |
| **Double-stage CLI** | The CLI has two verbs: `plan` (scans disk → produces manifest) and `execute` (runs a numbered stage against the manifest). |
| **No global state** | Every class receives its dependencies via `__init__`. No module-level singletons. |
| **Crash-safe writes** | All manifest writes use `tempfile` + `os.replace` (atomic rename). |
| **Python 3.10.6, Windows** | The venv is at `./venv`. All paths use `os.path`; no `pathlib` in production code (tests may use it). |

---

## 2. Repository Layout (Current State after Stage 8)

```
Manga Translator/                    ← project root
│
├── requirements.txt                 ← pinned library versions (DO NOT CHANGE)
├── project_context.md               ← project status tracker (update when a stage completes)
├── next_steps.md                    ← THIS FILE
├── library_documentation.md         ← official API reference for all pinned libs
│
├── core/                            ← COMPLETE ✅
│   ├── models.py                    ← dataclasses: BoundingBox, TextRegion, PageManifest, ProjectManifest
│   └── manifest.py                  ← ManifestEngine class
│
├── quality/                         ← COMPLETE ✅
│   ├── __init__.py                  ← re-exports public API
│   ├── heuristics.py                ← six BaseHeuristic filter classes
│   └── checker.py                   ← QualityChecker, CheckReport, RegionCheckResult
│
├── workers/                         ← COMPLETE ✅ (Stages 5, 6, 7, 8)
│   ├── __init__.py                  ← exports: BaseWorker, VramManager, TextDetectorWorker, OcrWorker, TranslationWorker
│   ├── base.py                      ← abstract BaseWorker
│   ├── vram_manager.py              ← VramManager context-manager
│   ├── text_detector.py             ← TextDetectorWorker (Stage 6) ✅
│   ├── ocr_worker.py                ← OcrWorker (Stage 7) ✅
│   └── translator.py                ← TranslationWorker (Stage 8) ✅
│
├── stages/                          ← COMPLETE ✅ (Stage 5)
│   ├── __init__.py                  ← exports: StageController, StageReport, PageResult
│   └── controller.py                ← StageController orchestrator
│
├── tests/
│   ├── __init__.py
│   ├── test_manifest.py             ← Stage 3 tests (5 passing)
│   ├── test_quality.py              ← Stage 4 tests (51 passing)
│   ├── test_stage5.py               ← Stage 5 tests (13 passing)
│   ├── test_stage6.py               ← Stage 6 tests (31 passing) ✅
│   ├── test_stage7.py               ← Stage 7 tests (12 passing) ✅
│   └── test_stage8.py               ← Stage 8 tests (32 passing) ✅
│                                       Total: 144 tests, all passing
│
├── assets/                          ← reserved for fonts, model configs
└── venv/                            ← Python virtual environment
```

---

## 3. Complete API Reference — Existing Code

### 3.1 `core/models.py`

All models are plain `@dataclass`s with JSON round-trip support.

```python
@dataclass
class BoundingBox:
    x_min: int; y_min: int; x_max: int; y_max: int
    # Methods:
    def width() -> int          # x_max - x_min
    def height() -> int         # y_max - y_min
    def area() -> int           # width * height
    def center() -> tuple[int,int]
    @classmethod def from_dict(data: dict) -> 'BoundingBox'

@dataclass
class TextRegion:
    id: str                     # unique per page (e.g. "r0001")
    box: BoundingBox
    raw_text: str = ""          # Japanese OCR output (from MangaOCR)
    translated_text: str = ""   # English translation output (from Stage 8)
    ocr_confidence: float = 0.0 # 0.0–1.0; 1.0 = processed by MangaOCR
    is_vertical: bool = True    # True = tate-gumi (vertical Japanese)
    needs_review: bool = False  # True = flagged for human inspection
    @classmethod def from_dict(data: dict) -> 'TextRegion'

@dataclass
class PageManifest:
    image_filename: str         # e.g. "page01.png" (basename only, no path)
    width: int                  # image width in pixels
    height: int                 # image height in pixels
    text_regions: List[TextRegion] = field(default_factory=list)
    phase_a_completed: bool = False   # ← True after Stage 8 finishes
    phase_b_completed: bool = False   # ← True after Stage 10 finishes
    @classmethod def from_dict(data: dict) -> 'PageManifest'

@dataclass
class ProjectManifest:
    project_name: str
    pages: Dict[str, PageManifest]   # key = image_filename (basename)
    source_language: str = "ja"
    target_language: str = "en"
    def to_dict() -> dict
    def save_to_file(filepath: str) -> None
    @classmethod def from_dict(data: dict) -> 'ProjectManifest'
    @classmethod def load_from_file(filepath: str) -> 'ProjectManifest'
```

### 3.2 `workers/translator.py` — `TranslationWorker` ✅ (Stage 8)

```python
class TranslationWorker(BaseWorker):
    """
    Stage 8 worker. Translates region.raw_text (Japanese) → region.translated_text (English).
    After Stage 8 runs, every page has phase_a_completed=True and all regions have:
      - TextRegion.translated_text = English string from argostranslate or GoogleTranslator
      - TextRegion.needs_review = True only if both backends failed
    """
    def __init__(
        source_language: str = "ja",
        target_language: str = "en",
        prefer_offline: bool = True,    # argostranslate first, then GoogleTranslator
        install_argos_pkg: bool = False # auto-download argos pack if missing
    )
    @property worker_name -> "Translation Worker"
    def load() -> None          # validates backends; raises RuntimeError if none available
    def process_page(page, manifest) -> PageManifest
    # reads region.raw_text, calls argostranslate.translate.translate() or GoogleTranslator
    # writes region.translated_text; sets needs_review=True if both fail
    def unload() -> None        # resets _argos_available / _google_available; idempotent
    def is_loaded() -> bool
```

### 3.3 — All other APIs (unchanged from Stage 7 reference)

Refer to Stages 3.1–3.8 in the previous version of `next_steps.md`, or to the
actual source files. All APIs are stable; Stage 9 adds a new module without
modifying any existing code.

---

## 4. Pinned Library Versions & Key APIs

```
onnxruntime-gpu==1.17.1
manga-ocr==0.1.11
easyocr==1.7.1
deep-translator==1.11.4
argostranslate==1.11.0
opencv-python-headless==4.8.1.78
pillow==10.2.0
```

### OpenCV Headless — used in Stage 9 (inpainting)

```python
import cv2
import numpy as np

# Load image
img = cv2.imread('page01.png')           # BGR uint8 array

# Create a binary mask (white=area to inpaint, black=keep)
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

# Inpaint: fills masked pixels using surrounding context
result = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
# flags options: cv2.INPAINT_TELEA  (fast, good for manga)
#                cv2.INPAINT_NS    (Navier-Stokes, slower)

# Save result
cv2.imwrite('page01_inpainted.png', result)
```

### Pillow — used in Stage 9 (saving output)

```python
from PIL import Image
img = Image.open('page01.png')
img.save('page01_inpainted.png')
```

---

## 5. Stage Completion Status

| Stage | Name | Status | Primary Files |
|---|---|---|---|
| 1 | Environment & Dependencies | ✅ Complete | `requirements.txt`, `venv/` |
| 2 | Core Data Models | ✅ Complete | `core/models.py` |
| 3 | Manifest Engine | ✅ Complete | `core/manifest.py`, `tests/test_manifest.py` |
| 4 | Quality Control & Heuristics | ✅ Complete | `quality/heuristics.py`, `quality/checker.py`, `tests/test_quality.py` |
| 5 | VRAM Worker & Stage Controller | ✅ Complete | `workers/base.py`, `workers/vram_manager.py`, `stages/controller.py` |
| 6 | Phase A: Text Detection Layer | ✅ Complete | `workers/text_detector.py`, `tests/test_stage6.py` |
| 7 | Phase A: Cross-Check OCR Engine | ✅ Complete | `workers/ocr_worker.py`, `tests/test_stage7.py` |
| 8 | Phase A: Translation Subsystem | ✅ Complete | `workers/translator.py`, `tests/test_stage8.py` |
| 9 | Phase B: Inpainting Engine | ✅ Complete | `workers/inpainter.py`, `tests/test_stage9.py` |
| 10 | Phase B: Typesetting & CLI Integration | ✅ Complete | `workers/typesetter.py`, `tests/test_stage10.py` |

---

## 6. Stage 8 — Completed Summary (For Context)

Stage 8 created `TranslationWorker` which:
- Translates `region.raw_text` (Japanese) → `region.translated_text` (English)
- Backend priority: `argostranslate` (offline) → `GoogleTranslator` (online)
- `prefer_offline=True` by default; `prefer_offline=False` reverses order
- Empty `raw_text` regions are skipped silently
- Empty translation results trigger fallback to the next backend
- Regions where both backends fail: `translated_text = raw_text`, `needs_review = True`
- Used with `StageController(..., marks_phase="a")` — the final Phase A stage
- After Stage 8: every page has `phase_a_completed = True`

---

## 7. Stage 9 — Full Specification: Phase B Inpainting Engine

### 7.1 Purpose

Phase B begins here. Stage 9 reads each page image and, for every `TextRegion`
whose bounding box contains original Japanese text, **paints over the Japanese
characters** using OpenCV's `inpaint()` algorithm. The result is a clean
background image (no Japanese text) saved to an `output/` subdirectory — ready
for Stage 10 to typeset English translations on top.

Key design constraints:
- **Input**: page image from `input/` directory (read-only, never modified)
- **Output**: inpainted image saved to `output/<same_filename>` (created if absent)
- **Mask**: per-region binary mask; each `TextRegion.box` becomes a white rectangle
- **Algorithm**: `cv2.INPAINT_TELEA` (fast, suitable for manga halftone patterns)
- **Inpaint radius**: 3 pixels (good default for manga text bubble sizing)
- **Phase flag**: `StageController(..., marks_phase="b")` sets `phase_b_completed`

### 7.2 Files to Create

```
workers/
└── inpainter.py             ← InpainterWorker (new Stage 9 worker)

tests/
└── test_stage9.py           ← unit tests (no GPU — all mocked)
```

Also update:
```
workers/__init__.py           ← add InpainterWorker to exports
```

### 7.3 `workers/inpainter.py` — `InpainterWorker`

```python
import logging
import os
import numpy as np
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from core.models import PageManifest, ProjectManifest
from workers.base import BaseWorker

logger = logging.getLogger("workers.inpainter")


class InpainterWorker(BaseWorker):
    """
    Stage 9 worker — inpaints (erases) the original Japanese text from each
    page image using OpenCV's INPAINT_TELEA algorithm.

    Lifecycle (managed by VramManager / StageController):
        worker.load()                        # validates cv2 importable
        updated_page = worker.process_page(page, manifest)
        worker.unload()                      # releases references (minimal)

    Args:
        input_dir:       Absolute path to input image directory.
        output_dir:      Absolute path to output directory (created if missing).
        inpaint_radius:  Radius used by cv2.inpaint() (default 3).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        inpaint_radius: int = 3,
    ) -> None:
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._inpaint_radius = inpaint_radius
        self._loaded = False

    @property
    def worker_name(self) -> str:
        return "OpenCV Inpainter"

    def load(self) -> None:
        """
        Validate that cv2 is importable. Creates output_dir if needed.
        Idempotent — safe to call multiple times.
        """
        if self._loaded:
            logger.info("[InpainterWorker] Already loaded — skipping.")
            return
        if cv2 is None:
            raise ImportError(
                "opencv-python-headless is not installed. "
                "Run: pip install opencv-python-headless==4.8.1.78"
            )
        os.makedirs(self._output_dir, exist_ok=True)
        self._loaded = True
        logger.info("[InpainterWorker] Ready (output_dir=%s).", self._output_dir)

    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Inpaint all text regions on the page.

        Steps:
          1. Load the source image with cv2.imread() from input_dir.
          2. Build a binary mask: for each TextRegion.box, draw a white
             filled rectangle on a zeros mask (same HxW as image, uint8).
          3. Call cv2.inpaint(img, mask, inpaintRadius, cv2.INPAINT_TELEA).
          4. Write the result to output_dir/<page.image_filename> using
             cv2.imwrite(). Use a tempfile + os.replace for crash safety.
          5. Return the unchanged PageManifest (no mutation).

        Pages with zero text regions are still processed (empty mask → copy).
        Per-region drawing errors are caught and logged; the mask is written
        as-is. cv2 errors from inpaint() or imwrite() propagate up to
        StageController (which handles per-page failures).
        """
        ...

    def unload(self) -> None:
        """Release worker state. Idempotent."""
        if not self._loaded:
            logger.info("[InpainterWorker] Already unloaded — skipping.")
            return
        self._loaded = False
        logger.info("[InpainterWorker] Unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded
```

### 7.4 `process_page()` Implementation Detail

```python
def process_page(
    self,
    page: PageManifest,
    manifest: ProjectManifest,
) -> PageManifest:
    if not self._loaded:
        raise RuntimeError(
            "InpainterWorker.load() must be called before process_page()."
        )

    img_path = os.path.join(self._input_dir, page.image_filename)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(
            f"[InpainterWorker] Image not found: {img_path}"
        )

    logger.info(
        "[InpainterWorker] Inpainting %d region(s) in: %s",
        len(page.text_regions),
        page.image_filename,
    )

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(
            f"[InpainterWorker] cv2.imread failed for: {img_path}"
        )

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in page.text_regions:
        box = region.box
        try:
            # Clamp coordinates to image bounds
            x1 = max(0, box.x_min)
            y1 = max(0, box.y_min)
            x2 = min(w, box.x_max)
            y2 = min(h, box.y_max)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                logger.debug("[InpainterWorker] Masked region %s.", region.id)
        except Exception as exc:
            logger.warning(
                "[InpainterWorker] Failed to mask region %s: %s", region.id, exc
            )

    inpainted = cv2.inpaint(img, mask, self._inpaint_radius, cv2.INPAINT_TELEA)

    # Crash-safe write: write to temp file, then atomic rename
    out_path = os.path.join(self._output_dir, page.image_filename)
    tmp_path = out_path + ".tmp"
    cv2.imwrite(tmp_path, inpainted)
    os.replace(tmp_path, out_path)

    logger.info("[InpainterWorker] Saved inpainted page: %s", out_path)
    return page
```

### 7.5 `workers/__init__.py` — Updated Exports

After Stage 9, the workers package must export:

```python
# workers/__init__.py
from workers.base import BaseWorker
from workers.vram_manager import VramManager
from workers.text_detector import TextDetectorWorker
from workers.ocr_worker import OcrWorker
from workers.translator import TranslationWorker
from workers.inpainter import InpainterWorker

__all__ = [
    "BaseWorker", "VramManager", "TextDetectorWorker",
    "OcrWorker", "TranslationWorker", "InpainterWorker",
]
```

### 7.6 `tests/test_stage9.py` — Test Coverage Requirements

Tests **must not require cv2 installation or real images**. Use `unittest.mock`.

#### Construction & Properties

- `worker_name` returns `"OpenCV Inpainter"`
- `is_loaded()` is `False` before `load()`, `True` after, `False` after `unload()`
- Default `_inpaint_radius` is `3`

#### `load()` behaviour

- `load()` calls `os.makedirs(output_dir, exist_ok=True)` exactly once
- `load()` is idempotent (second call does NOT call `makedirs` again)
- `load()` raises `ImportError` if `cv2 is None`

#### `unload()` behaviour

- `unload()` sets `_loaded = False`
- `unload()` is idempotent (safe to call twice)
- `unload()` before `load()` is safe

#### `process_page()` behaviour

- Raises `RuntimeError` if called before `load()`
- Raises `FileNotFoundError` if image does not exist
- Raises `RuntimeError` if `cv2.imread()` returns `None`
- For each `TextRegion`, calls `cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)`
- Calls `cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)` exactly once
- Writes result to `output_dir/<image_filename>` via temp file + `os.replace`
- Per-region drawing exceptions are caught and logged; inpaint still runs
- Returns the same `PageManifest` object (not mutated)
- Works correctly when page has zero text regions (empty mask)

#### `StageController` integration

- `StageController` with `InpainterWorker` marks `phase_b_completed` on each page
- `worker.load()` called once, `worker.unload()` called once per `run()`
- Pages already with `phase_b_completed=True` are skipped when `force=False`
- `engine.save_manifest()` called once per successfully processed page

---

## 8. Critical Implementation Rules

1. **Import guard for `cv2`** — wrap `import cv2` at module level with
   `try/except ImportError` (same pattern as other workers).

2. **`cv2.imread()` can return `None`** — always check; raise `RuntimeError`
   if the result is None. This happens when the file exists but is corrupt.

3. **Coordinate clamping** — clamp all box coordinates to `[0, w)` and `[0, h)`.
   EasyOCR sometimes returns bounding boxes that extend beyond image boundaries.

4. **Crash-safe write** — write to `<output_path>.tmp` then `os.replace()` to
   the final path. Never write directly to the final path.

5. **No mutation of PageManifest** — Stage 9 only reads from `page.text_regions`;
   it does NOT write to any `TextRegion` fields. The output image goes to disk.

6. **`marks_phase="b"`** — use `StageController(..., marks_phase="b")` to mark
   `phase_b_completed = True` after inpainting.

7. **Logger name**: `workers.inpainter` (follows existing hierarchy).

8. **Output directory**: always created in `load()` via `os.makedirs(..., exist_ok=True)`.

---

## 9. How `InpainterWorker` Fits Into the Full Pipeline

```
After Stage 8 (Phase A complete):
  page.text_regions = [
    TextRegion(id="r0000", raw_text="テスト", translated_text="Test", phase_a_completed=True)
  ]
  page.phase_a_completed = True

Stage 9 (InpainterWorker):
  → reads input/page01.png
  → builds mask covering each region's bounding box
  → calls cv2.inpaint() to erase Japanese text
  → writes output/page01.png (clean background)
  → page.phase_b_completed = True (via StageController marks_phase="b")

Stage 10 (TypesetterWorker, not yet implemented):
  → reads output/page01.png (clean background)
  → reads region.translated_text (English)
  → draws English text inside each bounding box using Pillow
  → saves final typeset image
```

---

## 10. Verification Checklist for Stage 9

After implementation, all of the following must be true:

- [ ] `python -m pytest tests/test_stage9.py -v` → all tests pass
- [ ] `python -m pytest tests/` → all 144 existing tests still pass (no regressions)
- [ ] `workers/__init__.py` exports `InpainterWorker`
- [ ] `InpainterWorker` subclasses `BaseWorker` (can be passed to `StageController`)
- [ ] `output/` directory is created in `load()` if it does not exist
- [ ] `cv2.inpaint()` is called with `cv2.INPAINT_TELEA`
- [ ] Output image is written crash-safely (tempfile + `os.replace`)
- [ ] Per-region mask drawing exceptions are swallowed
- [ ] `PageManifest` is NOT mutated (no `TextRegion` field writes)
- [ ] `project_context.md` updated: Stage 9 marked complete, Stage 10 set as active

---

---

## 11. Post-Stage 9: What Stage 10 Will Do

Stage 10 (**Phase B: Typesetting & CLI Integration**) will:
- Subclass `BaseWorker` as `TypesetterWorker` in `workers/typesetter.py`
- Read `output/<page>` (inpainted background from Stage 9)
- For each region, use Pillow `ImageDraw.Draw.text()` to render `region.translated_text`
- Implement word-wrapping within the bounding box (ImageFont + manual line-break logic)
- Optionally respect `region.is_vertical` to layout text top-to-bottom vs left-to-right
- Save final output image (overwrite `output/<page>` in-place)
- Implement the CLI entry point: `plan` verb → scans disk, `execute` verb → runs numbered stage
- Use `StageController(..., marks_phase="b")` to mark Phase B complete

---

## 12. Next Steps: Web UI Enhancements (Phase 2) ✅ (Completed 2026-07-03)

Following the complete scaffolding and visual image viewer implementation in Phase 1, Phase 2 is fully complete:
- **Interactive Quality Control (QC) Editor:**
  - **Bounding Box Editor:** Allows users to draw new boxes, drag/move them, and resize them with corner handles via an SVG overlay layer in the workspace.
  - **Translation Editor:** Sidebar dedicated tab with original Japanese OCR text reference, translation editing area, and "Needs Review" status toggle.
  - **Save API:** `/save-page-regions` writes changes crash-safely, estimates fitted font sizes using a cached font loader, generates manifest backups, and triggers auto-inpainting and typesetting asynchronously in the background. This ensures instantaneous API responses (< 0.1s) and prevents connection hang-ups.
- **Model Training Integration:**
  - Added training endpoint executing layout Random Forest regressor retraining with SSE streaming logging directly on the dashboard.
- **High-Performance Front-End Optimizations (Completed 2026-06-29):**
  - GPU Viewport Panning: Integrated `react-zoom-pan-pinch`.
  - Zero-Jitter Compare Slider: Integrated `<img-comparison-slider>` Web Component.
  - Log Virtualization: Virtualized logs console using `react-virtuoso`.
  - Optimized hooks & grids.

---

## 13. Decoupled Backend & Engine Docker Integration (Phase 3) ✅ (Completed 2026-07-03)
We have containerized and decoupled the FastAPI backend gateway and PyTorch ML translation engine. All tasks from the integration plan are completed:
- Exposed `/api/engine/plan`, `/api/engine/execute`, `/api/engine/train`, and `/api/engine/fit-font` in the engine container server.
- Built an execution queue manager (`queue_manager.py`) and connection manager (`ws_manager.py`) in the backend.
- Decoupled `project.py`, `pipeline.py`, and `training.py` from direct python imports.
- Created `Dockerfile` and `docker-compose.yml` for multi-container orchestration.

## 14. Premium Web UI Rebuild — Liquid Claymorphism (Completed 2026-07-08) ✅
We have rebuilt the local Next.js frontend UI to use a premium hybrid Liquid Claymorphism style:
- Global layout features floating radial liquid blobs (`#06b6d4` & `#a855f7`) with blur-3xl backdrop filtering.
- Sidebars, headers, and control cards utilize glassmorphic panels (`rgba(28, 30, 42, 0.6)` + blur-16px) and 3D claymorphic slate panels (`#1e2230`).
- Interactive buttons and inputs use tactile raised and sunken shadows.
- Fixed TS compile/optional-chaining type issues and verified build success (`npm run build`).

---
*Last updated: 2026-07-08. All Stage 1-11 features, WebUI Phase 1-2 interactive enhancements, decoupled Docker integration, and Premium Liquid Claymorphic UI rebuild are fully implemented and verified.*
