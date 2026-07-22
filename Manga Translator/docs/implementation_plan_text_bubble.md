# Deterministic Tone Scoring Engine — Resolved Implementation Plan (v2)

Supersedes the draft plan. Both open questions are resolved below.

## Context & Problem
Currently, the TypesetterWorker applies a single font to all translated text, with only basic formatting (ALL CAPS, white stroke outline). Manga heavily relies on typography to convey tone (shouting, thinking, whispering, narration). Previous if/elif proposals based solely on text patterns create brittle rules where real dialogue triggers multiple branches simultaneously.

## Proposed Architecture (Resolved V2)
Implement a zero-ML, zero-VRAM, pure-Python deterministic scoring engine that evaluates multiple signals (bubble shape, punctuation, case, geometry) and uses a weighted voting system to assign the optimal font variant.

1. **Bubble Shape Classifier (`quality/bubble_shape.py`)**
   - Implements `detect_bubble_contour(page_image, bbox, padding=15)` to dynamically crop and threshold the inpainted page image (detecting the bubble contour).
   - Classifies bubble shapes into categories: `RECTANGLE`, `OVAL`, `SPIKY`, `CLOUD`, or `UNKNOWN` using contour math.

2. **Tone Scoring Engine (`quality/tone_engine.py`)**
   - Evaluates weighted signals: bubble shape, punctuation patterns, text case ratios, and bounding box geometry.
   - Assigns tone using `argmax` selection (`SHOUT`, `THOUGHT`, `WHISPER`, `NARRATOR`, `STANDARD`).
   - Handles `color_for_speaker` mapping logic, defaulting to fallback if `speaker_id` is empty.

3. **Smart Word Wrap**
   - Replaces fixed word-count wrapping with a width-aware greedy-fit + rebalance algorithm in Stage 10.

4. **Integration into Stage 10**
   - Wires the predicted tone into `TypesetterWorker` to dynamically select fonts via `FONT_MAP`.
   - Optional feature flag `tone_engine_enabled` to toggle shape-based scoring per-project.
   - Serializes `debug_tone_reasons` back to the `TextRegion` manifest. Note: `speaker_id` requires prerequisite item #6 to be built first.

## Resolved Decisions

### Decision 1: Bubble shape detection is computed fresh in Stage 10, not passed from Stage 9
**Do not** add `mask_path` to `TextRegion`. **Do not** serialize masks to a `masks/` directory.

Reason: Stage 9's existing mask is the ink/text-pixel mask for `cv2.INPAINT_TELEA` — it covers the characters being erased, not the bubble's outline. These are different artifacts. Bubble shape classification needs the bubble's *border*, which Stage 9 never computes. Passing Stage 9's mask forward would not solve the problem.

Instead, Stage 10 detects the bubble contour itself, on demand, from a small crop around the region:

```python
# quality/bubble_shape.py

def detect_bubble_contour(page_image: np.ndarray, bbox: BoundingBox, padding: int = 15) -> np.ndarray:
    """
    Crops a small region around the text bbox and isolates the bubble's
    fill area via adaptive thresholding. Use the INPAINTED page image —
    inpainting only removes ink inside the bubble, never the border, and
    the absence of text makes contour detection cleaner.
    """
    x0 = max(0, bbox.x_min - padding)
    y0 = max(0, bbox.y_min - padding)
    x1 = min(page_image.shape[1], bbox.x_max + padding)
    y1 = min(page_image.shape[0], bbox.y_max + padding)
    crop = page_image[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask
```

`classify_bubble_shape(mask)` (from the original spec) consumes this output unchanged — no changes needed there.

Cost: operates on a bbox-sized crop, not the full page. No new disk I/O, no new manifest fields, no Stage 9 changes, no migration needed for existing manifests.

**Safety net:** wrap detection in a try/except — on any failure, return `BubbleShape.UNKNOWN` and let the tone engine fall back to text-only signals (it already does this by design).

### Decision 2: `speaker_id` does not exist yet — it is unbuilt roadmap Item #6, not a populated field
Confirmed against the project state: Stage 8 and the ML Finetuning Translation Memory work do **not** populate any speaker field. Item #6 (pronoun/catchphrase dictionary lookup, Option B) is still planned.

Required order of work:
1. Add `speaker_id: str = ""` to `TextRegion`.
2. Implement the actual resolution logic (Item #6, Option B — pronoun/catchphrase → speaker_id dictionary lookup) as its own task. This is a prerequisite, not part of the tone engine.
3. `color_for_speaker()` must check for empty `speaker_id` and skip custom coloring (fall back to default/manual color from Item #2) rather than hashing an empty string.

```python
def color_for_speaker(speaker_id: str) -> str | None:
    if not speaker_id:
        return None  # caller falls back to default text color
    idx = int(hashlib.md5(speaker_id.encode()).hexdigest(), 16) % len(PALETTE)
    return PALETTE[idx]
```

---

## Proposed Changes (corrected)

### core
#### [MODIFY] `core/models.py`
- Add `debug_tone_reasons: List[str] = field(default_factory=list)` to `TextRegion`.
- Add `speaker_id: str = ""` to `TextRegion`.

### quality
#### [NEW] `quality/bubble_shape.py`
- `detect_bubble_contour(page_image, bbox, padding=15) -> np.ndarray` — new, on-demand crop+threshold (Decision 1).
- `BubbleShape` Enum (`RECTANGLE`, `OVAL`, `SPIKY`, `CLOUD`, `UNKNOWN`).
- `classify_bubble_shape(mask: np.ndarray) -> BubbleShape` — contour math (solidity, circularity, rectangularity, convexity defects), as originally specced.

#### [NEW] `quality/tone_engine.py`
- `Tone` Enum (`SHOUT`, `THOUGHT`, `WHISPER`, `NARRATOR`, `STANDARD`).
- `ToneScore` dataclass.
- `score_tone(region, bubble_shape) -> ToneScore`.
- `FONT_MAP: dict[Tone, str]`.
- `color_for_speaker(speaker_id: str) -> str | None` (Decision 2).

### workers
#### [MODIFY] `workers/typesetter.py`
- Load the page image already in use for rendering (inpainted output).
- Call `detect_bubble_contour()` → `classify_bubble_shape()` → `score_tone()`, wrapped in try/except defaulting to `BubbleShape.UNKNOWN` on failure.
- Map resulting `Tone` to font path via `FONT_MAP`.
- Call `color_for_speaker(region.speaker_id)`; if `None`, use existing default color logic.
- Replace fixed word-count wrap with the width-aware greedy-fit + rebalance algorithm.
- Persist `region.debug_tone_reasons`.

### config (new — recommended addition, not in original draft)
- Add a single feature flag, e.g. `tone_engine_enabled: bool = True` in pipeline config. If a manga's art style consistently confuses the shape classifier, this lets you disable shape-based scoring per-project without touching code — text-only rules still apply.

---

## Verification Plan

### Automated Tests
- `tests/test_bubble_shape.py`:
  - Synthetic masks (perfect rectangle, perfect circle/oval, star/spiky polygon, cloud-bump polygon) → assert correct `BubbleShape`.
  - Synthetic page crop with a drawn bubble → assert `detect_bubble_contour()` + `classify_bubble_shape()` round-trips correctly.
  - Malformed/empty crop → assert graceful `BubbleShape.UNKNOWN`, no exception escapes.
- `tests/test_tone_engine.py`:
  - Each punctuation/case rule fires the intended `Tone` in isolation.
  - **Conflict case** (explicit, since this is the bug the weighted design fixes): whisper-case text + ellipsis inside a `SPIKY` bubble — assert deterministic resolution per the tie-break order, and assert `reasons` reflects both contributing signals.
  - `speaker_id=""` → `color_for_speaker()` returns `None`.
- `tests/test_stage10.py` (update existing):
  - Assert font selection varies by tone across a set of fixture regions.
  - Assert `debug_tone_reasons` is persisted to the manifest.

### Manual Verification
- `python main.py execute --stage typeset --force` on a test page with at least one bubble of each shape (shout/thought/oval/rectangle) — visually confirm correct font per bubble.
- Confirm no Stage 9 changes were required and existing manifests load without migration.

## Execution Guidelines for Next Agent

> [!NOTE]
> This plan is fully finalized and ready for implementation. The user will be executing this in a new chat.

When you begin execution, please adhere to the following rules:
1. **Safety and Rules First**: Ensure you follow all established project rules (e.g., no model retraining, no API calls, zero ML/VRAM footprint for this feature).
2. **Comprehensive Testing**: Implement everything safely with extensive test coverage as defined in the Verification Plan above.
3. **Manual Verification Protocol**: As part of your manual verification, use exactly 5 images from the `input/` folder to test the end-to-end typesetting output. You may also include any synthetic test images of your own to thoroughly validate edge cases (such as testing each bubble shape classifier).
