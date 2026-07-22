# Manga Translator — Tone Engine & Roadmap Additions (Implementation Spec)

**Status:** Planning phase — for agent implementation
**Touches:** `workers/typesetter.py` (Stage 10), `core/models.py`, new module `quality/tone_engine.py`, new module `quality/bubble_shape.py`
**Constraint carried over from roadmap:** No model retraining, no API calls, no extra VRAM. Pure Python + OpenCV/PIL math.

---

## 0. Why the original if/elif proposal needs one change

The proposed logic was:

```python
if text.endswith("!!") or text.endswith("!?"):
    font = IMPACT
elif text.startswith("(") or text.startswith("...") or text.endswith("..."):
    font = ANIME_ACE_ITALIC
else:
    font = WILD_WORDS
```

**Problem:** real dialogue triggers multiple branches at once (`"...I'll kill you...!!"` matches both shout and thought conditions), and a chain of `if/elif` resolves ties by *accident of code order*, not by design. There's also no signal from the bubble's actual shape — only from text strings — even though shape is the single strongest tone signal in real manga (jagged/spiky outline = shout, cloud + trailing bubbles = thought, plain rounded = speech, rectangle = narration).

**Fix:** keep every rule you proposed, but make them **vote with weights into a score table**, then pick `argmax`. This is still zero-ML, zero-VRAM, fully deterministic, and — critically — it's now testable and debuggable (you can log *why* a tone was picked).

---

## 1. Deterministic Tone Scoring Engine

### 1.1 Signals used

| # | Signal | Source | Cost |
|---|---|---|---|
| 1 | Bubble contour shape (spiky / cloud / oval / rectangle) | OpenCV on the **clean background mask** (you already extract this in Stage 9's ink-mask logic) | cheap, pure math |
| 2 | Punctuation pattern of `translated_text` | string ops | free |
| 3 | Case pattern (ALL CAPS vs Mixed/lower) | string ops | free |
| 4 | Bounding box aspect ratio + position on page | already in `BoundingBox` | free |
| 5 | Text length / word count | already known | free |

### 1.2 Bubble shape classifier (`quality/bubble_shape.py`)

This is the new piece. It looks at the bubble's **outline contour** (the same white/background blob Stage 9 already finds before inpainting) and classifies it mathematically — no ML model, just geometry that's been used in document/comic-panel analysis for years:

```python
import cv2
import numpy as np
from enum import Enum

class BubbleShape(Enum):
    RECTANGLE = "rectangle"     # narrator caption
    OVAL = "oval"                # standard speech
    SPIKY = "spiky"              # shout / explosion bubble
    CLOUD = "cloud"               # thought bubble
    UNKNOWN = "unknown"           # contour detection failed -> fall back to text-only rules

def classify_bubble_shape(mask: np.ndarray) -> BubbleShape:
    """
    mask: binary image of the bubble's filled background region
    (the same contour data Stage 9's ink-mask extraction already computes).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BubbleShape.UNKNOWN

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 50:
        return BubbleShape.UNKNOWN

    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / hull_area if hull_area > 0 else 1.0          # spikiness measure
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0  # roundness measure

    # rectangle check: low deviation from minAreaRect, near page top/bottom handled separately
    rect = cv2.minAreaRect(contour)
    rect_area = rect[1][0] * rect[1][1]
    rectangularity = area / rect_area if rect_area > 0 else 0

    # Spike counting via convexity defects (true "explosion star" shapes have many deep defects)
    spike_count = 0
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if len(hull_indices) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is not None:
            depths = defects[:, 0, 3] / 256.0
            spike_count = int(np.sum(depths > (0.04 * np.sqrt(area))))  # scale threshold to bubble size

    if rectangularity > 0.92 and solidity > 0.92:
        return BubbleShape.RECTANGLE
    if solidity < 0.80 and spike_count >= 5:
        return BubbleShape.SPIKY
    if 0.80 <= solidity < 0.93 and circularity < 0.75:
        return BubbleShape.CLOUD
    if circularity >= 0.75:
        return BubbleShape.OVAL
    return BubbleShape.UNKNOWN
```

This reuses contour math you're already running in Stage 9 — no new CV pipeline, just a classification step on data you already have in memory.

### 1.3 The scoring table (`quality/tone_engine.py`)

```python
from dataclasses import dataclass
from enum import Enum

class Tone(Enum):
    SHOUT = "shout"
    THOUGHT = "thought"
    WHISPER = "whisper"
    NARRATOR = "narrator"
    STANDARD = "standard"

@dataclass
class ToneScore:
    tone: Tone
    score: float
    reasons: list[str]   # for logging / QA — "why did it pick this"

FONT_MAP = {
    Tone.SHOUT:    "Impact.ttf",
    Tone.THOUGHT:  "AnimeAce-Italic.ttf",
    Tone.WHISPER:  "WildWords-Light.ttf",
    Tone.NARRATOR: "CleanSerif.ttf",
    Tone.STANDARD: "WildWords-Regular.ttf",
}

def score_tone(region, bubble_shape) -> ToneScore:
    text = region.translated_text.strip()
    bbox = region.bbox
    width, height = bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min

    scores = {t: 0.0 for t in Tone}
    reasons: dict[Tone, list[str]] = {t: [] for t in Tone}

    def add(tone, weight, reason):
        scores[tone] += weight
        reasons[tone].append(reason)

    # --- Shape signals (strongest — geometry doesn't lie, punctuation can be noisy OCR) ---
    if bubble_shape == BubbleShape.SPIKY:
        add(Tone.SHOUT, 3.0, "spiky bubble contour")
    elif bubble_shape == BubbleShape.CLOUD:
        add(Tone.THOUGHT, 3.0, "cloud bubble contour")
    elif bubble_shape == BubbleShape.RECTANGLE:
        add(Tone.NARRATOR, 4.0, "rectangular bubble contour")

    # --- Geometry fallback (when contour extraction fails / BubbleShape.UNKNOWN) ---
    if bubble_shape == BubbleShape.UNKNOWN:
        if width > height * 2.2:
            near_edge = bbox.y_min < 0.12 * region.page_height or bbox.y_max > 0.88 * region.page_height
            add(Tone.NARRATOR, 2.5 if near_edge else 1.0, "wide rectangle geometry")

    # --- Punctuation signals ---
    if text.endswith("!!") or text.endswith("!?") or text.count("!") >= 2:
        add(Tone.SHOUT, 2.0, "exclamation punctuation")
    if text.startswith("(") or text.startswith("...") or text.endswith("..."):
        add(Tone.THOUGHT, 2.0, "ellipsis/parenthesis punctuation")

    # --- Case signal ---
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio < 0.5:
            add(Tone.WHISPER, 2.0, "mixed/lowercase text (manga standard is caps)")
        elif upper_ratio > 0.95 and "!" in text:
            add(Tone.SHOUT, 1.0, "all-caps + exclamation")

    # --- Baseline ---
    add(Tone.STANDARD, 0.5, "default baseline")

    # Resolve: highest score wins; explicit tie-break order for reproducibility
    TIE_BREAK = [Tone.NARRATOR, Tone.SHOUT, Tone.THOUGHT, Tone.WHISPER, Tone.STANDARD]
    best = max(TIE_BREAK, key=lambda t: (scores[t], -TIE_BREAK.index(t)))
    return ToneScore(tone=best, score=scores[best], reasons=reasons[best])
```

**Why this is better than the original chain, while keeping its spirit:**
- Every rule you proposed is still here — punctuation, caps, box width/height — nothing is thrown away.
- Conflicting signals (e.g. a whispered ellipsis line in a spiky bubble) get *resolved by total weight*, not by which `if` happened to be checked first.
- `reasons` gives you a debug log per region — when a font looks wrong on a page, you can print `reasons` and immediately see which signal caused it, instead of re-reading the if-chain.
- Adding a new signal later (e.g. tail-direction, color) is just one more `add(...)` line — no restructuring.

### 1.4 Integration point in `TypesetterWorker`

```python
shape = classify_bubble_shape(region.background_mask)   # from Stage 9 mask data
tone_result = score_tone(region, shape)
font_path = FONT_MAP[tone_result.tone]
region.debug_tone_reasons = tone_result.reasons   # optional: persist to manifest for QA
```

### 1.5 Narrow-bubble font variant (your "tall vs wide" point)

Keep this as a *post-selection adjustment*, not a separate tone — it modifies line spacing/font width, not which of the 5 tones was picked:

```python
NARROW_ASPECT_THRESHOLD = 0.6  # width / height

if width / height < NARROW_ASPECT_THRESHOLD:
    line_spacing_multiplier = 0.85
    font_path = font_path.replace("-Regular", "-Condensed")
```

---

## 2. Smart Word Wrap — replace fixed "3–5 words/line" with a real line-breaking algorithm

Fixed word-count-per-line ignores actual rendered pixel width (a 5-word line of "I"/"a"/"go" is much shorter than 5 words of "extraordinary"/"impossible"). Since you already compute font metrics for binary-search auto-sizing in Stage 10, reuse that to do **greedy width-fit with minimum-raggedness balancing** — a known, deterministic algorithm (a simplified Knuth-Plass), not a heuristic guess:

```python
def wrap_text(words, font, draw, max_width):
    # Pass 1: greedy fit
    lines, current = [], []
    for word in words:
        trial = " ".join(current + [word])
        if draw.textlength(trial, font=font) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    # Pass 2: rebalance — if the last line is much shorter than the rest,
    # pull one word back from the previous line (reduces ragged "orphan" lines)
    if len(lines) >= 2:
        last_len = draw.textlength(lines[-1], font=font)
        if last_len < max_width * 0.35:
            prev_words = lines[-2].split()
            if len(prev_words) > 1:
                moved = prev_words.pop()
                lines[-2] = " ".join(prev_words)
                lines[-1] = moved + " " + lines[-1]
    return lines
```

This is still pure math against pixel widths you already measure — no ML, fully deterministic, same output every run.

---

## 3. Other Roadmap Additions (non-LLM, deterministic)

### 3.1 Speaker-consistent color assignment (concrete version of your Item 2 + Item 6 Option B)
Instead of manual tagging or LLM lookup, hash the speaker/character key deterministically into a fixed palette so the same character always gets the same color, with zero lookup tables to maintain by hand:

```python
import hashlib

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

def color_for_speaker(speaker_id: str) -> str:
    idx = int(hashlib.md5(speaker_id.encode()).hexdigest(), 16) % len(PALETTE)
    return PALETTE[idx]
```
Pair this with your existing Translation Memory dictionary lookup (pronoun/catchphrase → speaker_id) from Item 6 Option B — once a speaker_id is resolved, color falls out automatically and stays consistent across the whole book.

### 3.2 Reading-order via XY-cut panel segmentation
For Item 5, instead of generic contour detection for panels, use the **XY-cut algorithm** — a classic, deterministic document-layout segmentation method (recursively split the page along the widest horizontal/vertical whitespace gutter). It's well-suited to manga because panels are gutter-separated rectangles, and unlike generic contour detection it directly produces a reading-order tree:

1. Find the widest pure-whitespace vertical gutter spanning full page height → split left/right (for right-to-left manga, traverse right child first).
2. Recurse into each half, alternating cut direction, until no gutter wider than a threshold remains.
3. Leaves = panels, in correct reading order by construction.

This gives you reading order *and* panel grouping in one deterministic pass, rather than detecting panels and sorting separately.

### 3.3 Tail-direction speaker heuristic (zero-LLM alternative to Item 6 Option A)
If a face/character detector is ever added later, bubble tails point at the speaker — but even without one, the tail's exit-angle from the contour (the narrowest point of the bubble outline, found via the same convex-hull math in §1.2) tells you whether to anchor a speaker-color guess to the nearest text region's left or right neighbor. Cheap, geometric, no model.

### 3.4 SFX/garbage classifier feature additions (strengthens your planned Item 4)
When you train the scikit-learn classifier from `datasetv2.jsonl`, add two cheap geometric features alongside text length / aspect ratio / OCR confidence:
- **Stroke variance**: SFX text is usually hand-drawn into the art with high local pixel-color variance versus bubble text on a flat background — sample mean/variance of pixel intensity inside the bbox from the *original* (pre-inpaint) image.
- **Isolation**: distance from the bbox centroid to the nearest detected bubble contour — SFX sits outside bubbles, dialogue sits inside them. This single feature alone is a strong, free signal.

### 3.5 QA logging for the tone engine
Persist `tone_result.reasons` per region into the manifest (`region.debug_tone_reasons`). When a translator/editor reviews output and a font looks wrong, this turns "why is this shouting?" into a one-line lookup instead of a re-read of the code.

---

## 4. Suggested Implementation Order

1. `quality/bubble_shape.py` — shape classifier (depends only on existing Stage 9 mask data)
2. `quality/tone_engine.py` — scoring engine (depends on #1 + existing `TextRegion` fields)
3. Wire into `workers/typesetter.py` Stage 10, replacing the single-font path
4. Add `debug_tone_reasons` to `core/models.py::TextRegion` + manifest serialization
5. Replace fixed-word-count wrap with the width-aware wrap function (§2) — independent of 1–4, can ship first if you want a quick visible win
6. Speaker color hashing (§3.1) — independent, low effort
7. XY-cut panel segmentation (§3.2) — replaces planned generic contour panel detection in Item 5
8. SFX classifier feature additions (§3.4) — fold into the Item 4 training work already planned

Each step is independently testable with synthetic `TextRegion`/mask fixtures — no GPU, no model files, no API calls required for any of the above.