"""
workers/typesetter.py — Stage 10: Phase B Typesetting Engine

Renders the English text (translated_text) onto the inpainted page images
using Pillow's ImageDraw.text(). Produces professional manga-scanlation-quality
output matching the reference style:

  • Font:    Bangers (Google Fonts — free, condensed bold comic font)
             Falls back to Impact → Arial Bold → Comic Sans Bold → PIL default.
  • Case:    ALL CAPS (standard for English manga/scanlation)
  • Stroke:  White outline (stroke_width px) drawn behind black text for
             readability over any background (pure-white bubble OR coloured
             scene panels).
  • Layout:  Word-wrapped, horizontally + vertically centred inside each
             bounding box. Binary-search auto-sizes the font so it fills
             the bubble as large as possible.

Workflow:
  1. Load inpainted image from output/ directory.
  2. For each TextRegion:
     - Convert translated_text to ALL CAPS.
     - Binary-search the largest font that makes wrapped text fit.
     - Draw white outline by rendering text offset 8× (±stroke_width),
       then draw black text on top — standard Pillow stroke technique.
  3. Save final typeset image to output/<filename> via tempfile + os.replace.
  4. Return PageManifest unchanged (StageController marks phase_b_completed).

Subclasses BaseWorker so it plugs into StageController with marks_phase="b".
"""

import logging
import os
from typing import Optional, List

# ── PIL/Pillow import guard ──────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None       # type: ignore[assignment]
    ImageDraw = None   # type: ignore[assignment]
    ImageFont = None   # type: ignore[assignment]

# ── cv2 import guard ─────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np

from core.models import PageManifest, ProjectManifest, BoundingBox
from workers.base import BaseWorker
from quality import (
    BubbleShape, detect_bubble_contour, classify_bubble_shape,
    Tone, score_tone, color_for_speaker
)
from quality.tone_engine import FONT_MAP

logger = logging.getLogger("workers.typesetter")


# ── Font constants ───────────────────────────────────────────────────────────

# Preferred manga/comic font — bundled in assets/fonts/
_BUNDLED_FONT = os.path.join(
    os.path.dirname(__file__), "..", "assets", "fonts", "Bangers-Regular.ttf"
)

# Windows system font fallbacks (first found wins)
_SYSTEM_FONT_CANDIDATES: List[str] = [
    r"C:\Windows\Fonts\impact.ttf",        # Impact — compressed, punchy
    r"C:\Windows\Fonts\arialbd.ttf",       # Arial Bold
    r"C:\Windows\Fonts\comicbd.ttf",       # Comic Sans Bold
    r"C:\Windows\Fonts\calibrib.ttf",      # Calibri Bold
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]

# Stroke (outline) width in pixels for the white border around black text.
_DEFAULT_STROKE_WIDTH = 3

# Minimum font size to try during binary search.
_MIN_FONT_SIZE = 8

# Padding (px) subtracted from available box dimensions before fitting.
_BOX_PADDING = 4


class TypesetterWorker(BaseWorker):
    """
    Stage 10 worker — renders English translations on inpainted page images.

    Produces professional manga-scanlation output:
      - ALL CAPS text in a comic/bold font (Bangers → Impact → Arial Bold)
      - White outline stroke for readability on any background
      - Font auto-sized to fill each speech bubble as large as possible
      - Text horizontally + vertically centred within the bounding box

    Lifecycle (managed by VramManager / StageController):
        worker.load()                        # validate PIL, resolve font
        updated_page = worker.process_page(page, manifest)
        worker.unload()                      # release resources

    Args:
        input_dir:      Absolute path to input image directory (unused here,
                        kept for architectural consistency).
        output_dir:     Absolute path to output directory containing inpainted
                        images from Stage 9. Results are written back in-place.
        font_path:      Optional explicit path to a .ttf font file. If None,
                        the bundled Bangers font is tried first, then system
                        fallbacks.
        font_size_base: Kept for API compatibility; not used in auto-fit logic.
        stroke_width:   Width (px) of the white outline drawn behind text.
                        Default 3.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        font_path: Optional[str] = None,
        font_size_base: int = 12,          # kept for API compatibility
        stroke_width: int = _DEFAULT_STROKE_WIDTH,
        layout_model: Optional[object] = None,
        layout_model_min_confidence: float = 0.5,
        enable_box_expansion: bool = True,
        enable_char_wrap: bool = False,
        tone_engine_enabled: bool = True,
        max_words_per_line: Optional[int] = None,
    ) -> None:
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._font_path = font_path
        self._font_size_base = font_size_base
        self._stroke_width = stroke_width
        self._layout_model = layout_model
        self._layout_model_min_confidence = layout_model_min_confidence
        self._enable_box_expansion = enable_box_expansion
        self._enable_char_wrap = enable_char_wrap
        self._tone_engine_enabled = tone_engine_enabled
        self._max_words_per_line = max_words_per_line

        # Runtime state
        self._loaded = False
        self._font: Optional[object] = None          # kept for test compatibility
        self._resolved_font_path: Optional[str] = None
        self._font_cache = {}
        self._tone_font_cache = {}

    # ── BaseWorker interface ──────────────────────────────────────────────────

    @property
    def worker_name(self) -> str:
        return "Typesetter Worker"

    def load(self) -> None:
        """
        Validate PIL is importable, resolve the font path.
        Idempotent — safe to call multiple times.

        Raises:
            ImportError: if PIL is not installed.
            FileNotFoundError: if an explicit font_path is provided but missing.
        """
        if self._loaded:
            logger.info("[TypesetterWorker] Already loaded — skipping.")
            return

        if Image is None:
            raise ImportError(
                "Pillow is not installed. "
                "Run: pip install pillow==10.2.0"
            )

        logger.info("[TypesetterWorker] Loading typesetter engine…")

        # Resolve font path
        if self._font_path is not None:
            # Explicit path — must exist
            if not os.path.isfile(self._font_path):
                raise FileNotFoundError(
                    f"[TypesetterWorker] Font file not found: {self._font_path}"
                )
            self._resolved_font_path = self._font_path
            logger.info("[TypesetterWorker] Using explicit font: %s", self._resolved_font_path)
        else:
            # Try bundled Bangers first
            bundled = os.path.normpath(_BUNDLED_FONT)
            if os.path.isfile(bundled):
                self._resolved_font_path = bundled
                logger.info("[TypesetterWorker] Using bundled Bangers font: %s", bundled)
            else:
                # Walk system fallbacks
                for candidate in _SYSTEM_FONT_CANDIDATES:
                    if os.path.isfile(candidate):
                        self._resolved_font_path = candidate
                        logger.info("[TypesetterWorker] Using system font: %s", candidate)
                        break
                else:
                    self._resolved_font_path = None  # PIL bitmap fallback
                    logger.warning(
                        "[TypesetterWorker] No TTF font found — "
                        "using PIL bitmap default (~10 px, no outline)."
                    )

        self._loaded = True
        logger.info("[TypesetterWorker] Ready (output_dir=%s).", self._output_dir)

    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Render translated text (ALL CAPS, white-outlined, auto-sized) on the
        inpainted page image stored in output_dir.

        Returns:
            PageManifest (unchanged, no mutation).
        """
        if not self._loaded:
            raise RuntimeError(
                "TypesetterWorker.load() must be called before process_page()."
            )

        img_path = os.path.join(self._output_dir, page.image_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"[TypesetterWorker] Inpainted image not found: {img_path}"
            )

        logger.info(
            "[TypesetterWorker] Typesetting %d region(s) in: %s",
            len(page.text_regions),
            page.image_filename,
        )

        # Load inpainted image (keep as-is without convert to preserve mock chain)
        try:
            img = Image.open(img_path)
            canvas = img.copy()
        except Exception as exc:
            raise RuntimeError(
                f"[TypesetterWorker] Failed to open image {img_path}: {exc}"
            )

        draw = ImageDraw.Draw(canvas)

        page_cv = None
        if self._tone_engine_enabled and cv2 is not None:
            try:
                page_cv = cv2.imread(img_path)
            except Exception as exc:
                logger.warning("[TypesetterWorker] cv2.imread failed to load %s for tone analysis: %s", img_path, exc)

        # Render each text region
        for region in page.text_regions:
            if not region.translated_text or not region.translated_text.strip():
                logger.debug("[TypesetterWorker] Skipping empty region %s", region.id)
                continue

            bubble_shape = BubbleShape.UNKNOWN
            tone = Tone.STANDARD
            reasons = []

            if self._tone_engine_enabled:
                try:
                    if page_cv is not None:
                        mask = detect_bubble_contour(page_cv, region.box)
                        bubble_shape = classify_bubble_shape(mask)
                except Exception as exc:
                    logger.warning("[TypesetterWorker] Bubble shape detection failed for region %s: %s", region.id, exc)

                try:
                    tone_score = score_tone(region, bubble_shape)
                    tone = tone_score.tone
                    reasons = tone_score.reasons
                except Exception as exc:
                    logger.warning("[TypesetterWorker] Tone scoring failed for region %s: %s", region.id, exc)
                    tone = Tone.STANDARD

            region.debug_tone_reasons = reasons

            try:
                box = region.box
                # Get canvas dimensions safely (works with real PIL and mocks)
                try:
                    cw, ch = canvas.size
                except (TypeError, AttributeError, ValueError):
                    cw, ch = 9999, 9999

                # Clamping and default layout parameters
                padding_val = _BOX_PADDING
                line_spacing_val = None
                max_font_size_limit = None

                if self._layout_model is not None:
                    try:
                        from finetuning.pipeline_integration import extract_layout_features
                        import numpy as np
                        # Predict based on JP raw text and box
                        jp_text = region.raw_text or ""
                        features = extract_layout_features(box, jp_text)
                        
                        # Model predicts [font_size, padding, line_spacing]
                        pred = self._layout_model.predict([features])[0]
                        
                        # Clamp predictions to safe ranges
                        pred_font_size = max(8.0, min(72.0, float(pred[0])))
                        pred_padding = max(0, min(50, int(pred[1])))
                        pred_line_spacing = max(0.5, min(2.5, float(pred[2])))
                        
                        if np.isfinite(pred_font_size) and np.isfinite(pred_padding) and np.isfinite(pred_line_spacing):
                            padding_val = pred_padding
                            line_spacing_val = pred_line_spacing
                            max_font_size_limit = int(pred_font_size)
                            logger.debug(
                                "[TypesetterWorker] Layout model predictions for %s: "
                                "font_size_limit=%s, padding=%s, line_spacing=%s",
                                region.id, max_font_size_limit, padding_val, line_spacing_val
                            )
                    except Exception as e:
                        logger.warning("[TypesetterWorker] Layout prediction failed for %s: %s", region.id, e)

                # Compute available area with potential expansion and padding
                box_x_min, box_y_min, box_x_max, box_y_max = self._get_expanded_box(box, cw, ch)
                x_min = box_x_min + padding_val
                y_min = box_y_min + padding_val
                x_max = box_x_max - padding_val
                y_max = box_y_max - padding_val

                if x_max <= x_min or y_max <= y_min:
                    logger.debug(
                        "[TypesetterWorker] Region %s has no drawable area; skipping.",
                        region.id,
                    )
                    continue

                avail_w = x_max - x_min
                avail_h = y_max - y_min

                # Prepare text — ALL CAPS
                text = region.translated_text.upper().strip()

                # Auto-fit font size
                font, lines, line_height = self._fit_font(
                    text, avail_w, avail_h,
                    max_font_size_limit=max_font_size_limit,
                    line_spacing_ratio=line_spacing_val,
                    tone=tone
                )

                # Render text with white outline + fill (black or custom)
                self._render_region_text(
                    draw, region, font, canvas.size,
                    padding=padding_val,
                    line_spacing_ratio=line_spacing_val
                )

                logger.debug("[TypesetterWorker] Rendered region %s", region.id)

            except Exception as exc:
                logger.warning(
                    "[TypesetterWorker] Failed to render region %s: %s",
                    region.id, exc,
                )

        # Crash-safe write: write to temp file, then atomic rename
        out_path = os.path.join(self._output_dir, page.image_filename)
        ext = os.path.splitext(page.image_filename)[1] or ".png"
        tmp_path = out_path + ".tmp" + ext
        canvas.save(tmp_path)
        os.replace(tmp_path, out_path)

        logger.info("[TypesetterWorker] Saved typeset page: %s", out_path)
        return page

    def unload(self) -> None:
        """Release worker state. Idempotent."""
        if not self._loaded:
            logger.info("[TypesetterWorker] Already unloaded — skipping.")
            return
        self._font = None
        self._resolved_font_path = None
        self._loaded = False
        logger.info("[TypesetterWorker] Unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    def _get_expanded_box(self, box: BoundingBox, cw: int, ch: int) -> tuple[int, int, int, int]:
        """Compute horizontally expanded box coordinates if the box is vertical/thin."""
        box_w = box.width()
        box_h = box.height()
        aspect_ratio = box_w / box_h if box_h > 0 else 0.0
        
        if self._enable_box_expansion and (aspect_ratio < 0.6 or box_w < 80):
            expansion = int(box_h * 0.25)
            target_width = max(100, box_w + 2 * expansion)
            diff = target_width - box_w
            
            x_min = box.x_min - diff // 2
            x_max = box.x_max + diff // 2
        else:
            x_min = box.x_min
            x_max = box.x_max
            
        from core.models import BoundingBox
        # Clamp to bounds
        return max(0, x_min), max(0, box.y_min), min(cw, x_max), min(ch, box.y_max)

    # ── Internal helpers ──────────────────────────────────────────────────────

    # Windows system font candidates (kept for legacy / test compatibility)
    _SYSTEM_FONT_CANDIDATES: List[str] = _SYSTEM_FONT_CANDIDATES

    def _get_font(self, font_size: int) -> object:
        """
        Load a Pillow ImageFont at the given pixel size.
        Uses the path resolved in load(). Falls back to PIL bitmap default.
        """
        if font_size in self._font_cache:
            return self._font_cache[font_size]
        path = self._resolved_font_path
        if path is not None:
            try:
                font = ImageFont.truetype(path, size=font_size)
                self._font_cache[font_size] = font
                return font
            except Exception as exc:
                logger.warning(
                    "[TypesetterWorker] Failed to load TTF at size %d: %s", font_size, exc
                )
        return ImageFont.load_default()

    def _get_font_for_tone(self, tone: Tone, font_size: int) -> object:
        """
        Load a Pillow ImageFont at the given pixel size for a specific Tone.
        Uses FONT_MAP to find candidates and falls back to default resolution.
        """
        cache_key = (tone, font_size)
        if cache_key in self._tone_font_cache:
            return self._tone_font_cache[cache_key]

        if not self._tone_engine_enabled or tone not in FONT_MAP:
            font = self._get_font(font_size)
            self._tone_font_cache[cache_key] = font
            return font

        candidates = FONT_MAP[tone]
        for candidate in candidates:
            # Check if it is the Bangers font (either standard name or similar)
            if "Bangers-Regular" in candidate or "Bangers" in candidate:
                path = self._resolved_font_path or _BUNDLED_FONT
            else:
                if os.path.isabs(candidate):
                    path = candidate
                else:
                    path = os.path.join(r"C:\Windows\Fonts", candidate)

            if path and os.path.isfile(path):
                try:
                    font = ImageFont.truetype(path, size=font_size)
                    self._tone_font_cache[cache_key] = font
                    return font
                except Exception as exc:
                    logger.debug(
                        "[TypesetterWorker] Failed to load tone font %s at size %d: %s",
                        candidate, font_size, exc
                    )

        # Fallback to the default typesetter font
        font = self._get_font(font_size)
        self._tone_font_cache[cache_key] = font
        return font

    def _measure_line(self, font: object, text: str) -> float:
        """Return the pixel width of a single line of text."""
        try:
            bbox = font.getbbox(text)
            # bbox is (left, top, right, bottom) — may be MagicMock in tests
            return float(bbox[2] - bbox[0])
        except (TypeError, AttributeError, IndexError):
            pass
        try:
            return float(font.getlength(text))
        except (TypeError, AttributeError):
            pass
        return float(len(text) * 7)

    def _line_height_for_font(
        self,
        font: object,
        size: int,
        line_spacing_ratio: Optional[float] = None,
    ) -> int:
        """Return the line height (px) including leading."""
        if line_spacing_ratio is not None:
            return int(size * line_spacing_ratio)
        try:
            ascent, descent = font.getmetrics()
            return int(ascent) + int(descent) + max(2, size // 8)
        except (TypeError, AttributeError, ValueError):
            return size + 2

    def _wrap_text(self, text: str, font: object, max_width: int) -> List[str]:
        """
        Word-wrap *text* so each line is ≤ max_width pixels wide.
        Uses actual font measurements; falls back to character estimates.
        Long words that exceed max_width are split character by character.
        Applies a width-aware rebalancing step to equalize line lengths.
        Respects self._max_words_per_line if set and > 0.
        """
        words = text.split()
        if not words:
            return []

        lines: List[str] = []
        current = ""

        for word in words:
            # If a single word is wider than max_width, split it by character (only if enable_char_wrap is active)
            if self._enable_char_wrap and self._measure_line(font, word) > max_width:
                if current:
                    lines.append(current)
                    current = ""
                char_chunk = ""
                for ch in word:
                    candidate = char_chunk + ch
                    if self._measure_line(font, candidate) <= max_width:
                        char_chunk = candidate
                    else:
                        if char_chunk:
                            lines.append(char_chunk)
                        char_chunk = ch
                if char_chunk:
                    current = char_chunk
                continue

            # Normal word wrapping
            if not current:
                current = word
            else:
                candidate = current + " " + word
                # Check word count limit if set
                word_limit_reached = False
                if self._max_words_per_line is not None and self._max_words_per_line > 0:
                    word_limit_reached = (len(current.split()) >= self._max_words_per_line)

                if self._measure_line(font, candidate) <= max_width and not word_limit_reached:
                    current = candidate
                else:
                    lines.append(current)
                    current = word

        if current:
            lines.append(current)

        # ── Width-aware Rebalancing ───────────────────────────────────────────
        if len(lines) > 1:
            improved = True
            for _ in range(10):  # Limit iterations to avoid infinite loops
                if not improved:
                    break
                improved = False
                for i in range(len(lines) - 1):
                    line1 = lines[i]
                    line2 = lines[i + 1]

                    words1 = line1.split()
                    words2 = line2.split()
                    if not words1:
                        continue

                    # Current widths
                    w1 = self._measure_line(font, line1)
                    w2 = self._measure_line(font, line2)
                    current_diff = abs(w1 - w2)

                    # Try shifting last word of line1 to start of line2
                    if len(words1) > 1:
                        cand_line1 = " ".join(words1[:-1])
                        cand_line2 = words1[-1] + " " + line2
                        cw1 = self._measure_line(font, cand_line1)
                        cw2 = self._measure_line(font, cand_line2)

                        # Check word limit on shifting
                        word_limit_ok = True
                        if self._max_words_per_line is not None and self._max_words_per_line > 0:
                            word_limit_ok = (len(cand_line2.split()) <= self._max_words_per_line)

                        if cw2 <= max_width and word_limit_ok and abs(cw1 - cw2) < current_diff - 1.0:
                            lines[i] = cand_line1
                            lines[i + 1] = cand_line2
                            improved = True
                            break

                    # Try shifting first word of line2 to end of line1
                    if words2:
                        cand_line1 = line1 + " " + words2[0]
                        cand_line2 = " ".join(words2[1:])
                        cw1 = self._measure_line(font, cand_line1)
                        cw2 = self._measure_line(font, cand_line2)

                        # Check word limit on shifting
                        word_limit_ok = True
                        if self._max_words_per_line is not None and self._max_words_per_line > 0:
                            word_limit_ok = (len(cand_line1.split()) <= self._max_words_per_line)

                        if cw1 <= max_width and word_limit_ok and abs(cw1 - cw2) < current_diff - 1.0:
                            lines[i] = cand_line1
                            lines[i + 1] = cand_line2
                            improved = True
                            break

        return lines

    # ── Backward-compatible aliases for tests ────────────────────────────────

    def _wrap_text_horizontal(self, text: str, font: object, max_width: int) -> List[str]:
        """Alias for _wrap_text — kept for test compatibility."""
        return self._wrap_text(text, font, max_width)

    def _wrap_text_vertical(self, text: str, font: object, max_height: int) -> List[str]:
        """
        Character-by-character CJK vertical wrapping.
        Returns a list of column strings.
        """
        columns: List[str] = []
        current_col = ""
        char_h = 12  # approx char height px

        for char in text:
            if len(current_col) * char_h >= max_height and current_col:
                columns.append(current_col)
                current_col = char
            else:
                current_col += char

        if current_col:
            columns.append(current_col)
        return columns

    def _fit_font(
        self,
        text: str,
        avail_w: int,
        avail_h: int,
        max_font_size_limit: Optional[int] = None,
        line_spacing_ratio: Optional[float] = None,
        tone: Tone = Tone.STANDARD,
    ):
        """
        Binary-search the largest font size where word-wrapped text fits
        within avail_w × avail_h pixels.

        Returns:
            (font, lines, line_height) tuple.
        """
        lo = _MIN_FONT_SIZE
        hi = max(avail_h, avail_w, _MIN_FONT_SIZE + 1)
        if max_font_size_limit is not None:
            hi = min(hi, max_font_size_limit)
            hi = max(hi, lo + 1)

        best_font = self._get_font_for_tone(tone, lo)
        best_lines = self._wrap_text(text, best_font, avail_w)
        best_lh = self._line_height_for_font(best_font, lo, line_spacing_ratio=line_spacing_ratio)

        for _ in range(24):  # 24 iterations → ≤1 px precision on any range
            if lo > hi:
                break
            mid = (lo + hi) // 2
            font = self._get_font_for_tone(tone, mid)
            lines = self._wrap_text(text, font, avail_w)
            lh = self._line_height_for_font(font, mid, line_spacing_ratio=line_spacing_ratio)
            total_h = lh * len(lines)

            if total_h <= avail_h:
                best_font = font
                best_lines = lines
                best_lh = lh
                lo = mid + 1      # try bigger
            else:
                hi = mid - 1      # too big

        return best_font, best_lines, best_lh

    def _fit_font_to_box(self, text: str, avail_w: int, avail_h: int) -> object:
        """Alias kept for legacy test compatibility."""
        font, _, _ = self._fit_font(text, avail_w, avail_h)
        return font

    def _render_region_text(
        self,
        draw: object,
        region: object,
        font: object,
        canvas_size: tuple,
        padding: Optional[int] = None,
        line_spacing_ratio: Optional[float] = None,
    ) -> None:
        """
        Render a single region's translated text on the canvas.
        Text is converted to ALL CAPS and drawn with a white outline stroke.

        Args:
            draw:        PIL ImageDraw object.
            region:      TextRegion with box and translated_text.
            font:        PIL ImageFont object for rendering.
            canvas_size: (width, height) of the canvas.
            padding:     Optional padding (px) to override _BOX_PADDING.
            line_spacing_ratio: Optional line spacing ratio.
        """
        box = region.box
        text = region.translated_text.upper().strip()

        # Get canvas dimensions safely
        try:
            cw, ch = canvas_size
        except (TypeError, AttributeError, ValueError):
            cw, ch = 9999, 9999

        pad = _BOX_PADDING if padding is None else padding
        # Clamp box to canvas with potential horizontal expansion
        box_x_min, box_y_min, box_x_max, box_y_max = self._get_expanded_box(box, cw, ch)
        x_min = box_x_min + pad
        y_min = box_y_min + pad
        x_max = box_x_max - pad
        y_max = box_y_max - pad

        if x_max <= x_min or y_max <= y_min:
            logger.debug(
                "[TypesetterWorker] Region %s has zero area; skipping.", region.id
            )
            return

        avail_w = x_max - x_min
        avail_h = y_max - y_min

        # Get custom fill color from speaker_id if enabled
        fill_color = (0, 0, 0)
        if self._tone_engine_enabled:
            speaker_id = getattr(region, "speaker_id", "")
            color = color_for_speaker(speaker_id)
            if color is not None:
                fill_color = color

        self._render_horizontal_text(
            draw, text, font, x_min, y_min, avail_w, avail_h,
            line_spacing_ratio=line_spacing_ratio, fill_color=fill_color
        )

    def _render_horizontal_text(
        self,
        draw: object,
        text: str,
        font: object,
        x_min: int,
        y_min: int,
        available_width: int,
        available_height: int,
        line_spacing_ratio: Optional[float] = None,
        fill_color: tuple | str = (0, 0, 0),
    ) -> None:
        """
        Render horizontal text (left-to-right, top-to-bottom lines) with
        white outline stroke + black or custom speaker fill.

        Each line is centred horizontally. The text block is centred vertically.
        """
        lines = self._wrap_text(text, font, available_width)

        try:
            size = getattr(font, "size", 14)
            line_height = self._line_height_for_font(font, size, line_spacing_ratio=line_spacing_ratio)
        except (TypeError, AttributeError, ValueError):
            line_height = 14

        total_h = line_height * len(lines)
        current_y = y_min + max(0, (available_height - total_h) // 2)

        sw = self._stroke_width

        for line in lines:
            if current_y + line_height > y_min + available_height:
                break

            # Measure line width for horizontal centering
            line_w = self._measure_line(font, line)
            line_x = x_min + max(0, (available_width - int(line_w)) // 2)

            # ── White outline (stroke) ────────────────────────────────────────
            if sw > 0:
                for ox in range(-sw, sw + 1):
                    for oy in range(-sw, sw + 1):
                        if ox == 0 and oy == 0:
                            continue
                        try:
                            draw.text(
                                (line_x + ox, current_y + oy),
                                line,
                                fill=(255, 255, 255),
                                font=font,
                            )
                        except Exception:
                            pass

            # ── Fill (Black or speaker-specific color) ────────────────────────
            try:
                draw.text(
                    (line_x, current_y),
                    line,
                    fill=fill_color,
                    font=font,
                )
            except Exception as exc:
                logger.warning(
                    "[TypesetterWorker] draw.text failed for line %r: %s", line, exc
                )

            current_y += line_height

    def _render_vertical_text(
        self,
        draw: object,
        text: str,
        font: object,
        x_min: int,
        y_min: int,
        available_width: int,
        available_height: int,
    ) -> None:
        """
        Render vertical text (top-to-bottom, left-to-right columns).
        Kept for API compatibility; English text is rendered horizontally.
        """
        # English translated text always rendered horizontally for readability
        self._render_horizontal_text(
            draw, text, font, x_min, y_min, available_width, available_height
        )
