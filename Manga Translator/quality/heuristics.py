"""
quality/heuristics.py
---------------------
Stateless, composable heuristic filters for the Quality Control stage.

Each filter exposes a single method:
    check(region: TextRegion, page: PageManifest | None) -> HeuristicResult

A HeuristicResult carries:
    - passed (bool)      : True if the region cleared this filter.
    - reason (str)       : Human-readable explanation (non-empty when failed).
    - flag_review (bool) : True if the region should be marked needs_review.
"""

from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from core.models import TextRegion, PageManifest, BoundingBox

logger = logging.getLogger("quality.heuristics")


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeuristicResult:
    """Outcome of a single heuristic filter applied to one TextRegion."""
    filter_name: str
    passed: bool
    reason: str = ""
    flag_review: bool = False

    def __bool__(self) -> bool:          # convenience: `if result:` → passed
        return self.passed


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseHeuristic(ABC):
    """Abstract base class for all heuristic filters."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        """
        Evaluate a single TextRegion against this heuristic.

        Args:
            region: The TextRegion to evaluate.
            page:   Optional PageManifest for context (e.g. page dimensions).

        Returns:
            HeuristicResult describing whether the region passed.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 1. Bounding-Box Size Filter
# ──────────────────────────────────────────────────────────────────────────────

class BoundingBoxSizeFilter(BaseHeuristic):
    """
    Rejects text regions whose bounding box is either too small (likely OCR
    noise) or too large (likely a full-page capture or misdetection).

    Default thresholds are tuned for typical manga pages (~1000–2000 px tall):
        - min_area: 200 px²   (noise floor)
        - max_area: 400 000 px²  (~640×640 px, should not encompass the full page)
        - min_side: 10 px  (thin sliver guard)
    """

    def __init__(
        self,
        min_area: int = 200,
        max_area: int = 400_000,
        min_side: int = 10,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.min_side = min_side

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        box = region.box
        area = box.area()
        w, h = box.width(), box.height()

        if w < self.min_side or h < self.min_side:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"Bounding box too thin: {w}×{h} px "
                    f"(min side = {self.min_side} px)."
                ),
                flag_review=False,  # hard reject, no human value
            )

        if area < self.min_area:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"Area {area} px² is below minimum {self.min_area} px² "
                    f"— likely OCR noise."
                ),
                flag_review=False,
            )

        if area > self.max_area:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"Area {area} px² exceeds maximum {self.max_area} px² "
                    f"— likely a full-page or panel misdetection."
                ),
                flag_review=True,  # worth human inspection
            )

        return HeuristicResult(filter_name=self.name, passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Aspect-Ratio Filter
# ──────────────────────────────────────────────────────────────────────────────

class AspectRatioFilter(BaseHeuristic):
    """
    Rejects regions whose width-to-height ratio falls outside a sane range.

    Manga text bubbles are roughly square to moderately rectangular.
    Extremely thin strips (ratio < 0.05 or > 20) are almost always artefacts.

    Args:
        min_ratio: Minimum allowed width/height ratio (default 0.05).
        max_ratio: Maximum allowed width/height ratio (default 20.0).
    """

    def __init__(self, min_ratio: float = 0.05, max_ratio: float = 20.0) -> None:
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        box = region.box
        h = box.height()
        if h == 0:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason="Bounding box has zero height.",
                flag_review=False,
            )

        ratio = box.width() / h

        if ratio < self.min_ratio:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"Aspect ratio {ratio:.3f} is below minimum {self.min_ratio} "
                    f"(extremely tall/thin strip)."
                ),
                flag_review=False,
            )

        if ratio > self.max_ratio:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"Aspect ratio {ratio:.3f} exceeds maximum {self.max_ratio} "
                    f"(extremely wide/flat strip)."
                ),
                flag_review=False,
            )

        return HeuristicResult(filter_name=self.name, passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. OCR Confidence Filter
# ──────────────────────────────────────────────────────────────────────────────

class OcrConfidenceFilter(BaseHeuristic):
    """
    Flags or rejects text regions below an OCR confidence threshold.

    Regions between soft_threshold and hard_threshold are kept but marked
    for human review.  Regions below hard_threshold are rejected outright.

    Args:
        hard_threshold: Confidence below which the region is rejected (default 0.20).
        soft_threshold: Confidence below which needs_review is set (default 0.50).
    """

    def __init__(
        self,
        hard_threshold: float = 0.20,
        soft_threshold: float = 0.50,
    ) -> None:
        if hard_threshold > soft_threshold:
            raise ValueError(
                "hard_threshold must be ≤ soft_threshold "
                f"(got hard={hard_threshold}, soft={soft_threshold})."
            )
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        conf = region.ocr_confidence

        # Newly added regions with zero confidence (not yet OCR'd) pass silently.
        if conf == 0.0 and not region.raw_text:
            return HeuristicResult(filter_name=self.name, passed=True)

        if conf < self.hard_threshold:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"OCR confidence {conf:.3f} is below hard threshold "
                    f"{self.hard_threshold:.3f} — likely garbage detection."
                ),
                flag_review=False,
            )

        if conf < self.soft_threshold:
            return HeuristicResult(
                filter_name=self.name,
                passed=True,
                reason=(
                    f"OCR confidence {conf:.3f} is below soft threshold "
                    f"{self.soft_threshold:.3f} — flagged for human review."
                ),
                flag_review=True,
            )

        return HeuristicResult(filter_name=self.name, passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Text Density Filter
# ──────────────────────────────────────────────────────────────────────────────

class TextDensityFilter(BaseHeuristic):
    """
    Checks whether the raw_text contains a suspiciously low or high number of
    Unicode characters relative to the bounding-box area, which can indicate
    OCR hallucination or capture of non-text artwork.

    Density = len(stripped_text) / sqrt(area).
    Acceptable range: [min_density, max_density].

    Args:
        min_density: Floor below which the region is flagged (default 0.02).
        max_density: Ceiling above which the region is flagged (default 5.0).
    """

    def __init__(
        self,
        min_density: float = 0.02,
        max_density: float = 5.0,
    ) -> None:
        self.min_density = min_density
        self.max_density = max_density

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        text = (region.raw_text or "").strip()

        # Skip density check for regions that haven't been OCR'd yet.
        if not text:
            return HeuristicResult(filter_name=self.name, passed=True)

        area = region.box.area()
        if area == 0:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason="Bounding box has zero area — cannot compute density.",
                flag_review=False,
            )

        density = len(text) / (area ** 0.5)

        if density < self.min_density:
            return HeuristicResult(
                filter_name=self.name,
                passed=True,
                reason=(
                    f"Text density {density:.4f} below minimum {self.min_density} "
                    f"— very sparse text; may be SFX or label."
                ),
                flag_review=True,
            )

        if density > self.max_density:
            return HeuristicResult(
                filter_name=self.name,
                passed=True,
                reason=(
                    f"Text density {density:.4f} exceeds maximum {self.max_density} "
                    f"— possible OCR hallucination or tightly-packed text."
                ),
                flag_review=True,
            )

        return HeuristicResult(filter_name=self.name, passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Duplicate / Overlap Region Filter
# ──────────────────────────────────────────────────────────────────────────────

def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection-over-Union of two BoundingBoxes."""
    inter_x1 = max(a.x_min, b.x_min)
    inter_y1 = max(a.y_min, b.y_min)
    inter_x2 = min(a.x_max, b.x_max)
    inter_y2 = min(a.y_max, b.y_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    union = a.area() + b.area() - intersection
    return intersection / union if union > 0 else 0.0


class DuplicateRegionFilter(BaseHeuristic):
    """
    Detects near-duplicate regions on the same page by comparing IoU (Intersection
    over Union) between the candidate region and every already-accepted region.

    This filter is *stateful per page*: call reset() when moving to a new page.

    Args:
        iou_threshold: Regions whose IoU with any accepted region exceeds this
                       value are considered duplicates and rejected (default 0.5).
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._accepted_boxes: List[Tuple[str, BoundingBox]] = []

    def reset(self) -> None:
        """Clear the accepted-region memory (call when starting a new page)."""
        self._accepted_boxes = []

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        for accepted_id, accepted_box in self._accepted_boxes:
            iou_score = _iou(region.box, accepted_box)
            if iou_score >= self.iou_threshold:
                return HeuristicResult(
                    filter_name=self.name,
                    passed=False,
                    reason=(
                        f"IoU {iou_score:.3f} with existing region '{accepted_id}' "
                        f"exceeds threshold {self.iou_threshold} — duplicate detected."
                    ),
                    flag_review=False,
                )

        # Accept and remember this region
        self._accepted_boxes.append((region.id, region.box))
        return HeuristicResult(filter_name=self.name, passed=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Garbage-Text / Noise-Character Filter
# ──────────────────────────────────────────────────────────────────────────────

# Codepoint ranges considered meaningful for Japanese manga OCR output
_VALID_CHAR_PATTERN = re.compile(
    r"[\u3000-\u9FFF"          # CJK Unified, Kana, Kangxi radicals
    r"\uF900-\uFAFF"           # CJK Compatibility Ideographs
    r"\uFF00-\uFFEF"           # Halfwidth & Fullwidth Forms
    r"A-Za-z0-9"               # ASCII alphanumeric (e.g. loanword Roman letters)
    r"\u0020-\u007E"           # Printable ASCII (punctuation etc.)
    r"\u3000-\u303F"           # CJK Symbols & Punctuation
    r"\uFF61-\uFF9F"           # Halfwidth Katakana
    r"]"
)

_GARBAGE_RATIO_THRESHOLD = 0.4  # >40 % unrecognised chars → garbage


class GarbageTextFilter(BaseHeuristic):
    """
    Detects regions whose OCR output is dominated by unrecognised/garbage
    Unicode codepoints (e.g. control characters, Latin Extended junk produced
    by a misfire of the Japanese OCR engine on artwork).

    Args:
        garbage_ratio: Maximum allowed fraction of unrecognised characters
                       before the region is rejected (default 0.40).
        min_text_len:  Minimum raw_text length required to apply the filter.
                       Very short strings are skipped (default 2).
    """

    def __init__(
        self,
        garbage_ratio: float = _GARBAGE_RATIO_THRESHOLD,
        min_text_len: int = 2,
    ) -> None:
        self.garbage_ratio = garbage_ratio
        self.min_text_len = min_text_len

    def check(
        self,
        region: TextRegion,
        page: Optional[PageManifest] = None,
    ) -> HeuristicResult:
        text = (region.raw_text or "").strip()

        if len(text) < self.min_text_len:
            # Too short to judge — pass silently.
            return HeuristicResult(filter_name=self.name, passed=True)

        valid_chars = len(_VALID_CHAR_PATTERN.findall(text))
        total_chars = len(text)
        garbage_frac = 1.0 - (valid_chars / total_chars)

        if garbage_frac > self.garbage_ratio:
            return HeuristicResult(
                filter_name=self.name,
                passed=False,
                reason=(
                    f"{garbage_frac:.1%} of characters are unrecognised/garbage "
                    f"(threshold = {self.garbage_ratio:.0%}). "
                    f"Text: {text!r}"
                ),
                flag_review=False,
            )

        return HeuristicResult(filter_name=self.name, passed=True)
