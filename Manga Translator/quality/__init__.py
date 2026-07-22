"""
Quality Control & Heuristics module for the Manga Translator pipeline.

Provides stateless heuristic filters (quality/heuristics.py) and the
orchestrating QualityChecker (quality/checker.py) that applies all rules
to TextRegion objects before committing them to the manifest.
"""

from quality.heuristics import (
    HeuristicResult,
    BoundingBoxSizeFilter,
    AspectRatioFilter,
    OcrConfidenceFilter,
    TextDensityFilter,
    DuplicateRegionFilter,
    GarbageTextFilter,
)
from quality.checker import QualityChecker, CheckReport
from quality.bubble_shape import BubbleShape, detect_bubble_contour, classify_bubble_shape
from quality.tone_engine import Tone, ToneScore, score_tone, color_for_speaker

__all__ = [
    "HeuristicResult",
    "BoundingBoxSizeFilter",
    "AspectRatioFilter",
    "OcrConfidenceFilter",
    "TextDensityFilter",
    "DuplicateRegionFilter",
    "GarbageTextFilter",
    "QualityChecker",
    "CheckReport",
    "BubbleShape",
    "detect_bubble_contour",
    "classify_bubble_shape",
    "Tone",
    "ToneScore",
    "score_tone",
    "color_for_speaker",
]
