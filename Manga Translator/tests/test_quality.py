"""
tests/test_quality.py
---------------------
Comprehensive unit tests for Stage 4: Quality Control & Heuristics.

Covers:
    - All six individual heuristic filters (pass/fail/flag-review branches)
    - QualityChecker with the default pipeline on a full PageManifest
    - QualityChecker dry-run mode (remove_rejected=False)
    - CheckReport / RegionCheckResult accessors
    - Custom filter pipeline injection
    - Edge cases: zero area, zero confidence, empty text, etc.
"""

import unittest
from typing import List

from core.models import BoundingBox, TextRegion, PageManifest
from quality.heuristics import (
    HeuristicResult,
    BoundingBoxSizeFilter,
    AspectRatioFilter,
    OcrConfidenceFilter,
    TextDensityFilter,
    DuplicateRegionFilter,
    GarbageTextFilter,
    _iou,
)
from quality.checker import QualityChecker, CheckReport, RegionCheckResult


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_region(
    rid: str = "r1",
    x_min: int = 50,
    y_min: int = 50,
    x_max: int = 200,
    y_max: int = 200,
    raw_text: str = "こんにちは",
    ocr_confidence: float = 0.85,
    translated_text: str = "",
    needs_review: bool = False,
    is_vertical: bool = True,
) -> TextRegion:
    return TextRegion(
        id=rid,
        box=BoundingBox(x_min, y_min, x_max, y_max),
        raw_text=raw_text,
        translated_text=translated_text,
        ocr_confidence=ocr_confidence,
        is_vertical=is_vertical,
        needs_review=needs_review,
    )


def make_page(regions: List[TextRegion] = None) -> PageManifest:
    return PageManifest(
        image_filename="page01.png",
        width=1200,
        height=1800,
        text_regions=regions or [],
        phase_a_completed=False,
        phase_b_completed=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. BoundingBoxSizeFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestBoundingBoxSizeFilter(unittest.TestCase):

    def setUp(self):
        self.f = BoundingBoxSizeFilter(min_area=200, max_area=400_000, min_side=10)

    def test_normal_region_passes(self):
        r = make_region(x_min=0, y_min=0, x_max=100, y_max=100)  # area=10000
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertFalse(result.flag_review)

    def test_too_small_area_rejected(self):
        r = make_region(x_min=0, y_min=0, x_max=10, y_max=10)  # area=100 < 200
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertFalse(result.flag_review)
        self.assertIn("noise", result.reason)

    def test_too_thin_side_rejected(self):
        r = make_region(x_min=0, y_min=0, x_max=5, y_max=500)  # width=5 < min_side=10
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertIn("thin", result.reason)

    def test_too_large_area_flagged_for_review(self):
        # area = 700 * 700 = 490000 > 400000
        r = make_region(x_min=0, y_min=0, x_max=700, y_max=700)
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertTrue(result.flag_review)  # large → worth human review

    def test_border_area_exactly_min_passes(self):
        # area = 200 exactly → should pass
        # 10 * 20 = 200
        r = make_region(x_min=0, y_min=0, x_max=20, y_max=10)
        result = self.f.check(r)
        self.assertTrue(result.passed)

    def test_custom_thresholds(self):
        f = BoundingBoxSizeFilter(min_area=50, max_area=1000, min_side=5)
        small_ok = make_region(x_min=0, y_min=0, x_max=10, y_max=10)   # area=100 ∈ [50, 1000]
        self.assertTrue(f.check(small_ok).passed)

        too_big = make_region(x_min=0, y_min=0, x_max=50, y_max=50)   # area=2500 > 1000
        self.assertFalse(f.check(too_big).passed)


# ──────────────────────────────────────────────────────────────────────────────
# 2. AspectRatioFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestAspectRatioFilter(unittest.TestCase):

    def setUp(self):
        self.f = AspectRatioFilter(min_ratio=0.05, max_ratio=20.0)

    def test_square_region_passes(self):
        r = make_region(x_min=0, y_min=0, x_max=100, y_max=100)  # ratio=1.0
        self.assertTrue(self.f.check(r).passed)

    def test_vertical_bubble_passes(self):
        r = make_region(x_min=0, y_min=0, x_max=60, y_max=200)  # ratio=0.3
        self.assertTrue(self.f.check(r).passed)

    def test_too_tall_strip_rejected(self):
        # ratio = 5 / 200 = 0.025 < 0.05
        r = make_region(x_min=0, y_min=0, x_max=5, y_max=200)
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertIn("tall", result.reason)

    def test_too_wide_strip_rejected(self):
        # ratio = 2100 / 100 = 21 > 20
        r = make_region(x_min=0, y_min=0, x_max=2100, y_max=100)
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertIn("wide", result.reason)

    def test_zero_height_rejected(self):
        r = make_region(x_min=0, y_min=50, x_max=100, y_max=50)  # height=0
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertIn("zero height", result.reason)


# ──────────────────────────────────────────────────────────────────────────────
# 3. OcrConfidenceFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestOcrConfidenceFilter(unittest.TestCase):

    def setUp(self):
        self.f = OcrConfidenceFilter(hard_threshold=0.20, soft_threshold=0.50)

    def test_high_confidence_passes_clean(self):
        r = make_region(ocr_confidence=0.92)
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertFalse(result.flag_review)

    def test_medium_confidence_passes_with_review_flag(self):
        r = make_region(ocr_confidence=0.35, raw_text="テスト")
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertTrue(result.flag_review)

    def test_below_hard_threshold_rejected(self):
        r = make_region(ocr_confidence=0.10, raw_text="テスト")
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertFalse(result.flag_review)

    def test_zero_confidence_no_text_passes_silently(self):
        # New un-OCR'd region — must pass without flagging
        r = make_region(ocr_confidence=0.0, raw_text="")
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertFalse(result.flag_review)

    def test_invalid_threshold_order_raises(self):
        with self.assertRaises(ValueError):
            OcrConfidenceFilter(hard_threshold=0.8, soft_threshold=0.3)

    def test_exactly_at_soft_threshold_is_flagged(self):
        # 0.50 is *below* soft_threshold=0.50? No, it equals it.
        # Our check is `conf < soft_threshold`, so conf=0.50 should NOT flag.
        r = make_region(ocr_confidence=0.50, raw_text="テスト")
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertFalse(result.flag_review)

    def test_just_below_soft_threshold_flagged(self):
        r = make_region(ocr_confidence=0.499, raw_text="テスト")
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertTrue(result.flag_review)


# ──────────────────────────────────────────────────────────────────────────────
# 4. TextDensityFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestTextDensityFilter(unittest.TestCase):

    def setUp(self):
        self.f = TextDensityFilter(min_density=0.02, max_density=5.0)

    def test_normal_density_passes(self):
        # box 150×150=22500, sqrt=150, text len=10 → density≈0.067 ∈ [0.02, 5.0]
        r = make_region(x_min=0, y_min=0, x_max=150, y_max=150, raw_text="こんにちは世界!!")
        self.assertTrue(self.f.check(r).passed)

    def test_empty_text_passes_silently(self):
        r = make_region(raw_text="")
        self.assertTrue(self.f.check(r).passed)

    def test_zero_area_rejected(self):
        r = make_region(x_min=50, y_min=50, x_max=50, y_max=50, raw_text="テスト")
        result = self.f.check(r)
        self.assertFalse(result.passed)

    def test_very_sparse_density_flags_for_review(self):
        # Large box, very few characters
        # box 500×500 = 250000, sqrt=500, text="A" (len=1) → density=0.002 < 0.02
        r = make_region(x_min=0, y_min=0, x_max=500, y_max=500, raw_text="A")
        result = self.f.check(r)
        self.assertTrue(result.passed)     # accepted
        self.assertTrue(result.flag_review)

    def test_very_dense_text_flags_for_review(self):
        # tiny box 5×5=25, sqrt=5, text of 100 chars → density=20 > 5
        r = make_region(x_min=0, y_min=0, x_max=5, y_max=5, raw_text="あ" * 100)
        result = self.f.check(r)
        self.assertTrue(result.passed)
        self.assertTrue(result.flag_review)


# ──────────────────────────────────────────────────────────────────────────────
# 5. DuplicateRegionFilter + _iou helper
# ──────────────────────────────────────────────────────────────────────────────

class TestIoU(unittest.TestCase):

    def test_no_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        self.assertAlmostEqual(_iou(a, b), 0.0)

    def test_full_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(0, 0, 10, 10)
        self.assertAlmostEqual(_iou(a, b), 1.0)

    def test_partial_overlap(self):
        a = BoundingBox(0, 0, 10, 10)     # area=100
        b = BoundingBox(5, 0, 15, 10)     # area=100, intersection=5×10=50
        # union = 100+100-50=150, iou=50/150=1/3
        self.assertAlmostEqual(_iou(a, b), 1 / 3, places=5)


class TestDuplicateRegionFilter(unittest.TestCase):

    def setUp(self):
        self.f = DuplicateRegionFilter(iou_threshold=0.5)

    def test_first_region_always_passes(self):
        r = make_region("r1", 0, 0, 100, 100)
        result = self.f.check(r)
        self.assertTrue(result.passed)

    def test_identical_region_rejected(self):
        r1 = make_region("r1", 0, 0, 100, 100)
        r2 = make_region("r2", 0, 0, 100, 100)
        self.f.check(r1)
        result = self.f.check(r2)
        self.assertFalse(result.passed)
        self.assertIn("r1", result.reason)

    def test_non_overlapping_region_passes(self):
        r1 = make_region("r1", 0, 0, 100, 100)
        r2 = make_region("r2", 200, 200, 300, 300)
        self.f.check(r1)
        result = self.f.check(r2)
        self.assertTrue(result.passed)

    def test_slight_overlap_below_threshold_passes(self):
        # Overlap is ~10%, well below 0.5
        r1 = make_region("r1", 0, 0, 100, 100)
        r2 = make_region("r2", 90, 0, 190, 100)  # 10px overlap horizontally
        self.f.check(r1)
        result = self.f.check(r2)
        self.assertTrue(result.passed)

    def test_reset_clears_state(self):
        r1 = make_region("r1", 0, 0, 100, 100)
        self.f.check(r1)
        self.f.reset()
        # After reset, the same bounding box should pass again
        r2 = make_region("r2", 0, 0, 100, 100)
        result = self.f.check(r2)
        self.assertTrue(result.passed)


# ──────────────────────────────────────────────────────────────────────────────
# 6. GarbageTextFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestGarbageTextFilter(unittest.TestCase):

    def setUp(self):
        self.f = GarbageTextFilter(garbage_ratio=0.40, min_text_len=2)

    def test_japanese_text_passes(self):
        r = make_region(raw_text="お前はもう死んでいる。")
        self.assertTrue(self.f.check(r).passed)

    def test_ascii_text_passes(self):
        r = make_region(raw_text="Hello World!")
        self.assertTrue(self.f.check(r).passed)

    def test_mostly_garbage_rejected(self):
        # Control characters + garbage symbols dominate
        junk = "\x01\x02\x03\x04\x05\x06\x07\x08"
        r = make_region(raw_text=junk)
        result = self.f.check(r)
        self.assertFalse(result.passed)
        self.assertIn("garbage", result.reason)

    def test_mixed_text_with_low_garbage_passes(self):
        # "こんにちは!" — all valid chars
        r = make_region(raw_text="こんにちは!")
        self.assertTrue(self.f.check(r).passed)

    def test_very_short_text_skipped(self):
        # len("a") = 1 < min_text_len=2 → skip filter
        r = make_region(raw_text="a")
        self.assertTrue(self.f.check(r).passed)

    def test_empty_text_passes(self):
        r = make_region(raw_text="")
        self.assertTrue(self.f.check(r).passed)


# ──────────────────────────────────────────────────────────────────────────────
# 7. QualityChecker — integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQualityChecker(unittest.TestCase):

    def _make_good_region(self, rid: str, offset: int = 0) -> TextRegion:
        """Create a valid region with a unique non-overlapping bounding box."""
        x0 = offset * 300          # 300 px gap between each region
        return make_region(
            rid=rid,
            x_min=x0 + 10, y_min=10, x_max=x0 + 200, y_max=200,
            raw_text="こんにちは",
            ocr_confidence=0.90,
        )

    def test_all_good_regions_accepted(self):
        checker = QualityChecker()
        # offset each region by 300 px so DuplicateRegionFilter sees no overlap
        regions = [self._make_good_region(f"r{i}", offset=i) for i in range(3)]
        page = make_page(regions)
        report = checker.check_page(page)

        self.assertEqual(report.total, 3)
        self.assertEqual(len(report.accepted_ids), 3)
        self.assertEqual(len(report.rejected_ids), 0)
        self.assertEqual(len(page.text_regions), 3)   # none removed

    def test_bad_size_region_rejected_and_removed(self):
        good = self._make_good_region("good")
        bad = make_region("bad", x_min=0, y_min=0, x_max=5, y_max=5,  # area=25 < 200
                          raw_text="テ", ocr_confidence=0.8)

        checker = QualityChecker()
        page = make_page([good, bad])
        report = checker.check_page(page)

        self.assertIn("bad", report.rejected_ids)
        self.assertIn("good", report.accepted_ids)
        self.assertEqual(len(page.text_regions), 1)
        self.assertEqual(page.text_regions[0].id, "good")

    def test_low_confidence_region_flagged_not_removed(self):
        low_conf = make_region("low", ocr_confidence=0.35, raw_text="テスト",
                               x_min=10, y_min=10, x_max=200, y_max=200)
        checker = QualityChecker()
        page = make_page([low_conf])
        report = checker.check_page(page)

        self.assertIn("low", report.accepted_ids)
        self.assertIn("low", report.flagged_ids)
        self.assertTrue(page.text_regions[0].needs_review)

    def test_dry_run_does_not_remove_regions(self):
        bad = make_region("bad", x_min=0, y_min=0, x_max=5, y_max=5,
                          raw_text="テ", ocr_confidence=0.8)
        checker = QualityChecker(remove_rejected=False)
        page = make_page([bad])
        report = checker.check_page(page)

        self.assertIn("bad", report.rejected_ids)
        self.assertEqual(len(page.text_regions), 1)  # NOT removed

    def test_duplicate_region_rejected_on_same_page(self):
        r1 = self._make_good_region("r1")
        # r2 has same bounding box → IoU=1.0 > 0.5 → duplicate
        r2 = make_region("r2", x_min=10, y_min=10, x_max=200, y_max=200,
                         raw_text="さようなら", ocr_confidence=0.90)

        checker = QualityChecker()
        page = make_page([r1, r2])
        report = checker.check_page(page)

        self.assertIn("r1", report.accepted_ids)
        self.assertIn("r2", report.rejected_ids)

    def test_stateful_filter_reset_between_pages(self):
        """DuplicateRegionFilter must reset when check_page is called on a new page."""
        checker = QualityChecker()

        # Page 1 — r1 accepted
        r1 = self._make_good_region("r1")
        page1 = make_page([r1])
        checker.check_page(page1)

        # Page 2 — r2 has same box as r1; should PASS because state was reset
        r2 = make_region("r2", x_min=10, y_min=10, x_max=200, y_max=200,
                         raw_text="テスト", ocr_confidence=0.90)
        page2 = make_page([r2])
        page2.image_filename = "page02.png"
        report2 = checker.check_page(page2)

        self.assertIn("r2", report2.accepted_ids)

    def test_custom_filter_pipeline(self):
        """Inject a single permissive filter and verify it runs."""

        class AlwaysPassFilter(BaseHeuristic if False else object):
            name = "AlwaysPass"

            def check(self, region, page=None):
                return HeuristicResult(filter_name=self.name, passed=True)

        from quality.heuristics import BaseHeuristic as BH

        class AlwaysPass(BH):
            def check(self, region, page=None):
                return HeuristicResult(filter_name=self.name, passed=True)

        checker = QualityChecker(filters=[AlwaysPass()])
        # Even a tiny garbage region should pass with this pipeline
        bad = make_region("bad", x_min=0, y_min=0, x_max=3, y_max=3,
                          raw_text="\x01\x02\x03", ocr_confidence=0.01)
        page = make_page([bad])
        report = checker.check_page(page)
        self.assertIn("bad", report.accepted_ids)

    def test_check_report_summary_string(self):
        checker = QualityChecker()
        regions = [self._make_good_region(f"r{i}", offset=i) for i in range(2)]
        page = make_page(regions)
        report = checker.check_page(page)
        summary = report.summary()
        self.assertIn("page01.png", summary)
        self.assertIn("total=2", summary)
        self.assertIn("accepted=2", summary)

    def test_check_region_single_does_not_mutate_page(self):
        checker = QualityChecker()
        r = self._make_good_region("r1")
        page = make_page([r])
        result = checker.check_region(r, page)
        self.assertTrue(result.accepted)
        # Page text_regions unchanged
        self.assertEqual(len(page.text_regions), 1)

    def test_existing_needs_review_preserved(self):
        """A region already flagged needs_review should keep the flag even if all checks pass."""
        r = make_region("r1", needs_review=True, ocr_confidence=0.95,
                        x_min=10, y_min=10, x_max=200, y_max=200)
        checker = QualityChecker()
        page = make_page([r])
        checker.check_page(page)
        self.assertTrue(page.text_regions[0].needs_review)

    def test_empty_page_produces_empty_report(self):
        checker = QualityChecker()
        page = make_page([])
        report = checker.check_page(page)
        self.assertEqual(report.total, 0)
        self.assertEqual(report.accepted_ids, [])
        self.assertEqual(report.rejected_ids, [])


# ──────────────────────────────────────────────────────────────────────────────
# 8. RegionCheckResult helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestRegionCheckResult(unittest.TestCase):

    def test_rejection_reason_returns_first_failed_filter(self):
        fr1 = HeuristicResult("FilterA", passed=True, reason="", flag_review=False)
        fr2 = HeuristicResult("FilterB", passed=False, reason="Too small", flag_review=False)
        result = RegionCheckResult("r1", accepted=False, needs_review=False,
                                   filter_results=[fr1, fr2])
        self.assertIn("FilterB", result.rejection_reason)
        self.assertIn("Too small", result.rejection_reason)

    def test_review_reasons_accumulates_all_flags(self):
        fr1 = HeuristicResult("FilterA", passed=True, reason="Low conf", flag_review=True)
        fr2 = HeuristicResult("FilterB", passed=True, reason="Sparse text", flag_review=True)
        result = RegionCheckResult("r1", accepted=True, needs_review=True,
                                   filter_results=[fr1, fr2])
        self.assertEqual(len(result.review_reasons), 2)

    def test_bool_on_heuristic_result(self):
        ok = HeuristicResult("F", passed=True)
        fail = HeuristicResult("F", passed=False)
        self.assertTrue(bool(ok))
        self.assertFalse(bool(fail))


if __name__ == "__main__":
    unittest.main()
