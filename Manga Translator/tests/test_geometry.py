"""Tests for finetuning geometry (IoU matching)."""

from __future__ import annotations

from core.models import BoundingBox
from finetuning.geometry import calculate_iou, match_regions, union_box
from finetuning.models import DetectedRegion


def _region(x_min: int, y_min: int, x_max: int, y_max: int, text: str = "x") -> DetectedRegion:
    return DetectedRegion(
        box=BoundingBox(x_min, y_min, x_max, y_max),
        text=text,
        confidence=1.0,
    )


class TestCalculateIou:
    def test_identical_boxes(self) -> None:
        box = BoundingBox(0, 0, 100, 100)
        assert calculate_iou(box, box) == 1.0

    def test_no_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert calculate_iou(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        iou = calculate_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area_union(self) -> None:
        a = BoundingBox(5, 5, 5, 5)
        b = BoundingBox(5, 5, 5, 5)
        assert calculate_iou(a, b) == 0.0


class TestMatchRegions:
    def test_greedy_assigns_best_pairs(self) -> None:
        jp = [
            _region(0, 0, 100, 100, "jp1"),
            _region(200, 0, 300, 100, "jp2"),
        ]
        en = [
            _region(5, 5, 95, 95, "en1"),
            _region(205, 5, 295, 95, "en2"),
        ]
        matches = match_regions(jp, en, iou_threshold=0.3)
        assert len(matches) == 2
        assert matches[0][0].text == "jp1"
        assert matches[0][1].text == "en1"

    def test_no_double_assign_en_box(self) -> None:
        jp = [_region(0, 0, 100, 100, "jp1"), _region(10, 10, 90, 90, "jp2")]
        en = [_region(5, 5, 95, 95, "en1")]
        matches = match_regions(jp, en, iou_threshold=0.3)
        assert len(matches) == 1

    def test_below_threshold_no_match(self) -> None:
        jp = [_region(0, 0, 50, 50)]
        en = [_region(200, 200, 250, 250)]
        assert match_regions(jp, en, iou_threshold=0.3) == []

    def test_empty_inputs(self) -> None:
        assert match_regions([], []) == []


class TestUnionBox:
    def test_union_overlapping(self) -> None:
        a = BoundingBox(10, 10, 50, 50)
        b = BoundingBox(20, 20, 60, 60)
        u = union_box(a, b)
        assert u == BoundingBox(10, 10, 60, 60)

    def test_union_disjoint(self) -> None:
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        u = union_box(a, b)
        assert u == BoundingBox(0, 0, 30, 30)

    def test_union_contained(self) -> None:
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(40, 40, 60, 60)
        u = union_box(a, b)
        assert u == BoundingBox(0, 0, 100, 100)
