"""Geometric matching utilities for finetuning extraction."""

from __future__ import annotations

from core.models import BoundingBox
from finetuning.models import DetectedRegion


def box_to_tuple(box: BoundingBox) -> tuple[int, int, int, int]:
    return (box.x_min, box.y_min, box.x_max, box.y_max)


def union_box(box_a: BoundingBox, box_b: BoundingBox) -> BoundingBox:
    """Axis-aligned union of two boxes."""
    return BoundingBox(
        x_min=min(box_a.x_min, box_b.x_min),
        y_min=min(box_a.y_min, box_b.y_min),
        x_max=max(box_a.x_max, box_b.x_max),
        y_max=max(box_a.y_max, box_b.y_max),
    )


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x_a = max(box_a.x_min, box_b.x_min)
    y_a = max(box_a.y_min, box_b.y_min)
    x_b = min(box_a.x_max, box_b.x_max)
    y_b = min(box_a.y_max, box_b.y_max)

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = box_a.area()
    area_b = box_b.area()
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def match_regions(
    jp_regions: list[DetectedRegion],
    en_regions: list[DetectedRegion],
    iou_threshold: float = 0.3,
) -> list[tuple[DetectedRegion, DetectedRegion, float]]:
    """Greedy IoU matcher: each EN box assigned at most once."""
    candidates: list[tuple[float, int, int]] = []

    for jp_idx, jp in enumerate(jp_regions):
        for en_idx, en in enumerate(en_regions):
            iou = calculate_iou(jp.box, en.box)
            if iou >= iou_threshold:
                candidates.append((iou, jp_idx, en_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_jp: set[int] = set()
    used_en: set[int] = set()
    matches: list[tuple[DetectedRegion, DetectedRegion, float]] = []

    for iou, jp_idx, en_idx in candidates:
        if jp_idx in used_jp or en_idx in used_en:
            continue
        used_jp.add(jp_idx)
        used_en.add(en_idx)
        matches.append((jp_regions[jp_idx], en_regions[en_idx], iou))

    return matches
