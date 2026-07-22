"""EasyOCR text detection for finetuning extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.models import BoundingBox
from finetuning.models import DetectedRegion

if TYPE_CHECKING:
    import easyocr

logger = logging.getLogger(__name__)

try:
    import easyocr as _easyocr
except ImportError:  # pragma: no cover
    _easyocr = None  # type: ignore[assignment]


def detect_text_regions(
    image_path: Path,
    languages: list[str],
    reader: easyocr.Reader | None = None,
    gpu: bool = True,
    enable_box_merge: bool = False,
) -> list[DetectedRegion]:
    """Run EasyOCR on an image and return detected text regions."""
    owns_reader = reader is None
    if reader is None:
        if _easyocr is None:
            raise ImportError("easyocr is not installed. Run: pip install easyocr==1.7.1")
        reader = _easyocr.Reader(languages, gpu=gpu)

    try:
        results = reader.readtext(str(image_path))
    except Exception:
        if owns_reader:
            del reader
        raise

    regions: list[DetectedRegion] = []
    for bbox_quad, text, confidence in results:
        top_left = bbox_quad[0]
        bottom_right = bbox_quad[2]
        x1, y1 = top_left
        x2, y2 = bottom_right
        box = BoundingBox(
            x_min=int(x1),
            y_min=int(y1),
            x_max=int(x2),
            y_max=int(y2),
        )
        regions.append(
            DetectedRegion(
                box=box,
                text=str(text),
                confidence=float(confidence),
            )
        )

    if owns_reader:
        del reader

    if enable_box_merge:
        is_cjk = any(lang in ["ja", "zh", "ko"] for lang in languages)
        regions = _merge_close_boxes(regions, is_cjk)

    logger.debug("Detected %d region(s) in %s", len(regions), image_path)
    return regions


def _merge_close_boxes(regions: list[DetectedRegion], is_cjk: bool) -> list[DetectedRegion]:
    if not regions:
        return []

    n = len(regions)
    adj = {i: set() for i in range(n)}
    
    for i in range(n):
        for j in range(i + 1, n):
            r1 = regions[i]
            r2 = regions[j]
            
            if r1.box.x_min <= r2.box.x_min:
                gap_x = r2.box.x_min - r1.box.x_max
            else:
                gap_x = r1.box.x_min - r2.box.x_max
                
            if gap_x > 35:
                continue
                
            overlap_y = max(0, min(r1.box.y_max, r2.box.y_max) - max(r1.box.y_min, r2.box.y_min))
            min_h = min(r1.box.height(), r2.box.height())
            overlap_ratio = overlap_y / min_h if min_h > 0 else 0.0
            
            if overlap_ratio >= 0.3:
                adj[i].add(j)
                adj[j].add(i)
                
    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
            
    merged = []
    for comp in components:
        comp_regions = [regions[i] for i in comp]
        if len(comp_regions) == 1:
            merged.append(comp_regions[0])
            continue
            
        x_min = min(r.box.x_min for r in comp_regions)
        y_min = min(r.box.y_min for r in comp_regions)
        x_max = max(r.box.x_max for r in comp_regions)
        y_max = max(r.box.y_max for r in comp_regions)
        
        if is_cjk:
            sorted_comp = sorted(comp_regions, key=lambda r: r.box.x_max, reverse=True)
            text_join = ""
        else:
            sorted_comp = sorted(comp_regions, key=lambda r: r.box.x_min)
            text_join = " "
            
        merged_text = text_join.join(r.text for r in sorted_comp)
        merged_confidence = sum(r.confidence for r in sorted_comp) / len(sorted_comp)
        
        from core.models import BoundingBox
        merged_box = BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        merged.append(
            DetectedRegion(
                box=merged_box,
                text=merged_text,
                confidence=merged_confidence,
            )
        )
        
    merged.sort(key=lambda r: (r.box.y_min, r.box.x_min))
    return merged
