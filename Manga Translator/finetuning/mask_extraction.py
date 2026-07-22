"""Mask extraction module for finetuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.models import BoundingBox
from finetuning.geometry import union_box
from finetuning.typesetting_features import get_text_ink_mask

logger = logging.getLogger(__name__)


@dataclass
class InpaintMaskResult:
    mask: np.ndarray          # cropped binary mask (H, W) uint8
    mask_confidence: float
    mask_pixel_count: int
    mask_roi_box: BoundingBox # page coords of crop (required for Phase 10)


def _to_box(b: BoundingBox | dict[str, Any]) -> BoundingBox:
    if isinstance(b, dict):
        return BoundingBox(
            x_min=b["x_min"],
            y_min=b["y_min"],
            x_max=b["x_max"],
            y_max=b["y_max"],
        )
    return b


def extract_inpaint_mask_from_pages(
    raw_gray: np.ndarray,
    eng_gray: np.ndarray,
    jp_box: BoundingBox | dict[str, Any],
    en_box: BoundingBox | dict[str, Any],
    *,
    roi_margin_px: int = 8,
    diff_threshold: int = 30,
    min_component_area: int = 16,
) -> InpaintMaskResult:
    jp = _to_box(jp_box)
    en = _to_box(en_box)

    h, w = raw_gray.shape

    # Step 1 - Compute ROI
    union = union_box(jp, en)
    x_min = max(0, union.x_min - roi_margin_px)
    y_min = max(0, union.y_min - roi_margin_px)
    x_max = min(w, union.x_max + roi_margin_px)
    y_max = min(h, union.y_max + roi_margin_px)

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Clamped ROI bounding box has zero area.")

    roi_box = BoundingBox(x_min, y_min, x_max, y_max)
    raw_crop = raw_gray[y_min:y_max, x_min:x_max]
    eng_crop = eng_gray[y_min:y_max, x_min:x_max]

    # Step 2 - Pixel differencing
    diff = cv2.absdiff(raw_crop, eng_crop)
    _, diff_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # Step 3 - Subtract English ink
    en_ink, _, crop_origin = get_text_ink_mask(eng_gray, en, crop_margin_px=2)
    # en_ink is relative to crop_origin
    en_x, en_y = crop_origin
    en_h, en_w = en_ink.shape

    # Build full-page-sized en_ink zeros array
    en_ink_full = np.zeros((h, w), dtype=np.uint8)
    # Be careful not to exceed bounds
    en_y_end = min(h, en_y + en_h)
    en_x_end = min(w, en_x + en_w)
    en_ink_full[en_y:en_y_end, en_x:en_x_end] = en_ink[:en_y_end-en_y, :en_x_end-en_x]

    # Restrict to ROI
    en_ink_roi = en_ink_full[y_min:y_max, x_min:x_max]

    # Dilate EN ink by 2-3px
    kernel_dilate = np.ones((3, 3), np.uint8)
    en_ink_dilated = cv2.dilate(en_ink_roi, kernel_dilate, iterations=1)

    erase_mask = diff_mask & ~en_ink_dilated

    # Fallback Trigger
    diff_fg = np.count_nonzero(diff_mask)
    if diff_fg > 0:
        remaining_fg = np.count_nonzero(erase_mask)
        if remaining_fg / diff_fg < 0.05:
            # Fallback: diff_mask clipped strictly to the jp_box interior
            erase_mask = np.zeros_like(diff_mask)
            jp_x_min = max(0, jp.x_min - x_min)
            jp_y_min = max(0, jp.y_min - y_min)
            jp_x_max = min(x_max - x_min, jp.x_max - x_min)
            jp_y_max = min(y_max - y_min, jp.y_max - y_min)
            if jp_x_max > jp_x_min and jp_y_max > jp_y_min:
                erase_mask[jp_y_min:jp_y_max, jp_x_min:jp_x_max] = diff_mask[jp_y_min:jp_y_max, jp_x_min:jp_x_max]
            fallback_used = True
        else:
            fallback_used = False
    else:
        fallback_used = False

    # Step 4 - Morphological cleanup
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(erase_mask, cv2.MORPH_OPEN, kernel_morph, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_morph, iterations=1)

    # Remove speckle
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    final_mask = np.zeros_like(clean)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            final_mask[labels == i] = 255

    # Re-threshold to strict 0/255 just in case
    _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 5 - Confidence score
    roi_area = (x_max - x_min) * (y_max - y_min)
    fg_pixels = int(np.count_nonzero(final_mask))

    confidence = 1.0
    if fg_pixels == 0:
        confidence = 0.0
    else:
        if fg_pixels > 0.8 * roi_area:
            confidence -= 0.8
        elif fg_pixels < 0.005 * roi_area:
            confidence -= 0.6
        if fallback_used:
            confidence -= 0.4
        
        # ROI extends beyond page?
        if (union.x_min - roi_margin_px < 0 or union.y_min - roi_margin_px < 0 or
            union.x_max + roi_margin_px > w or union.y_max + roi_margin_px > h):
            confidence -= 0.2

    confidence = max(0.0, min(1.0, confidence))

    return InpaintMaskResult(
        mask=final_mask,
        mask_confidence=confidence,
        mask_pixel_count=fg_pixels,
        mask_roi_box=roi_box
    )


def save_region_mask(
    mask: np.ndarray,
    masks_dir: str | Path,
    manga_id: str,
    page: str,
    region_index: int,
) -> str:
    """Write PNG; return relative mask_path like 'masks/manga1_1_region0.png'."""
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize
    import re
    safe_manga = re.sub(r'[^a-zA-Z0-9]', '_', manga_id)
    safe_page = re.sub(r'[^a-zA-Z0-9]', '_', page)
    
    filename = f"{safe_manga}_{safe_page}_region{region_index}.png"
    out_path = masks_dir / filename
    
    cv2.imwrite(str(out_path), mask)
    
    return f"masks/{filename}"
