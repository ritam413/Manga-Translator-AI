"""Typesetting feature extraction module for finetuning."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from core.models import BoundingBox
from finetuning.models import TypesettingFeatures

logger = logging.getLogger(__name__)


def get_text_ink_mask(
    page_gray: np.ndarray,
    box: BoundingBox,
    *,
    crop_margin_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Returns (text_mask_crop, crop_gray, crop_origin). Encapsulates cropping and binarization."""
    h, w = page_gray.shape
    x_min = max(0, box.x_min - crop_margin_px)
    y_min = max(0, box.y_min - crop_margin_px)
    x_max = min(w, box.x_max + crop_margin_px)
    y_max = min(h, box.y_max + crop_margin_px)

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Clamped bounding box has zero area.")

    crop_gray = page_gray[y_min:y_max, x_min:x_max]
    crop_origin = (x_min, y_min)
    
    crop_h, crop_w = crop_gray.shape

    # 1. Binarize
    # Blur to suppress JPEG noise
    blurred = cv2.GaussianBlur(crop_gray, (3, 3), 0)
    
    # Otsu thresholding
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Perimeter sampling for mask inversion
    # We sample a 1-2px border (we use 1px here)
    if crop_h > 2 and crop_w > 2:
        top = crop_gray[0:1, :]
        bottom = crop_gray[-1:, :]
        left = crop_gray[1:-1, 0:1]
        right = crop_gray[1:-1, -1:]
        perimeter = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
        border_median = np.median(perimeter)
    else:
        border_median = np.median(crop_gray)

    # Calculate mean of each Otsu class to figure out which is closer to the border median
    class_0_mean = np.mean(blurred[otsu_mask == 0]) if np.any(otsu_mask == 0) else 0
    class_255_mean = np.mean(blurred[otsu_mask == 255]) if np.any(otsu_mask == 255) else 255

    dist_0 = abs(border_median - class_0_mean)
    dist_255 = abs(border_median - class_255_mean)

    # We want text to be True (255) in text_mask
    if dist_255 <= dist_0:
        # Background is white, text is black. Otsu 0 is black.
        text_mask = (otsu_mask == 0).astype(np.uint8) * 255
    else:
        # Background is black, text is white. Otsu 255 is white.
        text_mask = (otsu_mask == 255).astype(np.uint8) * 255

    # Morphological close to merge stroke fragments
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed, crop_gray, crop_origin


def extract_typesetting_features(
    page_gray: np.ndarray,
    en_box: BoundingBox | dict[str, Any],
    *,
    crop_margin_px: int = 2,
) -> TypesettingFeatures:
    """Extract typesetting features from a grayscale page image using the en_box."""
    if isinstance(en_box, dict):
        en_box_obj = BoundingBox(
            x_min=en_box["x_min"],
            y_min=en_box["y_min"],
            x_max=en_box["x_max"],
            y_max=en_box["y_max"],
        )
    else:
        en_box_obj = en_box

    closed, crop_gray, crop_origin = get_text_ink_mask(
        page_gray, en_box_obj, crop_margin_px=crop_margin_px
    )

    return _extract_from_crop(closed, crop_gray, en_box_obj, crop_origin)


def _extract_from_crop(
    closed: np.ndarray,
    crop_gray: np.ndarray,
    en_box: BoundingBox,
    crop_origin: tuple[int, int],
) -> TypesettingFeatures:
    crop_h, crop_w = crop_gray.shape

    # 2. Line clustering via connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # Filter components to remove speckle
    min_area = 4
    min_w = 2
    min_h = 4
    components = []
    for i in range(1, num_labels):  # skip background 0
        x, y, w, h, area = stats[i]
        cy = centroids[i][1]
        if area >= min_area and w >= min_w and h >= min_h:
            components.append({"x": x, "y": y, "w": w, "h": h, "cy": cy, "area": area})
            
    if not components:
        logger.debug("No components found after filtering.")
        raise ValueError("No text components found")

    components.sort(key=lambda c: c["cy"])

    # Font size estimate
    heights = [c["h"] for c in components]
    if len(heights) >= 3:
        cutoff = np.percentile(heights, 20)
        usable_heights = [h for h in heights if h >= cutoff]
    else:
        usable_heights = heights
    
    font_size_estimate = int(np.median(usable_heights))
    font_size_estimate = max(8, min(72, font_size_estimate))

    # Clustering into lines
    lines = []
    current_line = []
    current_line_cy = components[0]["cy"]

    for c in components:
        # If cy differs by < font_size_estimate * 0.6, it's the same line
        if abs(c["cy"] - current_line_cy) < font_size_estimate * 0.6:
            current_line.append(c)
            current_line_cy = np.mean([cc["cy"] for cc in current_line])
        else:
            lines.append(current_line)
            current_line = [c]
            current_line_cy = c["cy"]
    if current_line:
        lines.append(current_line)

    line_count = len(lines)

    # 3. Compute features
    # font_size
    font_size = font_size_estimate

    # padding
    ink_x_min = min(c["x"] for c in components)
    ink_y_min = min(c["y"] for c in components)
    ink_x_max = max(c["x"] + c["w"] for c in components)
    ink_y_max = max(c["y"] + c["h"] for c in components)

    crop_x_offset, crop_y_offset = crop_origin

    left_pad = ink_x_min + crop_x_offset - en_box.x_min
    top_pad = ink_y_min + crop_y_offset - en_box.y_min
    right_pad = en_box.x_max - (ink_x_max + crop_x_offset)
    bottom_pad = en_box.y_max - (ink_y_max + crop_y_offset)

    raw_padding = min(left_pad, top_pad, right_pad, bottom_pad)
    # padding can't be more than half the box height, and can't be negative
    padding = int(max(0, min(raw_padding, (en_box.y_max - en_box.y_min) // 2)))

    # line spacing
    if line_count >= 2:
        gaps = []
        for i in range(len(lines) - 1):
            prev_y_max = max(c["y"] + c["h"] for c in lines[i])
            next_y_min = min(c["y"] for c in lines[i+1])
            gaps.append(next_y_min - prev_y_max)
        
        ratio = np.median(gaps) / font_size if font_size > 0 else 1.0
        line_spacing = float(max(1.0, min(2.5, ratio)))
    else:
        line_spacing = 1.0

    # wrap shape
    en_box_w = en_box.x_max - en_box.x_min
    en_box_h = en_box.y_max - en_box.y_min
    en_box_area = max(1, en_box_w * en_box_h)
    
    en_box_mask = closed.copy()
    fill_ratio = float(np.count_nonzero(en_box_mask)) / en_box_area
    
    cw = max(1, crop_w // 4)
    ch = max(1, crop_h // 4)
    corner_ink = (
        np.count_nonzero(en_box_mask[:ch, :cw]) +
        np.count_nonzero(en_box_mask[:ch, -cw:]) +
        np.count_nonzero(en_box_mask[-ch:, :cw]) +
        np.count_nonzero(en_box_mask[-ch:, -cw:])
    )
    corner_area = 4 * cw * ch
    corner_ink_ratio = float(corner_ink) / corner_area

    aspect = en_box_w / en_box_h if en_box_h > 0 else 1.0

    if (corner_ink_ratio < 0.08 and fill_ratio < 0.55) or \
       (0.7 <= aspect <= 1.4 and corner_ink_ratio < 0.12):
        wrap_shape = "elliptical"
    else:
        wrap_shape = "rectangular"
        
    # confidence
    confidence = 1.0
    if padding > max(en_box_w, en_box_h) * 0.4:
        confidence -= 0.3
    if font_size == 8 or font_size == 72:
        confidence -= 0.2

    confidence = max(0.0, min(1.0, confidence))

    return TypesettingFeatures(
        font_size=font_size,
        padding=padding,
        line_spacing=round(line_spacing, 2),
        wrap_shape=wrap_shape,
        line_count=line_count,
        typesetting_confidence=round(confidence, 2),
    )
