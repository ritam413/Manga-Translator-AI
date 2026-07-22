"""Tests for typesetting features extraction."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.models import BoundingBox
from finetuning.models import TypesettingFeatures
from finetuning.typesetting_features import _extract_from_crop, extract_typesetting_features


def create_synthetic_crop(
    bg_color: int = 255,
    w: int = 100,
    h: int = 100,
    rects: list[tuple[int, int, int, int]] = None,
    text_color: int = 0,
) -> np.ndarray:
    """Helper to create a simple grayscale synthetic image with text blocks."""
    img = Image.new("L", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)
    if rects:
        for (rx, ry, rw, rh) in rects:
            draw.rectangle([rx, ry, rx + rw, ry + rh], fill=text_color)
    return np.array(img)


def test_extract_single_line_rectangular_bubble():
    # White 100x40 crop, black "HELLO" block 80x16 centered
    crop_gray = create_synthetic_crop(
        bg_color=255, w=100, h=40, rects=[(10, 12, 80, 16)], text_color=0
    )
    # padding is distance from crop_origin to box, plus margin
    en_box = BoundingBox(x_min=10, y_min=12, x_max=90, y_max=28)
    
    # Provide the exact crop and simulate origin (0,0) with identical en_box coordinates
    # We simulate passing the exact crop as if it were the whole page, so origin is 0,0
    features = extract_typesetting_features(crop_gray, en_box)

    assert features.line_count == 1
    assert 13 <= features.font_size <= 18
    assert features.padding == 0 # we made en_box tight around ink
    assert features.wrap_shape == "rectangular"


def test_two_line_spacing():
    # Two horizontal bars, gap = 30px, bar height 14px
    # Total h = 14 + 30 + 14 = 58. Let's make box height 65.
    crop_gray = create_synthetic_crop(
        bg_color=255,
        w=100,
        h=65,
        rects=[
            (10, 5, 80, 14),   # line 1
            (10, 49, 80, 14),  # line 2 (gap = 49 - 19 = 30)
        ],
        text_color=0,
    )
    en_box = BoundingBox(x_min=5, y_min=0, x_max=95, y_max=65)

    features = extract_typesetting_features(crop_gray, en_box)

    assert features.line_count == 2
    assert 13 <= features.font_size <= 16
    # Expected line spacing approx: gap / font_size = 28 / 16 = 1.75, or 29/15 = 1.93
    assert 1.60 <= features.line_spacing <= 2.00


def test_elliptical_corners_empty():
    # Text ellipse in center of 80x80, corners blank
    # To simulate this, we draw a block that's not quite touching corners
    crop_gray = create_synthetic_crop(
        bg_color=255,
        w=80,
        h=80,
        rects=[
            (25, 10, 30, 10), # top middle
            (10, 25, 60, 30), # center wide
            (25, 60, 30, 10), # bottom middle
        ],
        text_color=0,
    )
    en_box = BoundingBox(x_min=10, y_min=10, x_max=70, y_max=70)
    features = extract_typesetting_features(crop_gray, en_box)
    
    assert features.wrap_shape == "elliptical"


def test_black_bubble_white_text():
    # Dark bubble (panel art), white text.
    crop_gray = create_synthetic_crop(
        bg_color=0, w=100, h=40, rects=[(10, 12, 80, 16)], text_color=255
    )
    en_box = BoundingBox(x_min=10, y_min=12, x_max=90, y_max=28)
    features = extract_typesetting_features(crop_gray, en_box)

    assert features.line_count == 1
    assert 13 <= features.font_size <= 18
    assert features.wrap_shape == "rectangular"


def test_empty_crop_error_handled_by_fallback():
    crop_gray = create_synthetic_crop(bg_color=255, w=100, h=40, rects=[])
    en_box = BoundingBox(x_min=10, y_min=10, x_max=90, y_max=30)
    
    with pytest.raises(ValueError, match="No text components found"):
        extract_typesetting_features(crop_gray, en_box)
