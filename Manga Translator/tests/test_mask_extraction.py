"""Tests for finetuning inpaint mask extraction."""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.models import BoundingBox
from finetuning.mask_extraction import (
    extract_inpaint_mask_from_pages,
    save_region_mask,
)


def test_jp_text_erased():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    
    # JP text: black rect in raw
    cv2.rectangle(raw, (20, 20), (60, 40), 0, -1)
    
    jp_box = BoundingBox(20, 20, 60, 40)
    en_box = BoundingBox(20, 20, 60, 40)
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_box, en_box)
    
    # Should have a blob of roughly 40x20
    assert result.mask_pixel_count > 0
    assert result.mask_confidence > 0.0


def test_en_text_excluded():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    
    # JP text in raw
    cv2.rectangle(raw, (20, 20), (60, 40), 0, -1)
    # EN text in eng (slightly offset)
    cv2.rectangle(eng, (20, 50), (60, 70), 0, -1)
    
    jp_box = BoundingBox(20, 20, 60, 40)
    en_box = BoundingBox(20, 50, 60, 70)
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_box, en_box)
    
    # EN text shouldn't be in the mask
    en_crop = result.mask[
        max(0, 50 - result.mask_roi_box.y_min) : min(result.mask.shape[0], 70 - result.mask_roi_box.y_min),
        max(0, 20 - result.mask_roi_box.x_min) : min(result.mask.shape[1], 60 - result.mask_roi_box.x_min)
    ]
    assert np.count_nonzero(en_crop) == 0


def test_no_change():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    
    jp_box = BoundingBox(20, 20, 60, 40)
    en_box = BoundingBox(20, 20, 60, 40)
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_box, en_box)
    assert result.mask_pixel_count == 0
    assert result.mask_confidence == 0.0


def test_union_roi():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    
    # change only in jp_box
    cv2.rectangle(raw, (10, 10), (30, 30), 0, -1)
    
    jp_box = BoundingBox(10, 10, 30, 30)
    en_box = BoundingBox(70, 70, 90, 90)
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_box, en_box)
    assert result.mask_pixel_count > 0
    assert result.mask_roi_box.x_min <= 10
    assert result.mask_roi_box.x_max >= 90


def test_morph_noise():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    
    # JP text
    cv2.rectangle(raw, (20, 20), (60, 40), 0, -1)
    
    # 1-2px salt noise in eng
    eng[5, 5] = 0
    eng[95, 95] = 0
    
    jp_box = BoundingBox(0, 0, 100, 100)
    en_box = BoundingBox(0, 0, 100, 100)
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_box, en_box, min_component_area=16)
    
    assert result.mask_pixel_count > 0
    # Noise points shouldn't be in the mask
    assert result.mask[5, 5] == 0
    assert result.mask[95, 95] == 0


def test_save_region_mask():
    with tempfile.TemporaryDirectory() as tmpdir:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask_path = save_region_mask(mask, tmpdir, "manga-id/1", "page 1", 0)
        
        assert mask_path == "masks/manga_id_1_page_1_region0.png"
        assert (Path(tmpdir) / "manga_id_1_page_1_region0.png").exists()


def test_dict_box_input():
    raw = np.full((100, 100), 255, dtype=np.uint8)
    eng = np.full((100, 100), 255, dtype=np.uint8)
    cv2.rectangle(raw, (20, 20), (60, 40), 0, -1)
    
    jp_dict = {"x_min": 20, "y_min": 20, "x_max": 60, "y_max": 40}
    en_dict = {"x_min": 20, "y_min": 20, "x_max": 60, "y_max": 40}
    
    result = extract_inpaint_mask_from_pages(raw, eng, jp_dict, en_dict)
    assert result.mask_pixel_count > 0
