import os
import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.inpainter import InpainterWorker
from finetuning.pipeline_integration import build_jp_ink_inpaint_mask


def test_build_jp_ink_inpaint_mask_empty_crop():
    """Verify that build_jp_ink_inpaint_mask returns zero confidence for empty crop."""
    # Completely white image (background is white, no ink)
    page_gray = np.ones((200, 200), dtype=np.uint8) * 255
    box = BoundingBox(10, 10, 50, 50)
    
    mask, confidence = build_jp_ink_inpaint_mask(page_gray, box)
    assert confidence == 0.0
    assert np.count_nonzero(mask) == 0


def test_build_jp_ink_inpaint_mask_excessive_fg():
    """Verify that build_jp_ink_inpaint_mask penalizes excessive foreground area (e.g. solid black panel border)."""
    # 200x200 page
    page_gray = np.zeros((200, 200), dtype=np.uint8)  # all zero/black (massive ink/foreground)
    box = BoundingBox(10, 10, 50, 50)
    
    # get_text_ink_mask will binarize it. Since it's all black/solid, the Otsu thresholding
    # will treat it as massive foreground.
    mask, confidence = build_jp_ink_inpaint_mask(page_gray, box)
    # fg ratio will be 1.0 (all foreground), so it should penalize by 0.8 -> confidence <= 0.2
    assert confidence <= 0.2


def test_build_jp_ink_inpaint_mask_valid_ink():
    """Verify that build_jp_ink_inpaint_mask returns high confidence for moderate ink crop."""
    # White background (255) with some black text strokes (0) in the box
    page_gray = np.ones((200, 200), dtype=np.uint8) * 255
    
    # Draw some thin black lines in the box region (10, 10) -> (50, 50)
    # Area of box is 40 * 40 = 1600. Let's draw about 100 pixels of black.
    cv2.line(page_gray, (20, 20), (40, 20), 0, 2)
    cv2.line(page_gray, (20, 30), (40, 30), 0, 2)
    
    box = BoundingBox(10, 10, 50, 50)
    mask, confidence = build_jp_ink_inpaint_mask(page_gray, box)
    
    assert confidence > 0.5  # High confidence
    assert np.count_nonzero(mask) > 0
    # verify mask is confined to region ROI + crop margin
    # Box is (10,10,50,50). With crop_margin=2, it can be up to (8,8,52,52)
    non_zero_coords = np.argwhere(mask > 0)
    for y, x in non_zero_coords:
        assert 8 <= y <= 52
        assert 8 <= x <= 52


@patch("workers.inpainter.cv2.inpaint")
@patch("workers.inpainter.cv2.imwrite")
@patch("workers.inpainter.os.replace")
def test_inpainter_worker_uses_contour_mask_when_confident(mock_replace, mock_imwrite, mock_inpaint):
    """Verify that InpainterWorker uses contour mask when confidence is above threshold."""
    # Setup worker
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(input_dir)
        
        img_filename = "page01.png"
        img_path = os.path.join(input_dir, img_filename)
        
        # Write a mock page: white page with some black strokes
        from PIL import Image
        page_gray = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.line(page_gray, (20, 20), (40, 20), 0, 2)
        Image.fromarray(page_gray).save(img_path)
        
        worker = InpainterWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            use_contour_inpaint=True,
            layout_model_min_confidence=0.5
        )
        worker.load()
        
        region = TextRegion(id="r0", box=BoundingBox(10, 10, 50, 50))
        page = PageManifest(image_filename=img_filename, width=200, height=200, text_regions=[region])
        manifest = ProjectManifest("Test", pages={img_filename: page})
        
        # Spy on build_jp_ink_inpaint_mask
        with patch("finetuning.pipeline_integration.build_jp_ink_inpaint_mask", return_value=(np.ones((200,200), dtype=np.uint8)*255, 0.9)) as mock_build:
            worker.process_page(page, manifest)
            mock_build.assert_called_once()
            
            # Verify inpaint called
            mock_inpaint.assert_called_once()
            passed_mask = mock_inpaint.call_args[0][1]
            # Since mock returned all-white (255) ink mask, passed_mask should have active pixels
            assert np.count_nonzero(passed_mask) > 0


@patch("workers.inpainter.cv2.inpaint")
@patch("workers.inpainter.cv2.imwrite")
@patch("workers.inpainter.os.replace")
def test_inpainter_worker_falls_back_to_rectangle_when_not_confident(mock_replace, mock_imwrite, mock_inpaint):
    """Verify InpainterWorker falls back to rectangle when contour confidence is too low."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(input_dir)
        
        img_filename = "page01.png"
        img_path = os.path.join(input_dir, img_filename)
        
        from PIL import Image
        page_gray = np.ones((200, 200), dtype=np.uint8) * 255
        Image.fromarray(page_gray).save(img_path)
        
        worker = InpainterWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            use_contour_inpaint=True,
            layout_model_min_confidence=0.5
        )
        worker.load()
        
        region = TextRegion(id="r0", box=BoundingBox(10, 10, 50, 50))
        page = PageManifest(image_filename=img_filename, width=200, height=200, text_regions=[region])
        manifest = ProjectManifest("Test", pages={img_filename: page})
        
        # Mock confidence = 0.2 (below threshold 0.5)
        with patch("finetuning.pipeline_integration.build_jp_ink_inpaint_mask", return_value=(np.zeros((200,200), dtype=np.uint8), 0.2)):
            worker.process_page(page, manifest)
            
            mock_inpaint.assert_called_once()
            passed_mask = mock_inpaint.call_args[0][1]
            
            # The mask should be the fallback solid rectangle!
            # Solid rectangle at box (10,10) to (50,50) should have exactly 41 * 41 = 1681 non-zero pixels
            assert np.count_nonzero(passed_mask) == 1681
