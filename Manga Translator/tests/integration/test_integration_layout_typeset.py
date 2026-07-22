import os
import pickle
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.typesetter import TypesetterWorker
from finetuning.pipeline_integration import extract_layout_features, load_layout_model


def test_extract_layout_features():
    """Verify that layout features are extracted exactly as expected."""
    box = BoundingBox(10, 20, 110, 120)  # width=100, height=100
    jp_text = "こんにちは"                # char_count=5
    
    feats = extract_layout_features(box, jp_text)
    
    # [w, h, area, aspect_ratio, char_count, char_density]
    assert feats[0] == 100.0  # w
    assert feats[1] == 100.0  # h
    assert feats[2] == 10000.0  # area
    assert feats[3] == 1.0  # aspect_ratio
    assert feats[4] == 5.0  # char_count
    assert feats[5] == pytest.approx(5.0 / 10000.0)  # char_density


def test_load_layout_model_nonexistent():
    """Verify load_layout_model handles missing file gracefully."""
    model = load_layout_model("nonexistent_model.pkl")
    assert model is None


@patch("workers.typesetter.Image")
@patch("workers.typesetter.ImageDraw")
@patch("workers.typesetter.os.replace")
@patch("workers.typesetter.os.path.isfile", return_value=True)
def test_typesetter_uses_model_font_bounds(mock_isfile, mock_replace, mock_draw_cls, mock_image_cls, synthetic_layout_model_path):
    """Verify that TypesetterWorker uses predicted layout parameters."""
    mock_img = MagicMock()
    mock_canvas = MagicMock()
    mock_img.copy.return_value = mock_canvas
    mock_canvas.size = (800, 1200)
    mock_image_cls.open.return_value = mock_img
    mock_draw = MagicMock()
    mock_draw_cls.Draw.return_value = mock_draw
    
    # Load model
    model = load_layout_model(synthetic_layout_model_path)
    assert model is not None
    
    worker = TypesetterWorker(
        input_dir="/input",
        output_dir="/output",
        layout_model=model,
        layout_model_min_confidence=0.1
    )
    worker.load()
    
    # Create region
    # width=100, height=100, area=10000, aspect=1.0, char_count=5, density=0.0005
    # The synthetic model predictions should yield font_size ~ 14, padding ~ 4, line_spacing ~ 1.2
    region = TextRegion(
        id="r0",
        box=BoundingBox(10, 10, 110, 110),
        raw_text="こんにちは",
        translated_text="Hello"
    )
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region])
    manifest = ProjectManifest("Test", pages={"page01.png": page})
    
    # Spy on _fit_font and _render_region_text using MagicMock side_effect
    original_fit = worker._fit_font
    worker._fit_font = MagicMock(side_effect=original_fit)
    original_render = worker._render_region_text
    worker._render_region_text = MagicMock(side_effect=original_render)
    
    worker.process_page(page, manifest)
    
    # Verify that max_font_size_limit was passed to _fit_font
    worker._fit_font.assert_called_once()
    kwargs = worker._fit_font.call_args[1]
    assert "max_font_size_limit" in kwargs
    assert kwargs["max_font_size_limit"] is not None
    assert 8 <= kwargs["max_font_size_limit"] <= 72
    
    # Verify padding and line_spacing passed to _render_region_text
    worker._render_region_text.assert_called_once()
    r_kwargs = worker._render_region_text.call_args[1]
    assert "padding" in r_kwargs
    assert 0 <= r_kwargs["padding"] <= 50
    assert "line_spacing_ratio" in r_kwargs
    assert 0.5 <= r_kwargs["line_spacing_ratio"] <= 2.5


@patch("workers.typesetter.Image")
@patch("workers.typesetter.ImageDraw")
@patch("workers.typesetter.os.replace")
@patch("workers.typesetter.os.path.isfile", return_value=True)
def test_typesetter_model_disabled_baseline(mock_isfile, mock_replace, mock_draw_cls, mock_image_cls):
    """Verify TypesetterWorker falls back to baseline when model is disabled."""
    mock_img = MagicMock()
    mock_canvas = MagicMock()
    mock_img.copy.return_value = mock_canvas
    mock_canvas.size = (800, 1200)
    mock_image_cls.open.return_value = mock_img
    mock_draw = MagicMock()
    mock_draw_cls.Draw.return_value = mock_draw
    
    worker = TypesetterWorker(
        input_dir="/input",
        output_dir="/output",
        layout_model=None
    )
    worker.load()
    
    region = TextRegion(
        id="r0",
        box=BoundingBox(10, 10, 110, 110),
        raw_text="こんにちは",
        translated_text="Hello"
    )
    page = PageManifest(image_filename="page01.png", width=800, height=1200, text_regions=[region])
    manifest = ProjectManifest("Test", pages={"page01.png": page})
    
    original_fit = worker._fit_font
    worker._fit_font = MagicMock(side_effect=original_fit)
    
    worker.process_page(page, manifest)
    
    worker._fit_font.assert_called_once()
    kwargs = worker._fit_font.call_args[1]
    assert kwargs.get("max_font_size_limit") is None
    assert kwargs.get("line_spacing_ratio") is None
