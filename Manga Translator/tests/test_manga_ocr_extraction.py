"""Tests for MangaOCR extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from core.models import BoundingBox
from finetuning.models import JpOcrResult
from finetuning.manga_ocr_extraction import MangaOcrExtractor, extract_jp_text_from_page


@pytest.fixture
def mock_manga_ocr(monkeypatch):
    """Patch manga_ocr.MangaOcr so no model loads."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.return_value = "こんにちは"  # default OCR result
    monkeypatch.setattr("finetuning.manga_ocr_extraction.MangaOcr", mock_cls)
    return mock_instance


def test_fallback_dataclass() -> None:
    fallback = JpOcrResult.fallback()
    assert fallback.jp_ocr_text == ""
    assert fallback.jp_ocr_confidence == 0.0
    assert fallback.to_dict() == {
        "jp_ocr_text": "",
        "jp_ocr_confidence": 0.0,
    }


def test_load_unload_idempotent(mock_manga_ocr) -> None:
    extractor = MangaOcrExtractor()
    assert not extractor.is_loaded

    extractor.load()
    assert extractor.is_loaded
    assert extractor._mocr is not None

    # Load again (idempotent)
    extractor.load()
    assert extractor.is_loaded

    extractor.unload()
    assert not extractor.is_loaded
    assert extractor._mocr is None

    # Unload again (idempotent)
    extractor.unload()
    assert not extractor.is_loaded


def test_load_raises_if_not_installed(monkeypatch) -> None:
    monkeypatch.setattr("finetuning.manga_ocr_extraction.MangaOcr", None)
    extractor = MangaOcrExtractor()
    with pytest.raises(ImportError, match="manga-ocr is not installed"):
        extractor.load()


def test_extract_text_returns_ocr_result(mock_manga_ocr) -> None:
    mock_manga_ocr.return_value = "テスト"
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 50, 50)
    result = extractor.extract_text(img, box)

    assert result.jp_ocr_text == "テスト"
    assert result.jp_ocr_confidence == 1.0


def test_extract_text_empty_result(mock_manga_ocr) -> None:
    mock_manga_ocr.return_value = "  "
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 50, 50)
    result = extractor.extract_text(img, box)

    assert result.jp_ocr_text == ""
    assert result.jp_ocr_confidence == 0.5


def test_extract_text_zero_area_box(mock_manga_ocr) -> None:
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 10, 50)
    with pytest.raises(ValueError, match="Invalid crop area"):
        extractor.extract_text(img, box)


def test_extract_text_dict_box(mock_manga_ocr) -> None:
    mock_manga_ocr.return_value = "テスト"
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50}
    result = extractor.extract_text(img, box)

    assert result.jp_ocr_text == "テスト"
    assert result.jp_ocr_confidence == 1.0


def test_crop_margin_applied(mock_manga_ocr) -> None:
    extractor = MangaOcrExtractor(crop_margin_px=5)
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 50, 50)

    with patch.object(Image.Image, "crop", wraps=img.crop) as mock_crop:
        extractor.extract_text(img, box)
        # Expected crop bounds: 10-5 = 5, 50+5 = 55
        mock_crop.assert_called_once_with((5, 5, 55, 55))


def test_nfkc_normalization(mock_manga_ocr) -> None:
    # Full-width ABC and 123
    mock_manga_ocr.return_value = " ＡＢＣ１２３ "
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 50, 50)
    result = extractor.extract_text(img, box)

    assert result.jp_ocr_text == "ABC123"


def test_extract_jp_text_from_page(mock_manga_ocr) -> None:
    mock_manga_ocr.return_value = "テスト"
    extractor = MangaOcrExtractor()
    extractor.load()

    img = Image.new("RGB", (100, 100), color="white")
    box = BoundingBox(10, 10, 50, 50)
    result = extract_jp_text_from_page(img, box, extractor)

    assert result.jp_ocr_text == "テスト"
