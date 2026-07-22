"""Tests for finetuning EasyOCR detection wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.models import BoundingBox
from finetuning.detection import detect_text_regions


def _make_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class TestDetectTextRegions:
    def test_quad_to_bounding_box(self, tmp_path: Path) -> None:
        image = tmp_path / "page.png"
        _make_png(image)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [110, 20], [110, 80], [10, 80]], "hello", 0.95),
        ]

        regions = detect_text_regions(image, ["en"], reader=mock_reader)

        assert len(regions) == 1
        assert regions[0].box == BoundingBox(10, 20, 110, 80)
        assert regions[0].text == "hello"
        assert regions[0].confidence == 0.95
        mock_reader.readtext.assert_called_once_with(str(image))

    @patch("finetuning.detection._easyocr")
    def test_creates_reader_when_none(self, mock_easyocr_module, tmp_path: Path) -> None:
        image = tmp_path / "page.png"
        _make_png(image)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_easyocr_module.Reader.return_value = mock_reader

        regions = detect_text_regions(image, ["ja"], reader=None, gpu=False)

        assert regions == []
        mock_easyocr_module.Reader.assert_called_once_with(["ja"], gpu=False)
