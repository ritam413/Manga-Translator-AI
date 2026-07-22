"""Tests for the finetuning dataset extractor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from finetuning.checkpoint import load_checkpoint, save_checkpoint
from finetuning.extractor import DatasetExtractor
from finetuning.models import DetectedRegion, ExtractionRecord, ExtractionState
from core.models import BoundingBox


def _make_png(path: Path, size: tuple[int, int] = (1, 1)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if size == (1, 1):
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    else:
        img = Image.new("RGB", size, color="white")
        img.save(path)


def _matched_rows(
    pair_id: str = "manga1/1.png",
    manga_id: str = "manga1",
    count: int = 1,
) -> list[dict]:
    rows = []
    for idx in range(count):
        rows.append(
            {
                "pair_id": pair_id,
                "manga_id": manga_id,
                "page": "1",
                "region_index": idx,
                "status": "matched",
                "iou": 0.9,
                "jp_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
                "en_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
                "jp_text_detected": "jp",
                "en_text_detected": "en",
                "raw_path": "raw",
                "eng_path": "eng",
                "font_size": 12,
                "padding": 4,
                "line_spacing": 1.0,
                "wrap_shape": "rectangular",
                "line_count": 1,
                "typesetting_confidence": 1.0,
                "mask_path": f"masks/{manga_id}_1_region{idx}.png",
                "mask_roi_box": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
                "mask_confidence": 1.0,
                "mask_pixel_count": 100,
                "jp_ocr_text": "こんにちは",
                "jp_ocr_confidence": 1.0,
                "page_number": 1,
                "story_position": f"{manga_id}::0001::{idx:02d}",
            }
        )
    return rows


def _extractor(tmp_path: Path, **kwargs) -> DatasetExtractor:
    defaults = {
        "raw_dir": tmp_path / "raw",
        "eng_dir": tmp_path / "eng",
        "output_file": tmp_path / "training_data" / "dataset.jsonl",
        "checkpoint_file": tmp_path / "training_data" / "extraction_state.json",
        "checkpoint_interval": 50,
        "gpu": False,
    }
    defaults.update(kwargs)
    return DatasetExtractor(**defaults)


class TestFindImagePairs:
    def test_find_pairs_nested(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "manga1" / "1.png"
        eng = tmp_path / "eng" / "manga1" / "1.png"
        _make_png(raw)
        _make_png(eng)

        extractor = _extractor(tmp_path, raw_dir=tmp_path / "raw", eng_dir=tmp_path / "eng")
        pairs = extractor.find_image_pairs()

        assert len(pairs) == 1
        assert pairs[0][2] == "manga1"
        assert pairs[0][3] == "1.png"

    def test_find_pairs_flat(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "1.png"
        eng = tmp_path / "eng" / "1.png"
        _make_png(raw)
        _make_png(eng)

        extractor = _extractor(tmp_path, raw_dir=tmp_path / "raw", eng_dir=tmp_path / "eng")
        pairs = extractor.find_image_pairs()

        assert len(pairs) == 1
        assert pairs[0][2] == "default"
        assert pairs[0][3] == "1.png"

    def test_find_pairs_missing_eng_skipped(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "raw" / "manga1" / "1.png")

        extractor = _extractor(tmp_path, raw_dir=tmp_path / "raw", eng_dir=tmp_path / "eng")
        assert extractor.find_image_pairs() == []

    def test_find_pairs_sorted(self, tmp_path: Path) -> None:
        for name in ("2.png", "1.png"):
            _make_png(tmp_path / "raw" / "manga1" / name)
            _make_png(tmp_path / "eng" / "manga1" / name)

        extractor = _extractor(tmp_path, raw_dir=tmp_path / "raw", eng_dir=tmp_path / "eng")
        pairs = extractor.find_image_pairs()

        assert [p[3] for p in pairs] == ["1.png", "2.png"]


class TestCheckpoint:
    def test_atomic_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = ExtractionState(
            processed_pairs={"manga1/1.png", "manga1/2.png"},
            failed_pairs={"manga1/3.png": "read error"},
            total_processed=2,
        )

        save_checkpoint(state, path)
        loaded = load_checkpoint(path)

        assert loaded.processed_pairs == state.processed_pairs
        assert loaded.failed_pairs == state.failed_pairs
        assert loaded.total_processed == 2

    def test_corrupt_checkpoint_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        path.write_text("{not valid json", encoding="utf-8")

        with pytest.raises(ValueError, match="Corrupt checkpoint"):
            load_checkpoint(path)


class TestExtraction:
    def test_jsonl_appended_per_pair(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "raw" / "manga1" / "1.png")
        _make_png(tmp_path / "eng" / "manga1" / "1.png")

        output = tmp_path / "training_data" / "dataset.jsonl"
        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            output_file=output,
            checkpoint_interval=1,
        )
        extractor.process_pair = MagicMock(return_value=_matched_rows(count=2))  # type: ignore[method-assign]

        result = extractor.extract()

        assert result["processed"] == 1
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert all(r["status"] == "matched" for r in records)
        assert records[0]["pair_id"] == "manga1/1.png"

    def test_checkpoint_resume_skips_processed(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "raw" / "1.png")
        _make_png(tmp_path / "eng" / "1.png")

        output = tmp_path / "training_data" / "dataset.jsonl"
        checkpoint = tmp_path / "training_data" / "extraction_state.json"

        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            output_file=output,
            checkpoint_file=checkpoint,
        )
        extractor.process_pair = MagicMock(return_value=_matched_rows(pair_id="default/1.png", manga_id="default"))  # type: ignore[method-assign]
        first = extractor.extract()
        assert first["processed"] == 1

        second_extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            output_file=output,
            checkpoint_file=checkpoint,
        )
        second = second_extractor.extract()

        assert second["processed"] == 0
        assert second["skipped"] == 1
        assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 1

    def test_checkpoint_interval(self, tmp_path: Path) -> None:
        for i in (1, 2, 3):
            _make_png(tmp_path / "raw" / f"{i}.png")
            _make_png(tmp_path / "eng" / f"{i}.png")

        checkpoint = tmp_path / "training_data" / "extraction_state.json"
        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            checkpoint_file=checkpoint,
            checkpoint_interval=2,
        )
        extractor.process_pair = MagicMock(return_value=_matched_rows(pair_id="default/x.png", manga_id="default"))  # type: ignore[method-assign]

        with patch("finetuning.extractor.save_checkpoint") as mock_save:
            extractor.extract()
            assert mock_save.call_count == 2

    def test_failed_pair_continues_batch(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "raw" / "1.png")
        _make_png(tmp_path / "eng" / "1.png")
        _make_png(tmp_path / "raw" / "2.png")
        _make_png(tmp_path / "eng" / "2.png")

        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            checkpoint_interval=1,
        )

        def flaky_process(raw_path, eng_path, manga_name, img_name, reader=None):
            if img_name == "1.png":
                raise RuntimeError("simulated failure")
            return _matched_rows(pair_id=f"default/{img_name}", manga_id="default")

        extractor.process_pair = flaky_process  # type: ignore[method-assign]
        result = extractor.extract()

        assert result["processed"] == 1
        assert result["failed"] == 1
        assert "default/1.png" in extractor.state.failed_pairs

    def test_reset_checkpoint_keeps_dataset(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "training_data" / "extraction_state.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('{"processed_pairs": ["default/1.png"]}', encoding="utf-8")

        output = tmp_path / "training_data" / "dataset.jsonl"
        output.write_text("existing row\n", encoding="utf-8")

        _make_png(tmp_path / "raw" / "1.png")
        _make_png(tmp_path / "eng" / "1.png")

        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            checkpoint_file=checkpoint,
            output_file=output,
            reset=True,
        )
        extractor.process_pair = MagicMock(return_value=_matched_rows(pair_id="default/1.png", manga_id="default"))  # type: ignore[method-assign]
        result = extractor.extract()

        assert result["processed"] == 1
        assert result["skipped"] == 0

        # Verify that output file still has the existing row (appended, not overwritten)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "existing row"

    def test_reset_checkpoint_reset_overwrites_dataset(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "training_data" / "extraction_state.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text('{"processed_pairs": ["default/1.png"]}', encoding="utf-8")

        output = tmp_path / "training_data" / "dataset.jsonl"
        output.write_text("existing row\n", encoding="utf-8")

        _make_png(tmp_path / "raw" / "1.png")
        _make_png(tmp_path / "eng" / "1.png")

        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
            checkpoint_file=checkpoint,
            output_file=output,
            reset_checkpoint_reset=True,
        )
        extractor.process_pair = MagicMock(return_value=_matched_rows(pair_id="default/1.png", manga_id="default"))  # type: ignore[method-assign]
        result = extractor.extract()

        assert result["processed"] == 1
        assert result["skipped"] == 0

        # Verify that output file was overwritten (does not contain the "existing row" anymore)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert lines[0] != "existing row"

    def test_dimension_mismatch_fails_pair(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "raw" / "1.png", size=(100, 100))
        _make_png(tmp_path / "eng" / "1.png", size=(200, 200))

        extractor = _extractor(
            tmp_path,
            raw_dir=tmp_path / "raw",
            eng_dir=tmp_path / "eng",
        )
        result = extractor.extract()

        assert result["processed"] == 0
        assert result["failed"] == 1
        assert "default/1.png" in extractor.state.failed_pairs
        assert "Dimension mismatch" in extractor.state.failed_pairs["default/1.png"]


class TestProcessPair:
    @patch("finetuning.extractor.MangaOcrExtractor")
    @patch("finetuning.extractor.PageOrderTracker")
    @patch("finetuning.extractor.save_region_mask")
    @patch("finetuning.extractor.extract_inpaint_mask_from_pages")
    @patch("finetuning.extractor.extract_typesetting_features")
    @patch("finetuning.extractor.detect_text_regions")
    def test_process_pair_produces_matches(
        self,
        mock_detect,
        mock_extract_features,
        mock_extract_mask,
        mock_save_mask,
        mock_tracker_cls,
        mock_ocr_extractor_cls,
        tmp_path: Path,
    ) -> None:
        raw = tmp_path / "raw" / "1.png"
        eng = tmp_path / "eng" / "1.png"
        _make_png(raw, size=(100, 100))
        _make_png(eng, size=(100, 100))

        # Mock MangaOcrExtractor
        mock_ocr = MagicMock()
        mock_ocr.is_loaded = True
        from finetuning.models import JpOcrResult
        mock_ocr.extract_text.return_value = JpOcrResult(
            jp_ocr_text="テスト", jp_ocr_confidence=1.0
        )
        mock_ocr_extractor_cls.return_value = mock_ocr

        # Mock PageOrderTracker
        mock_tracker = MagicMock()
        mock_tracker.get_page_number.return_value = 1
        mock_tracker.get_story_position.return_value = "default::0001::00"
        mock_tracker_cls.return_value = mock_tracker

        box = BoundingBox(10, 10, 50, 50)
        mock_detect.side_effect = [
            [DetectedRegion(box=box, text="こんにちは", confidence=0.9)],
            [DetectedRegion(box=BoundingBox(12, 12, 48, 48), text="HELLO", confidence=0.9)],
        ]
        
        from finetuning.models import TypesettingFeatures
        mock_extract_features.return_value = TypesettingFeatures(
            font_size=16,
            padding=5,
            line_spacing=1.2,
            wrap_shape="rectangular",
            line_count=2,
            typesetting_confidence=0.9,
        )

        import numpy as np
        from finetuning.mask_extraction import InpaintMaskResult
        mock_extract_mask.return_value = InpaintMaskResult(
            mask=np.zeros((10, 10), dtype=np.uint8),
            mask_confidence=0.85,
            mask_pixel_count=120,
            mask_roi_box=BoundingBox(5, 5, 55, 55)
        )
        mock_save_mask.return_value = "masks/mock_path.png"

        extractor = _extractor(tmp_path, raw_dir=tmp_path / "raw", eng_dir=tmp_path / "eng")
        extractor._page_tracker = mock_tracker
        extractor._manga_ocr_extractor = mock_ocr
        rows = extractor.process_pair(raw, eng, "default", "1.png", reader=MagicMock())

        assert len(rows) == 1
        assert rows[0]["status"] == "matched"
        assert rows[0]["jp_text_detected"] == "こんにちは"
        assert rows[0]["en_text_detected"] == "HELLO"
        assert rows[0]["font_size"] == 16
        assert rows[0]["padding"] == 5
        assert rows[0]["line_spacing"] == 1.2
        assert rows[0]["wrap_shape"] == "rectangular"
        assert rows[0]["mask_path"] == "masks/mock_path.png"
        assert rows[0]["mask_roi_box"] == {"x_min": 5, "y_min": 5, "x_max": 55, "y_max": 55}
        assert rows[0]["mask_confidence"] == 0.85
        assert rows[0]["mask_pixel_count"] == 120
        assert rows[0]["jp_ocr_text"] == "テスト"
        assert rows[0]["jp_ocr_confidence"] == 1.0
        assert rows[0]["page_number"] == 1
        assert rows[0]["story_position"] == "default::0001::00"

    def test_no_manga_ocr_flag_produces_fallback(self, tmp_path: Path) -> None:
        """With use_manga_ocr=False, jp_ocr_text should be empty string."""
        extractor = _extractor(tmp_path, use_manga_ocr=False)
        assert extractor._manga_ocr_extractor is None

    def test_manga_ocr_lifecycle_unload_on_checkpoint(self, tmp_path: Path) -> None:
        for i in range(1, 4):
            _make_png(tmp_path / "raw" / f"{i}.png", size=(10, 10))
            _make_png(tmp_path / "eng" / f"{i}.png", size=(10, 10))

        with patch("finetuning.extractor.MangaOcrExtractor") as mock_cls:
            mock_ocr = MagicMock()
            mock_ocr.is_loaded = True
            mock_cls.return_value = mock_ocr

            extractor = _extractor(tmp_path, checkpoint_interval=2, use_manga_ocr=True)
            extractor.process_pair = MagicMock(return_value=_matched_rows(count=1))
            extractor.extract()

            # Unload called at least once (at checkpoint boundary)
            assert mock_ocr.unload.call_count >= 1


class TestExtractionRecord:
    def test_stub_record(self) -> None:
        record = ExtractionRecord.stub(
            raw_path="/raw/manga1/1.jpg",
            eng_path="/eng/manga1/1.jpg",
            manga_name="manga1",
            img_name="1.jpg",
        )
        data = record.to_dict()
        assert data["page"] == "1"
        assert data["manga_id"] == "manga1"

    def test_page_from_complex_filename(self) -> None:
        record = ExtractionRecord.stub(
            raw_path="/a",
            eng_path="/b",
            manga_name="m",
            img_name="page_03.webp",
        )
        assert record.page == "page_03"


class TestDatasetExtractorVersioning:
    @patch("finetuning.extractor.detect_text_regions")
    def test_process_pair_forwards_enable_box_merge(self, mock_detect, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "1.png"
        eng = tmp_path / "eng" / "1.png"
        _make_png(raw, size=(100, 100))
        _make_png(eng, size=(100, 100))

        mock_detect.return_value = []
        reader_mock = MagicMock()

        # Test model_version v2 (merging enabled)
        extractor_v2 = _extractor(tmp_path, model_version="v2")
        with pytest.raises(ValueError, match="No region matches"):
            extractor_v2.process_pair(raw, eng, "default", "1.png", reader=reader_mock)
        
        # Verify call arguments
        mock_detect.assert_any_call(raw, ["ja"], reader=reader_mock, gpu=False, enable_box_merge=True)
        mock_detect.assert_any_call(eng, ["en"], reader=reader_mock, gpu=False, enable_box_merge=True)

        # Test model_version v1 (merging disabled)
        mock_detect.reset_mock()
        extractor_v1 = _extractor(tmp_path, model_version="v1")
        with pytest.raises(ValueError, match="No region matches"):
            extractor_v1.process_pair(raw, eng, "default", "1.png", reader=reader_mock)

        # Verify call arguments
        mock_detect.assert_any_call(raw, ["ja"], reader=reader_mock, gpu=False, enable_box_merge=False)
        mock_detect.assert_any_call(eng, ["en"], reader=reader_mock, gpu=False, enable_box_merge=False)
