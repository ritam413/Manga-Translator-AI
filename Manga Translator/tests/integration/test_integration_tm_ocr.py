import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.ocr_worker import OcrWorker
from finetuning.pipeline_integration import correct_ocr_with_vocab


def test_ocr_worker_loads_tm_db(synthetic_tm_db):
    """Verify OcrWorker constructor loads and registers TM database."""
    worker = OcrWorker(input_dir="/input", tm_db=synthetic_tm_db, enable_tm_correction=True)
    assert worker._tm_db is synthetic_tm_db
    assert worker._enable_tm_correction is True


@pytest.mark.parametrize(
    "raw_ocr,expected_corrected",
    [
        ("ルフイが走る", "ルフィが走る"),
        ("コニチハ、世界", "こんにちは、世界"),
        ("ゾロ！", "ゾロ！"),
        ("海賊オウだ", "海賊王だ"),
        ("ゾロとルフイ", "ゾロとルフィ"),
        ("全然違うテキスト", "全然違うテキスト"), # No match, no correction
    ]
)
def test_correct_ocr_with_vocab_isolated(synthetic_tm_db, raw_ocr, expected_corrected):
    """Verify spell correction behaves correctly on isolated tokens."""
    corrected = correct_ocr_with_vocab(synthetic_tm_db, "manga1", raw_ocr)
    assert corrected == expected_corrected


def test_correct_ocr_with_vocab_wrong_manga_id(synthetic_tm_db):
    """Verify spell correction respects manga_id boundaries."""
    # "ルフイ" typo is defined in manga1 vocab (fuzzy key "ルフィ"), but not in manga2 vocab
    raw_ocr = "ルフイ！"
    corrected_m1 = correct_ocr_with_vocab(synthetic_tm_db, "manga1", raw_ocr)
    corrected_m2 = correct_ocr_with_vocab(synthetic_tm_db, "manga2", raw_ocr)
    
    assert corrected_m1 == "ルフィ！"
    assert corrected_m2 == "ルフイ！"  # Unchanged


@patch("workers.ocr_worker.Image")
@patch("workers.ocr_worker.MangaOcr")
def test_ocr_worker_correction_pipeline(mock_mocr_cls, mock_image_cls, synthetic_tm_db):
    """Verify OcrWorker correct text regions using TM vocab during process_page."""
    mock_mocr = MagicMock()
    mock_mocr_cls.return_value = mock_mocr
    
    # Mock MangaOCR return text with a spelling typo
    mock_mocr.return_value = "ルフイ"
    
    # Mock PIL Image
    mock_img = MagicMock()
    mock_image_cls.open.return_value = mock_img
    mock_img.convert.return_value = mock_img
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir)
        
        # Create image file
        img_filename = "page01.png"
        Path(os.path.join(input_dir, img_filename)).touch()
        
        worker = OcrWorker(input_dir=input_dir, tm_db=synthetic_tm_db, enable_tm_correction=True)
        worker.load()
        
        region = TextRegion(id="r00", box=BoundingBox(10, 10, 50, 50), raw_text="old_ocr")
        page = PageManifest(image_filename=img_filename, width=800, height=1200, text_regions=[region])
        manifest = ProjectManifest(project_name="Test", pages={img_filename: page}, manga_id="manga1")
        
        # Process
        updated_page = worker.process_page(page, manifest)
        
        assert updated_page.text_regions[0].raw_text == "ルフィ"  # Typo corrected to canonical
        assert updated_page.text_regions[0].ocr_confidence == 1.0  # Sentinel intact


@patch("workers.ocr_worker.Image")
@patch("workers.ocr_worker.MangaOcr")
def test_ocr_worker_correction_disabled(mock_mocr_cls, mock_image_cls, synthetic_tm_db):
    """Verify spelling correction is skipped when disabled."""
    mock_mocr = MagicMock()
    mock_mocr_cls.return_value = mock_mocr
    mock_mocr.return_value = "ルフイ"
    
    mock_img = MagicMock()
    mock_image_cls.open.return_value = mock_img
    mock_img.convert.return_value = mock_img
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir)
        
        img_filename = "page01.png"
        Path(os.path.join(input_dir, img_filename)).touch()
        
        worker = OcrWorker(input_dir=input_dir, tm_db=synthetic_tm_db, enable_tm_correction=False)
        worker.load()
        
        region = TextRegion(id="r00", box=BoundingBox(10, 10, 50, 50), raw_text="old_ocr")
        page = PageManifest(image_filename=img_filename, width=800, height=1200, text_regions=[region])
        manifest = ProjectManifest(project_name="Test", pages={img_filename: page}, manga_id="manga1")
        
        # Process
        updated_page = worker.process_page(page, manifest)
        
        assert updated_page.text_regions[0].raw_text == "ルフイ"  # Typo NOT corrected
