import os
import pytest
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from core.manifest import ManifestEngine
from stages.controller import StageController
from workers.text_detector import TextDetectorWorker
from workers.ocr_worker import OcrWorker
from workers.translator import TranslationWorker
from workers.inpainter import InpainterWorker
from workers.typesetter import TypesetterWorker
from finetuning.pipeline_integration import load_layout_model


@patch("workers.text_detector.easyocr")
@patch("workers.ocr_worker.Image")
@patch("workers.ocr_worker.MangaOcr")
@patch("workers.translator.GoogleTranslator")
def test_full_phase_a_pipeline_integration(mock_google_cls, mock_mocr_cls, mock_image_cls, mock_easyocr, synthetic_tm_db):
    """Verify end-to-end integration of Phase A stages (detector -> ocr -> translator) with TM enabled."""
    # 1. Mock Text Detector
    mock_reader = MagicMock()
    mock_easyocr.Reader.return_value = mock_reader
    mock_reader.readtext.return_value = [
        ([[10, 10], [100, 10], [100, 100], [10, 100]], "こんにちは", 0.95)
    ]
    
    # 2. Mock OCR worker
    mock_mocr = MagicMock()
    mock_mocr_cls.return_value = mock_mocr
    mock_mocr.return_value = "こんにちは！"  # Will match fuzzy TM: "こんにちは" -> "Hello"
    
    mock_img = MagicMock()
    mock_image_cls.open.return_value = mock_img
    mock_img.convert.return_value = mock_img
    
    # 3. Mock Google (should not be called due to TM hit)
    mock_google = MagicMock()
    mock_google_cls.return_value = mock_google
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir)
        img_filename = "page01.png"
        img_path = os.path.join(input_dir, img_filename)
        from PIL import Image
        Image.new("RGB", (800, 1200), "white").save(img_path)
        
        with patch("finetuning.pipeline_integration.load_translation_memory", return_value=synthetic_tm_db):
            from cli.config.models import CliConfig
            config = CliConfig()
            config.finetuning.enabled = True
            
            # Setup workspace manifest
            manifest = ProjectManifest(project_name="Test", pages={
                img_filename: PageManifest(img_filename, 800, 1200)
            }, manga_id="manga1")
            
            engine = ManifestEngine(project_dir=tmp_dir)
            engine.manifest = manifest
            
            # Stage 6: Detect text
            detector = TextDetectorWorker(input_dir=input_dir)
            controller_detect = StageController(engine, detector, "detect", checks_phase="a")
            controller_detect.run()
            
            assert len(manifest.pages[img_filename].text_regions) == 1
            assert manifest.pages[img_filename].text_regions[0].region_index == 0
            
            # Stage 7: OCR
            ocr = OcrWorker(input_dir=input_dir, tm_db=synthetic_tm_db, enable_tm_correction=True)
            controller_ocr = StageController(engine, ocr, "ocr", checks_phase="a")
            controller_ocr.run()
            
            # Spelling correction checks: "こんにちは！" has no typo, should remain "こんにちは！"
            assert manifest.pages[img_filename].text_regions[0].raw_text == "こんにちは！"
            
            # Stage 8: Translate
            translator = TranslationWorker(prefer_offline=False, tm_db=synthetic_tm_db)
            controller_trans = StageController(engine, translator, "translate", marks_phase="a", checks_phase="a")
            controller_trans.run()
            
            # The raw text "こんにちは！" fuzzy matches "こんにちは" in manga1 TM -> "Hello"
            assert manifest.pages[img_filename].text_regions[0].translated_text == "Hello"
            assert manifest.pages[img_filename].text_regions[0].translation_source == "tm_fuzzy"
            assert manifest.pages[img_filename].phase_a_completed is True


@patch("workers.inpainter.cv2.inpaint")
@patch("workers.inpainter.cv2.imwrite")
@patch("workers.inpainter.os.replace")
@patch("workers.typesetter.Image")
@patch("workers.typesetter.ImageDraw")
@patch("workers.typesetter.os.replace")
@patch("workers.typesetter.os.path.isfile", return_value=True)
def test_full_phase_b_pipeline_integration(
    mock_typeset_isfile, mock_typeset_replace, mock_typeset_draw_cls, mock_typeset_image_cls,
    mock_inpaint_replace, mock_inpaint_imwrite, mock_inpaint_inpaint,
    synthetic_layout_model_path
):
    """Verify integration of Phase B stages (inpainter -> typesetter) with layout model and contour inpaint."""
    mock_typeset_img = MagicMock()
    mock_typeset_canvas = MagicMock()
    mock_typeset_img.copy.return_value = mock_typeset_canvas
    mock_typeset_canvas.size = (800, 1200)
    mock_typeset_image_cls.open.return_value = mock_typeset_img
    mock_typeset_draw = MagicMock()
    mock_typeset_draw_cls.Draw.return_value = mock_typeset_draw
    
    # 1. Load layout model
    layout_model = load_layout_model(synthetic_layout_model_path)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(output_dir)
        
        img_filename = "page01.png"
        img_path = os.path.join(input_dir, img_filename)
        from PIL import Image
        Image.new("RGB", (800, 1200), "white").save(img_path)
        
        manifest = ProjectManifest(project_name="Test", pages={
            img_filename: PageManifest(
                image_filename=img_filename,
                width=800,
                height=1200,
                text_regions=[
                    TextRegion("r0", BoundingBox(10, 10, 110, 110), raw_text="こんにちは", translated_text="Hello", region_index=0)
                ],
                phase_a_completed=True
            )
        }, manga_id="manga1")
        
        engine = ManifestEngine(project_dir=tmp_dir)
        engine.manifest = manifest
        
        # Run Stage 9: Inpaint
        inpainter = InpainterWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            use_contour_inpaint=True,
            layout_model_min_confidence=0.5
        )
        controller_inpaint = StageController(engine, inpainter, "inpaint", checks_phase="b")
        # Stub reading raw image in inpainter to avoid file lookups
        with patch("workers.inpainter.cv2.imread", return_value=np.ones((1200, 800, 3), dtype=np.uint8)*255):
            controller_inpaint.run()
            
        # Run Stage 10: Typeset
        typesetter = TypesetterWorker(
            input_dir=output_dir,
            output_dir=output_dir,
            layout_model=layout_model,
            layout_model_min_confidence=0.5
        )
        controller_typeset = StageController(engine, typesetter, "typeset", marks_phase="b", checks_phase="b")
        controller_typeset.run()
        
        assert manifest.pages[img_filename].phase_b_completed is True
