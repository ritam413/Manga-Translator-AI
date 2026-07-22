import os
import unittest
from unittest.mock import patch, MagicMock

from core.models import BoundingBox, TextRegion, PageManifest, ProjectManifest
from workers.ocr_worker import OcrWorker

class TestStage7OcrWorker(unittest.TestCase):
    def setUp(self):
        self.input_dir = "mock_input_dir"
        self.worker = OcrWorker(input_dir=self.input_dir)

    def test_worker_name(self):
        self.assertEqual(self.worker.worker_name, "MangaOCR Cross-Check")

    def test_initial_state(self):
        self.assertFalse(self.worker.is_loaded())
        self.assertIsNone(self.worker._mocr)

    @patch("workers.ocr_worker.MangaOcr")
    def test_load_success(self, mock_manga_ocr_cls):
        mock_instance = MagicMock()
        mock_manga_ocr_cls.return_value = mock_instance

        self.worker.load()
        
        self.assertTrue(self.worker.is_loaded())
        self.assertIsNotNone(self.worker._mocr)
        mock_manga_ocr_cls.assert_called_once()

    @patch("workers.ocr_worker.MangaOcr")
    def test_load_idempotent(self, mock_manga_ocr_cls):
        self.worker.load()
        self.worker.load()
        mock_manga_ocr_cls.assert_called_once()

    @patch("workers.ocr_worker.MangaOcr", new=None)
    def test_load_import_error(self):
        with self.assertRaises(ImportError):
            self.worker.load()

    def test_unload_success(self):
        self.worker._loaded = True
        self.worker._mocr = MagicMock()
        
        self.worker.unload()
        
        self.assertFalse(self.worker.is_loaded())
        self.assertIsNone(self.worker._mocr)

    def test_unload_idempotent(self):
        self.worker._loaded = True
        self.worker._mocr = MagicMock()
        
        self.worker.unload()
        self.worker.unload()
        
        self.assertFalse(self.worker.is_loaded())

    def test_process_page_not_loaded(self):
        page = PageManifest(image_filename="test.png", width=1000, height=1000)
        manifest = ProjectManifest(project_name="Test", pages={})
        
        with self.assertRaises(RuntimeError):
            self.worker.process_page(page, manifest)

    @patch("workers.ocr_worker.os.path.isfile")
    def test_process_page_file_not_found(self, mock_isfile):
        self.worker._loaded = True
        self.worker._mocr = MagicMock()
        mock_isfile.return_value = False
        
        page = PageManifest(image_filename="test.png", width=1000, height=1000)
        manifest = ProjectManifest(project_name="Test", pages={})
        
        with self.assertRaises(FileNotFoundError):
            self.worker.process_page(page, manifest)

    @patch("workers.ocr_worker.Image")
    @patch("workers.ocr_worker.os.path.isfile")
    def test_process_page_success(self, mock_isfile, mock_image):
        self.worker._loaded = True
        
        mock_mocr = MagicMock()
        mock_mocr.side_effect = ["High Quality Japanese 1", "High Quality Japanese 2"]
        self.worker._mocr = mock_mocr
        
        mock_isfile.return_value = True
        
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        page = PageManifest(image_filename="test.png", width=1000, height=1000)
        page.text_regions = [
            TextRegion(id="r0000", box=BoundingBox(10, 10, 50, 50), raw_text="rough1", ocr_confidence=0.5),
            TextRegion(id="r0001", box=BoundingBox(100, 100, 150, 150), raw_text="rough2", ocr_confidence=0.6)
        ]
        manifest = ProjectManifest(project_name="Test", pages={})
        
        result_page = self.worker.process_page(page, manifest)
        
        mock_image.open.assert_called_once_with(os.path.join(self.input_dir, "test.png"))
        self.assertEqual(mock_img.crop.call_count, 2)
        mock_img.crop.assert_any_call((10, 10, 50, 50))
        mock_img.crop.assert_any_call((100, 100, 150, 150))
        
        self.assertEqual(mock_mocr.call_count, 2)
        
        self.assertEqual(result_page.text_regions[0].raw_text, "High Quality Japanese 1")
        self.assertEqual(result_page.text_regions[0].ocr_confidence, 1.0)
        self.assertEqual(result_page.text_regions[1].raw_text, "High Quality Japanese 2")
        self.assertEqual(result_page.text_regions[1].ocr_confidence, 1.0)
        self.assertIs(result_page, page)

    @patch("workers.ocr_worker.Image")
    @patch("workers.ocr_worker.os.path.isfile")
    def test_process_page_zero_area_box(self, mock_isfile, mock_image):
        self.worker._loaded = True
        mock_mocr = MagicMock()
        self.worker._mocr = mock_mocr
        mock_isfile.return_value = True
        
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        page = PageManifest(image_filename="test.png", width=1000, height=1000)
        page.text_regions = [
            TextRegion(id="r0000", box=BoundingBox(10, 10, 10, 10), raw_text="rough1", ocr_confidence=0.5)
        ]
        manifest = ProjectManifest(project_name="Test", pages={})
        
        self.worker.process_page(page, manifest)
        
        mock_img.crop.assert_not_called()
        mock_mocr.assert_not_called()
        self.assertEqual(page.text_regions[0].raw_text, "rough1")
        self.assertEqual(page.text_regions[0].ocr_confidence, 0.5)

    @patch("workers.ocr_worker.Image")
    @patch("workers.ocr_worker.os.path.isfile")
    def test_process_page_exception_caught(self, mock_isfile, mock_image):
        self.worker._loaded = True
        mock_mocr = MagicMock()
        # First crop raises exception, second succeeds
        mock_mocr.side_effect = [Exception("Crop failed"), "High Quality Japanese 2"]
        self.worker._mocr = mock_mocr
        mock_isfile.return_value = True
        
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image.open.return_value = mock_img
        
        page = PageManifest(image_filename="test.png", width=1000, height=1000)
        page.text_regions = [
            TextRegion(id="r0000", box=BoundingBox(10, 10, 50, 50), raw_text="rough1", ocr_confidence=0.5),
            TextRegion(id="r0001", box=BoundingBox(100, 100, 150, 150), raw_text="rough2", ocr_confidence=0.6)
        ]
        manifest = ProjectManifest(project_name="Test", pages={})
        
        # Should not raise exception
        result_page = self.worker.process_page(page, manifest)
        
        self.assertEqual(mock_img.crop.call_count, 2)
        self.assertEqual(mock_mocr.call_count, 2)
        
        # First region unchanged
        self.assertEqual(result_page.text_regions[0].raw_text, "rough1")
        self.assertEqual(result_page.text_regions[0].ocr_confidence, 0.5)
        
        # Second region updated
        self.assertEqual(result_page.text_regions[1].raw_text, "High Quality Japanese 2")
        self.assertEqual(result_page.text_regions[1].ocr_confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
