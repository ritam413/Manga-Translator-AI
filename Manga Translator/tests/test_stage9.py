import os
import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np

from core.models import PageManifest, ProjectManifest, TextRegion, BoundingBox
from workers.inpainter import InpainterWorker
from stages.controller import StageController

class TestStage9InpainterWorker(unittest.TestCase):
    def setUp(self):
        self.input_dir = "/fake/input"
        self.output_dir = "/fake/output"
        self.worker = InpainterWorker(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            inpaint_radius=3
        )
        
        # Dummy page and manifest
        self.page = PageManifest(
            image_filename="page01.png",
            width=1000,
            height=1500,
            text_regions=[]
        )
        self.manifest = ProjectManifest(
            project_name="Test",
            pages={"page01.png": self.page}
        )

    def test_worker_name_and_initial_state(self):
        self.assertEqual(self.worker.worker_name, "OpenCV Inpainter")
        self.assertFalse(self.worker.is_loaded())
        self.assertEqual(self.worker._inpaint_radius, 3)

    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.makedirs")
    def test_load_success(self, mock_makedirs, mock_cv2):
        self.worker.load()
        self.assertTrue(self.worker.is_loaded())
        mock_makedirs.assert_called_once_with(self.output_dir, exist_ok=True)
        
        # Test idempotency
        mock_makedirs.reset_mock()
        self.worker.load()
        mock_makedirs.assert_not_called()

    @patch("workers.inpainter.cv2", None)
    @patch("workers.inpainter.os.makedirs")
    def test_load_fails_without_cv2(self, mock_makedirs):
        with self.assertRaises(ImportError) as ctx:
            self.worker.load()
        self.assertIn("opencv-python-headless is not installed", str(ctx.exception))
        mock_makedirs.assert_not_called()
        self.assertFalse(self.worker.is_loaded())

    def test_unload(self):
        self.worker._loaded = True
        self.worker.unload()
        self.assertFalse(self.worker.is_loaded())
        
        # Test idempotency
        self.worker.unload()
        self.assertFalse(self.worker.is_loaded())

    def test_process_page_not_loaded(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.worker.process_page(self.page, self.manifest)
        self.assertIn("InpainterWorker.load() must be called before process_page", str(ctx.exception))

    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.path.isfile")
    def test_process_page_file_not_found(self, mock_isfile, mock_cv2):
        self.worker._loaded = True
        mock_isfile.return_value = False
        
        with self.assertRaises(FileNotFoundError) as ctx:
            self.worker.process_page(self.page, self.manifest)
        self.assertIn("Image not found", str(ctx.exception))

    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.path.isfile")
    def test_process_page_imread_fails(self, mock_isfile, mock_cv2):
        self.worker._loaded = True
        mock_isfile.return_value = True
        mock_cv2.imread.return_value = None
        
        with self.assertRaises(RuntimeError) as ctx:
            self.worker.process_page(self.page, self.manifest)
        self.assertIn("cv2.imread failed for", str(ctx.exception))

    @patch("workers.inpainter.np.zeros")
    @patch("workers.inpainter.os.replace")
    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.path.isfile")
    def test_process_page_success_zero_regions(self, mock_isfile, mock_cv2, mock_replace, mock_np_zeros):
        self.worker._loaded = True
        mock_isfile.return_value = True
        
        # Mock image
        mock_img = MagicMock()
        mock_img.shape = (1500, 1000, 3)
        mock_cv2.imread.return_value = mock_img
        
        # Mock mask
        mock_mask = MagicMock()
        mock_np_zeros.return_value = mock_mask
        
        # Mock inpaint result
        mock_inpainted = MagicMock()
        mock_cv2.inpaint.return_value = mock_inpainted
        
        mock_cv2.INPAINT_TELEA = 1
        
        # Call
        result = self.worker.process_page(self.page, self.manifest)
        
        self.assertIs(result, self.page)
        mock_cv2.imread.assert_called_once_with(os.path.join(self.input_dir, "page01.png"))
        mock_np_zeros.assert_called_once_with((1500, 1000), dtype=np.uint8)
        mock_cv2.rectangle.assert_not_called()
        mock_cv2.inpaint.assert_called_once_with(mock_img, mock_mask, 3, mock_cv2.INPAINT_TELEA)
        
        tmp_path = os.path.join(self.output_dir, "page01.png.tmp.png")
        out_path = os.path.join(self.output_dir, "page01.png")
        
        mock_cv2.imwrite.assert_called_once_with(tmp_path, mock_inpainted)
        mock_replace.assert_called_once_with(tmp_path, out_path)

    @patch("workers.inpainter.np.zeros")
    @patch("workers.inpainter.os.replace")
    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.path.isfile")
    def test_process_page_success_with_regions(self, mock_isfile, mock_cv2, mock_replace, mock_np_zeros):
        self.worker._loaded = True
        mock_isfile.return_value = True
        
        # Add regions
        self.page.text_regions = [
            TextRegion(id="r1", box=BoundingBox(10, 20, 100, 200)),
            TextRegion(id="r2", box=BoundingBox(-10, -20, 1100, 1600)), # Test clamping
            TextRegion(id="r3", box=BoundingBox(200, 200, 100, 100))    # Invalid box, shouldn't draw
        ]
        
        # Mock image
        mock_img = MagicMock()
        mock_img.shape = (1500, 1000, 3)
        mock_cv2.imread.return_value = mock_img
        
        # Mock mask
        mock_mask = MagicMock()
        mock_np_zeros.return_value = mock_mask
        
        mock_cv2.INPAINT_TELEA = 1
        
        # Call
        result = self.worker.process_page(self.page, self.manifest)
        
        self.assertIs(result, self.page)
        
        # Check rectangle calls
        self.assertEqual(mock_cv2.rectangle.call_count, 2)
        mock_cv2.rectangle.assert_has_calls([
            call(mock_mask, (10, 20), (100, 200), 255, -1),
            call(mock_mask, (0, 0), (1000, 1500), 255, -1)
        ], any_order=True)

    @patch("workers.inpainter.np.zeros")
    @patch("workers.inpainter.os.replace")
    @patch("workers.inpainter.cv2", create=True)
    @patch("workers.inpainter.os.path.isfile")
    def test_process_page_swallows_drawing_exception(self, mock_isfile, mock_cv2, mock_replace, mock_np_zeros):
        self.worker._loaded = True
        mock_isfile.return_value = True
        
        self.page.text_regions = [
            TextRegion(id="r1", box=BoundingBox(10, 20, 100, 200))
        ]
        
        mock_img = MagicMock()
        mock_img.shape = (1500, 1000, 3)
        mock_cv2.imread.return_value = mock_img
        
        # Make rectangle raise exception
        mock_cv2.rectangle.side_effect = Exception("Drawing failed")
        mock_cv2.INPAINT_TELEA = 1
        
        # Should not raise exception
        result = self.worker.process_page(self.page, self.manifest)
        self.assertIs(result, self.page)

class TestStage9ControllerIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_worker = MagicMock(spec=InpainterWorker)
        self.mock_worker.worker_name = "Mock Inpainter"
        
        self.mock_engine = MagicMock()
        
        self.manifest = ProjectManifest(
            project_name="Test",
            pages={
                "page01.png": PageManifest(image_filename="page01.png", width=1000, height=1500, phase_b_completed=False),
                "page02.png": PageManifest(image_filename="page02.png", width=1000, height=1500, phase_b_completed=True)
            }
        )
        self.mock_engine.manifest = self.manifest
        
        self.controller = StageController(
            engine=self.mock_engine,
            stage_name="Stage 9 Inpainter",
            worker=self.mock_worker,
            marks_phase="b"
        )
        
        # Make process_page just return the page
        def process_page_side_effect(page, manifest):
            return page
        self.mock_worker.process_page.side_effect = process_page_side_effect

    def test_controller_integration(self):
        # Run with force=False
        self.controller.force = False
        report = self.controller.run()
        
        self.mock_worker.load.assert_called_once()
        self.mock_worker.unload.assert_called_once()
        
        # Only page01 should be processed because page02 is phase_b_completed=True
        self.mock_worker.process_page.assert_called_once_with(self.manifest.pages["page01.png"], self.manifest)
        
        # Engine saves manifest
        self.mock_engine.save_manifest.assert_called_once()
        
        # phase_b_completed is set
        self.assertTrue(self.manifest.pages["page01.png"].phase_b_completed)
        self.assertTrue(self.manifest.pages["page02.png"].phase_b_completed)

    def test_controller_integration_force(self):
        # Run with force=True
        self.controller.force = True
        report = self.controller.run()
        
        self.mock_worker.load.assert_called_once()
        self.mock_worker.unload.assert_called_once()
        
        # Both pages should be processed
        self.assertEqual(self.mock_worker.process_page.call_count, 2)
        
        # Engine saves manifest twice
        self.assertEqual(self.mock_engine.save_manifest.call_count, 2)

if __name__ == '__main__':
    unittest.main()
