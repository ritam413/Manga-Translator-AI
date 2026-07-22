"""
tests/test_stage6.py — Unit tests for Stage 6: Phase A Text Detection Layer

All tests are mock-based. No GPU or real EasyOCR installation is required.
"""

import os
import unittest
from unittest.mock import MagicMock, call, patch

from core.models import BoundingBox, PageManifest, ProjectManifest, TextRegion
from workers.text_detector import TextDetectorWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(filename="page01.png", regions=None):
    """Create a minimal PageManifest for testing."""
    return PageManifest(
        image_filename=filename,
        width=800,
        height=1200,
        text_regions=regions or [],
    )


def _make_manifest():
    """Create a minimal ProjectManifest for testing."""
    return ProjectManifest(
        project_name="Test Manga",
        pages={},
    )


def _easyocr_result(x1, y1, x2, y2, text="テスト", confidence=0.95):
    """Build a fake EasyOCR result tuple: (bbox_quad, text, confidence)."""
    bbox_quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return (bbox_quad, text, confidence)


# ---------------------------------------------------------------------------
# Test: Properties and basic construction
# ---------------------------------------------------------------------------

class TestTextDetectorWorkerConstruction(unittest.TestCase):

    def test_worker_name(self):
        w = TextDetectorWorker(input_dir="/some/dir")
        self.assertEqual(w.worker_name, "EasyOCR Text Detector")

    def test_default_languages(self):
        w = TextDetectorWorker(input_dir="/some/dir")
        self.assertEqual(w._languages, ["ja", "en"])

    def test_custom_languages(self):
        w = TextDetectorWorker(input_dir="/some/dir", languages=["ja"])
        self.assertEqual(w._languages, ["ja"])

    def test_default_gpu_true(self):
        w = TextDetectorWorker(input_dir="/some/dir")
        self.assertTrue(w._gpu)

    def test_gpu_false(self):
        w = TextDetectorWorker(input_dir="/some/dir", gpu=False)
        self.assertFalse(w._gpu)

    def test_initially_not_loaded(self):
        w = TextDetectorWorker(input_dir="/some/dir")
        self.assertFalse(w.is_loaded())
        self.assertIsNone(w._reader)


# ---------------------------------------------------------------------------
# Test: load() behaviour
# ---------------------------------------------------------------------------

class TestTextDetectorWorkerLoad(unittest.TestCase):

    @patch("workers.text_detector.easyocr")
    def test_load_creates_reader(self, mock_easyocr):
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader

        w = TextDetectorWorker(input_dir="/some/dir", gpu=True)
        w.load()

        mock_easyocr.Reader.assert_called_once_with(["ja", "en"], gpu=True)
        self.assertTrue(w.is_loaded())
        self.assertIs(w._reader, mock_reader)

    @patch("workers.text_detector.easyocr")
    def test_load_is_idempotent(self, mock_easyocr):
        """Calling load() twice must not create two Reader instances."""
        mock_easyocr.Reader.return_value = MagicMock()

        w = TextDetectorWorker(input_dir="/some/dir")
        w.load()
        w.load()  # second call — should be a no-op

        mock_easyocr.Reader.assert_called_once()

    @patch("workers.text_detector.easyocr")
    def test_load_passes_gpu_flag(self, mock_easyocr):
        mock_easyocr.Reader.return_value = MagicMock()

        w = TextDetectorWorker(input_dir="/some/dir", gpu=False)
        w.load()

        mock_easyocr.Reader.assert_called_once_with(["ja", "en"], gpu=False)


# ---------------------------------------------------------------------------
# Test: unload() behaviour
# ---------------------------------------------------------------------------

class TestTextDetectorWorkerUnload(unittest.TestCase):

    @patch("workers.text_detector.easyocr")
    def test_unload_clears_reader(self, mock_easyocr):
        mock_easyocr.Reader.return_value = MagicMock()

        w = TextDetectorWorker(input_dir="/some/dir")
        w.load()
        self.assertTrue(w.is_loaded())

        w.unload()

        self.assertFalse(w.is_loaded())
        self.assertIsNone(w._reader)

    @patch("workers.text_detector.easyocr")
    def test_unload_is_idempotent(self, mock_easyocr):
        """Calling unload() when already unloaded must not raise."""
        mock_easyocr.Reader.return_value = MagicMock()

        w = TextDetectorWorker(input_dir="/some/dir")
        w.load()
        w.unload()
        w.unload()  # second call — should be a no-op, not raise

        self.assertFalse(w.is_loaded())

    def test_unload_before_load_is_safe(self):
        """Unloading without ever loading must not raise."""
        w = TextDetectorWorker(input_dir="/some/dir")
        w.unload()  # should not raise
        self.assertFalse(w.is_loaded())


# ---------------------------------------------------------------------------
# Test: is_loaded() state machine
# ---------------------------------------------------------------------------

class TestTextDetectorWorkerIsLoaded(unittest.TestCase):

    @patch("workers.text_detector.easyocr")
    def test_is_loaded_reflects_lifecycle(self, mock_easyocr):
        mock_easyocr.Reader.return_value = MagicMock()

        w = TextDetectorWorker(input_dir="/some/dir")
        self.assertFalse(w.is_loaded())

        w.load()
        self.assertTrue(w.is_loaded())

        w.unload()
        self.assertFalse(w.is_loaded())


# ---------------------------------------------------------------------------
# Test: process_page() — bounding box conversion
# ---------------------------------------------------------------------------

class TestTextDetectorWorkerProcessPage(unittest.TestCase):

    def _make_worker_with_mock_reader(self, results, input_dir="/fake/input"):
        """
        Returns a TextDetectorWorker whose internal _reader is pre-configured
        as a MagicMock that returns *results* from readtext().
        """
        w = TextDetectorWorker(input_dir=input_dir, gpu=False)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = results
        w._reader = mock_reader
        w._loaded = True
        return w

    def test_single_result_produces_one_region(self):
        results = [_easyocr_result(10, 20, 100, 80, text="テスト", confidence=0.9)]
        w = self._make_worker_with_mock_reader(results)
        page = _make_page()
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        self.assertEqual(len(updated.text_regions), 1)
        region = updated.text_regions[0]
        self.assertEqual(region.id, "r0000")
        self.assertEqual(region.box.x_min, 10)
        self.assertEqual(region.box.y_min, 20)
        self.assertEqual(region.box.x_max, 100)
        self.assertEqual(region.box.y_max, 80)
        self.assertEqual(region.raw_text, "テスト")
        self.assertAlmostEqual(region.ocr_confidence, 0.9)

    def test_multiple_results_produce_correct_ids(self):
        results = [
            _easyocr_result(0, 0, 50, 50, text="A", confidence=0.8),
            _easyocr_result(100, 100, 200, 150, text="B", confidence=0.7),
            _easyocr_result(300, 200, 400, 300, text="C", confidence=0.6),
        ]
        w = self._make_worker_with_mock_reader(results)
        page = _make_page()
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        self.assertEqual(len(updated.text_regions), 3)
        self.assertEqual(updated.text_regions[0].id, "r0000")
        self.assertEqual(updated.text_regions[1].id, "r0001")
        self.assertEqual(updated.text_regions[2].id, "r0002")

    def test_process_page_replaces_existing_regions(self):
        """Existing text_regions must be REPLACED, not appended to."""
        old_region = TextRegion(
            id="r9999",
            box=BoundingBox(0, 0, 10, 10),
        )
        page = _make_page(regions=[old_region])

        new_results = [_easyocr_result(5, 5, 50, 50, text="新", confidence=0.88)]
        w = self._make_worker_with_mock_reader(new_results)
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        self.assertEqual(len(updated.text_regions), 1)
        self.assertEqual(updated.text_regions[0].id, "r0000")  # fresh numbering

    def test_empty_easyocr_results_clears_regions(self):
        """If EasyOCR finds nothing, text_regions should be an empty list."""
        page = _make_page(regions=[TextRegion(id="r0000", box=BoundingBox(0, 0, 10, 10))])
        w = self._make_worker_with_mock_reader([])
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        self.assertEqual(updated.text_regions, [])

    def test_bounding_box_coordinates_are_ints(self):
        """Coordinates parsed from float EasyOCR output must be cast to int."""
        results = [_easyocr_result(10.7, 20.3, 100.9, 80.1, text="X", confidence=0.5)]
        w = self._make_worker_with_mock_reader(results)
        page = _make_page()
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        box = updated.text_regions[0].box
        self.assertIsInstance(box.x_min, int)
        self.assertIsInstance(box.y_min, int)
        self.assertIsInstance(box.x_max, int)
        self.assertIsInstance(box.y_max, int)
        self.assertEqual(box.x_min, 10)
        self.assertEqual(box.y_min, 20)
        self.assertEqual(box.x_max, 100)
        self.assertEqual(box.y_max, 80)

    def test_ocr_confidence_is_float(self):
        results = [_easyocr_result(0, 0, 10, 10, confidence=1)]
        w = self._make_worker_with_mock_reader(results)
        page = _make_page()
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)

        self.assertIsInstance(updated.text_regions[0].ocr_confidence, float)

    def test_process_page_returns_same_page_object(self):
        """process_page must return the mutated page (may be the same object)."""
        w = self._make_worker_with_mock_reader([])
        page = _make_page()
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=True):
            returned = w.process_page(page, manifest)

        self.assertIs(returned, page)

    def test_process_page_raises_if_not_loaded(self):
        w = TextDetectorWorker(input_dir="/some/dir")
        page = _make_page()
        manifest = _make_manifest()

        with self.assertRaises(RuntimeError):
            w.process_page(page, manifest)

    def test_process_page_raises_if_image_not_found(self):
        w = self._make_worker_with_mock_reader([])
        page = _make_page(filename="missing.png")
        manifest = _make_manifest()

        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(FileNotFoundError):
                w.process_page(page, manifest)

    def test_readtext_called_with_correct_path(self):
        """reader.readtext must receive the fully-joined image path."""
        results = []
        w = self._make_worker_with_mock_reader(results, input_dir="/my/input")
        page = _make_page(filename="page01.png")
        manifest = _make_manifest()

        expected_path = os.path.join("/my/input", "page01.png")

        with patch("os.path.isfile", return_value=True):
            w.process_page(page, manifest)

        w._reader.readtext.assert_called_once_with(expected_path)


# ---------------------------------------------------------------------------
# Test: StageController integration
# ---------------------------------------------------------------------------

class TestTextDetectorStageControllerIntegration(unittest.TestCase):
    """
    Verifies that TextDetectorWorker integrates correctly with StageController:
    - marks phase_a_completed after processing
    - load/unload called exactly once per run()
    - save_manifest called after each page
    """

    def _make_engine_mock(self, pages):
        """Create a mock ManifestEngine with the given list of PageManifests."""
        manifest_mock = MagicMock()
        manifest_mock.pages = {p.image_filename: p for p in pages}

        engine_mock = MagicMock()
        engine_mock.manifest = manifest_mock
        return engine_mock

    @patch("workers.text_detector.easyocr")
    def test_stage_controller_marks_phase_a_completed(self, mock_easyocr):
        from stages.controller import StageController

        mock_easyocr.Reader.return_value = MagicMock()
        mock_easyocr.Reader.return_value.readtext.return_value = []

        page = _make_page("page01.png")
        engine = self._make_engine_mock([page])

        worker = TextDetectorWorker(input_dir="/fake/input", gpu=False)

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Stage6-TextDetection",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )

        with patch("os.path.isfile", return_value=True):
            report = controller.run()

        self.assertTrue(page.phase_a_completed)
        self.assertIn("page01.png", report.succeeded)

    @patch("workers.text_detector.easyocr")
    def test_load_and_unload_called_once_per_run(self, mock_easyocr):
        from stages.controller import StageController

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        mock_easyocr.Reader.return_value = mock_reader

        pages = [_make_page(f"page0{i}.png") for i in range(3)]
        engine = self._make_engine_mock(pages)

        worker = TextDetectorWorker(input_dir="/fake/input", gpu=False)
        load_spy = MagicMock(side_effect=worker.load)
        unload_spy = MagicMock(side_effect=worker.unload)
        worker.load = load_spy
        worker.unload = unload_spy

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Stage6-TextDetection",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )

        with patch("os.path.isfile", return_value=True):
            controller.run()

        load_spy.assert_called_once()
        unload_spy.assert_called_once()

    @patch("workers.text_detector.easyocr")
    def test_save_manifest_called_per_page(self, mock_easyocr):
        from stages.controller import StageController

        mock_easyocr.Reader.return_value = MagicMock()
        mock_easyocr.Reader.return_value.readtext.return_value = []

        pages = [_make_page("p1.png"), _make_page("p2.png")]
        engine = self._make_engine_mock(pages)

        worker = TextDetectorWorker(input_dir="/fake/input", gpu=False)

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Stage6-TextDetection",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )

        with patch("os.path.isfile", return_value=True):
            controller.run()

        self.assertEqual(engine.save_manifest.call_count, 2)

    @patch("workers.text_detector.easyocr")
    def test_already_completed_pages_are_skipped(self, mock_easyocr):
        from stages.controller import StageController

        mock_easyocr.Reader.return_value = MagicMock()

        page_done = _make_page("done.png")
        page_done.phase_a_completed = True
        page_todo = _make_page("todo.png")

        engine = self._make_engine_mock([page_done, page_todo])

        worker = TextDetectorWorker(input_dir="/fake/input", gpu=False)
        worker._reader = MagicMock()
        worker._reader.readtext.return_value = []
        worker._loaded = True

        controller = StageController(
            engine=engine,
            worker=worker,
            stage_name="Stage6-TextDetection",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False,
        )

        with patch("os.path.isfile", return_value=True):
            report = controller.run()

        self.assertIn("done.png", report.skipped)
        self.assertIn("todo.png", report.succeeded)


# ---------------------------------------------------------------------------
# Test: EasyOCR quad → BoundingBox edge cases
# ---------------------------------------------------------------------------

class TestBoundingBoxConversion(unittest.TestCase):
    """Detailed tests of the quad-to-BoundingBox mapping logic."""

    def _run_with_results(self, results, input_dir="/fake"):
        w = TextDetectorWorker(input_dir=input_dir, gpu=False)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = results
        w._reader = mock_reader
        w._loaded = True

        with patch("os.path.isfile", return_value=True):
            page = w.process_page(_make_page(), _make_manifest())
        return page.text_regions

    def test_zero_origin_box(self):
        regions = self._run_with_results([_easyocr_result(0, 0, 50, 50)])
        self.assertEqual(regions[0].box, BoundingBox(0, 0, 50, 50))

    def test_large_coordinates(self):
        regions = self._run_with_results([_easyocr_result(600, 900, 780, 1150)])
        self.assertEqual(regions[0].box, BoundingBox(600, 900, 780, 1150))

    def test_confidence_stored_accurately(self):
        regions = self._run_with_results([_easyocr_result(0, 0, 10, 10, confidence=0.1234)])
        self.assertAlmostEqual(regions[0].ocr_confidence, 0.1234, places=4)

    def test_raw_text_stored_as_string(self):
        regions = self._run_with_results([_easyocr_result(0, 0, 10, 10, text=42)])
        self.assertIsInstance(regions[0].raw_text, str)
        self.assertEqual(regions[0].raw_text, "42")


class TestTextDetectorWorkerMerging(unittest.TestCase):

    def test_merge_close_regions(self):
        """Verify that close vertical columns are merged and text concatenated RTL."""
        w = TextDetectorWorker(input_dir="/fake", gpu=False, enable_box_merge=True)
        w._loaded = True
        
        # Region A: left column (x: 10 to 30)
        # Region B: right column (x: 40 to 60)
        # B's text should come before A's text (RTL order)
        results = [
            _easyocr_result(10, 10, 30, 100, text="A", confidence=0.8),
            _easyocr_result(40, 10, 60, 100, text="B", confidence=0.9),
        ]
        
        w._reader = MagicMock()
        w._reader.readtext.return_value = results
        
        page = _make_page()
        manifest = _make_manifest()
        
        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)
            
        self.assertEqual(len(updated.text_regions), 1)
        r = updated.text_regions[0]
        self.assertEqual(r.box.x_min, 10)
        self.assertEqual(r.box.y_min, 10)
        self.assertEqual(r.box.x_max, 60)
        self.assertEqual(r.box.y_max, 100)
        self.assertEqual(r.raw_text, "BA")
        self.assertAlmostEqual(r.ocr_confidence, 0.85)

    def test_no_merge_far_regions(self):
        """Verify that regions far apart horizontally are not merged."""
        w = TextDetectorWorker(input_dir="/fake", gpu=False, enable_box_merge=True)
        w._loaded = True
        
        results = [
            _easyocr_result(10, 10, 30, 100, text="A", confidence=0.8),
            _easyocr_result(80, 10, 100, 100, text="B", confidence=0.9),
        ]
        
        w._reader = MagicMock()
        w._reader.readtext.return_value = results
        
        page = _make_page()
        manifest = _make_manifest()
        
        with patch("os.path.isfile", return_value=True):
            updated = w.process_page(page, manifest)
            
        self.assertEqual(len(updated.text_regions), 2)


if __name__ == "__main__":
    unittest.main()
