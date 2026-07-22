import os
import shutil
import unittest
import tempfile
import sys
from unittest.mock import MagicMock, patch

from core.models import PageManifest, ProjectManifest
from core.manifest import ManifestEngine
from workers.base import BaseWorker
from workers.vram_manager import VramManager
from stages.controller import StageController, StageReport, PageResult
from quality.checker import QualityChecker, CheckReport


class MockWorker(BaseWorker):
    def __init__(self):
        self._loaded = False
        self.load_count = 0
        self.unload_count = 0
        self.processed_pages = []
        self.raise_on_pages = set()

    @property
    def worker_name(self) -> str:
        return "MockWorker"

    def load(self) -> None:
        self._loaded = True
        self.load_count += 1

    def process_page(self, page: PageManifest, manifest: ProjectManifest) -> PageManifest:
        if page.image_filename in self.raise_on_pages:
            raise ValueError(f"Forced failure on {page.image_filename}")
        self.processed_pages.append(page.image_filename)
        return page

    def unload(self) -> None:
        self._loaded = False
        self.unload_count += 1

    def is_loaded(self) -> bool:
        return self._loaded


class TestBaseWorker(unittest.TestCase):
    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            BaseWorker()  # type: ignore

    def test_subclass_instantiation(self):
        worker = MockWorker()
        self.assertEqual(worker.worker_name, "MockWorker")
        self.assertFalse(worker.is_loaded())
        
        worker.load()
        self.assertTrue(worker.is_loaded())
        self.assertEqual(worker.load_count, 1)
        
        worker.unload()
        self.assertFalse(worker.is_loaded())
        self.assertEqual(worker.unload_count, 1)


class TestVramManager(unittest.TestCase):
    def setUp(self):
        self.worker = MockWorker()

    def test_enter_exit_lifecycle(self):
        self.assertFalse(self.worker.is_loaded())
        with VramManager(self.worker, flush_cache=False) as w:
            self.assertEqual(w, self.worker)
            self.assertTrue(self.worker.is_loaded())
            self.assertEqual(self.worker.load_count, 1)
            self.assertEqual(self.worker.unload_count, 0)
        self.assertFalse(self.worker.is_loaded())
        self.assertEqual(self.worker.unload_count, 1)

    def test_exit_on_exception(self):
        self.assertFalse(self.worker.is_loaded())
        with self.assertRaises(RuntimeError):
            with VramManager(self.worker, flush_cache=False):
                self.assertTrue(self.worker.is_loaded())
                raise RuntimeError("Test exception")
        self.assertFalse(self.worker.is_loaded())
        self.assertEqual(self.worker.load_count, 1)
        self.assertEqual(self.worker.unload_count, 1)

    def test_flush_vram_torch_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            with VramManager(self.worker, flush_cache=True):
                pass
            mock_torch.cuda.is_available.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_flush_vram_torch_unavailable(self):
        with patch.dict(sys.modules, {'torch': None}):
            with VramManager(self.worker, flush_cache=True):
                pass  # Should not raise error even though torch is None


class TestStageController(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.engine = ManifestEngine(
            project_dir=self.test_dir,
            input_subdir="input",
            manifest_filename="manifest.json"
        )
        self.manifest = self.engine.load_or_create()
        
        # Populate manifest with some dummy pages
        self.page1 = PageManifest(image_filename="page1.png", width=100, height=200)
        self.page2 = PageManifest(image_filename="page2.png", width=100, height=200)
        self.page3 = PageManifest(image_filename="page3.png", width=100, height=200)
        
        self.engine.update_page("page1.png", self.page1)
        self.engine.update_page("page2.png", self.page2)
        self.engine.update_page("page3.png", self.page3)
        self.engine.save_manifest(backup=False)

        self.worker = MockWorker()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_should_process_filtering(self):
        # Set page1 complete for phase a, page2 complete for phase b
        self.page1.phase_a_completed = True
        self.page2.phase_b_completed = True
        self.engine.update_page("page1.png", self.page1)
        self.engine.update_page("page2.png", self.page2)

        controller_a = StageController(self.engine, self.worker, "Stage A", marks_phase="a", force=False)
        self.assertFalse(controller_a._should_process(self.page1))
        self.assertTrue(controller_a._should_process(self.page2))
        self.assertTrue(controller_a._should_process(self.page3))

        # With force=True
        controller_a_force = StageController(self.engine, self.worker, "Stage A", marks_phase="a", force=True)
        self.assertTrue(controller_a_force._should_process(self.page1))

        # With page_filter
        controller_filter = StageController(self.engine, self.worker, "Stage A", page_filter=["page1.png"])
        self.assertTrue(controller_filter._should_process(self.page1))
        self.assertFalse(controller_filter._should_process(self.page2))

    def test_run_success_lifecycle(self):
        controller = StageController(
            self.engine,
            self.worker,
            stage_name="TestStage",
            marks_phase="a",
            run_quality_check=False,
            flush_vram=False
        )
        
        report = controller.run()
        
        # Verify worker lifecycle
        self.assertEqual(self.worker.load_count, 1)
        self.assertEqual(self.worker.unload_count, 1)
        self.assertEqual(len(self.worker.processed_pages), 3)
        self.assertEqual(self.worker.processed_pages, ["page1.png", "page2.png", "page3.png"])
        
        # Verify phase marking
        self.assertTrue(self.engine.get_page("page1.png").phase_a_completed)
        self.assertTrue(self.engine.get_page("page2.png").phase_a_completed)
        self.assertTrue(self.engine.get_page("page3.png").phase_a_completed)
        
        # Verify report
        self.assertEqual(report.total_pages, 3)
        self.assertEqual(len(report.succeeded), 3)
        self.assertEqual(len(report.skipped), 0)
        self.assertEqual(len(report.failed), 0)

    def test_run_with_skips_and_force(self):
        self.page1.phase_a_completed = True
        self.engine.update_page("page1.png", self.page1)
        self.engine.save_manifest(backup=False)

        controller = StageController(
            self.engine,
            self.worker,
            stage_name="TestStage",
            marks_phase="a",
            run_quality_check=False,
            force=False,
            flush_vram=False
        )
        report = controller.run()
        
        # page1 skipped, page2 and page3 processed
        self.assertEqual(len(self.worker.processed_pages), 2)
        self.assertEqual(self.worker.processed_pages, ["page2.png", "page3.png"])
        self.assertEqual(len(report.skipped), 1)
        self.assertEqual(report.skipped, ["page1.png"])
        self.assertEqual(len(report.succeeded), 2)

    def test_run_with_page_exception_continues(self):
        self.worker.raise_on_pages.add("page2.png")
        
        # Patch engine.save_manifest to track save calls
        self.engine.save_manifest = MagicMock(side_effect=self.engine.save_manifest)

        controller = StageController(
            self.engine,
            self.worker,
            stage_name="TestStage",
            marks_phase="b",
            run_quality_check=False,
            flush_vram=False
        )
        report = controller.run()
        
        # Should process page1, fail on page2, and continue to page3
        self.assertEqual(self.worker.processed_pages, ["page1.png", "page3.png"])
        
        # Verify report status
        self.assertEqual(len(report.succeeded), 2)
        self.assertIn("page1.png", report.succeeded)
        self.assertIn("page3.png", report.succeeded)
        self.assertEqual(len(report.failed), 1)
        self.assertIn("page2.png", report.failed)
        self.assertEqual(report.page_results["page2.png"].error_message, "Forced failure on page2.png")
        
        # Save manifest should be called for page1 and page3 (2 times)
        self.assertEqual(self.engine.save_manifest.call_count, 2)

    def test_quality_checker_integration(self):
        mock_checker = MagicMock(spec=QualityChecker)
        mock_checker.check_page.return_value = CheckReport(
            image_filename="page1.png",
            region_results={}
        )

        controller = StageController(
            self.engine,
            self.worker,
            stage_name="TestStage",
            marks_phase="a",
            run_quality_check=True,
            quality_checker=mock_checker,
            page_filter=["page1.png"],
            flush_vram=False
        )
        report = controller.run()
        
        mock_checker.check_page.assert_called_once()
        page1_res = report.page_results["page1.png"]
        self.assertIsNotNone(page1_res.quality_report)
        self.assertEqual(page1_res.quality_report.image_filename, "page1.png")


class TestPageResultAndStageReport(unittest.TestCase):
    def test_page_result_dataclass(self):
        res = PageResult(image_filename="test.png", success=True, skipped=False)
        self.assertEqual(res.image_filename, "test.png")
        self.assertTrue(res.success)
        self.assertFalse(res.skipped)
        self.assertEqual(res.error_message, "")
        self.assertIsNone(res.quality_report)

    def test_stage_report_properties_and_summary(self):
        report = StageReport(stage_name="Detection")
        report.page_results = {
            "p1.png": PageResult(image_filename="p1.png", success=True),
            "p2.png": PageResult(image_filename="p2.png", success=True, skipped=True),
            "p3.png": PageResult(image_filename="p3.png", success=False, error_message="Failed")
        }

        self.assertEqual(report.total_pages, 3)
        self.assertEqual(report.succeeded, ["p1.png"])
        self.assertEqual(report.skipped, ["p2.png"])
        self.assertEqual(report.failed, ["p3.png"])
        self.assertEqual(
            report.summary(),
            "[Detection] total=3 ok=1 skipped=1 failed=1"
        )


if __name__ == "__main__":
    unittest.main()
