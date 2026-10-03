import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.manifest import ManifestEngine
from core.models import PageManifest, ProjectManifest
from workers.base import BaseWorker
from workers.vram_manager import VramManager
from quality.checker import QualityChecker, CheckReport

logger = logging.getLogger("stages.controller")


@dataclass
class PageResult:
    """Outcome record for one page processed by StageController."""
    image_filename: str
    success: bool
    skipped: bool = False
    error_message: str = ""
    quality_report: Optional[CheckReport] = None


@dataclass
class StageReport:
    """Full outcome summary for one StageController.run() call."""
    stage_name: str
    page_results: Dict[str, PageResult] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.page_results)

    @property
    def succeeded(self) -> List[str]:
        return [f for f, r in self.page_results.items() if r.success and not r.skipped]

    @property
    def skipped(self) -> List[str]:
        return [f for f, r in self.page_results.items() if r.skipped]

    @property
    def failed(self) -> List[str]:
        return [f for f, r in self.page_results.items() if not r.success and not r.skipped]

    def summary(self) -> str:
        return (
            f"[{self.stage_name}] "
            f"total={self.total_pages} "
            f"ok={len(self.succeeded)} "
            f"skipped={len(self.skipped)} "
            f"failed={len(self.failed)}"
        )


class StageController:
    """
    Generic orchestrator that drives one pipeline stage end-to-end.

    Responsibilities:
      - Iterates over all (or a selected subset of) pages in the manifest.
      - Skips pages already marked complete (unless force=True).
      - Calls worker via VramManager (single load/unload for the whole run).
      - Runs QualityChecker on the page result AFTER the worker processes it.
      - Saves the manifest after EVERY page (crash-safe incremental progress).
      - Marks page.phase_a_completed or page.phase_b_completed on success.

    Args:
        engine:          Initialised ManifestEngine (manifest already loaded).
        worker:          A BaseWorker subclass instance (not yet loaded).
        stage_name:      Human-readable name for logging (e.g. "Phase-A TextDetection").
        marks_phase:     Which completion flag to set: "a" → phase_a_completed,
                         "b" → phase_b_completed, None → neither.
        run_quality_check: If True (default), run QualityChecker after each page.
        quality_checker:   QualityChecker instance. If None, uses QualityChecker().
        page_filter:     Optional list of image_filenames to restrict processing to.
        force:           If True, re-process pages already marked complete.
        flush_vram:      Passed to VramManager (default True).
    """

    def __init__(
        self,
        engine: ManifestEngine,
        worker: BaseWorker,
        stage_name: str,
        marks_phase: Optional[str] = None,    # "a" | "b" | None
        run_quality_check: bool = True,
        quality_checker: Optional[QualityChecker] = None,
        page_filter: Optional[List[str]] = None,
        force: bool = False,
        flush_vram: bool = True,
        checks_phase: Optional[str] = None,
        on_page_done: Optional[Callable[[str, bool], None]] = None,
    ) -> None:
        self.engine = engine
        self.worker = worker
        self.stage_name = stage_name
        self.marks_phase = marks_phase
        self.checks_phase = checks_phase if checks_phase is not None else marks_phase
        self.run_quality_check = run_quality_check
        self.quality_checker = quality_checker or QualityChecker()
        self.page_filter = set(page_filter) if page_filter else None
        self.force = force
        self.flush_vram = flush_vram
        self.on_page_done = on_page_done

    def run(self) -> StageReport:
        report = StageReport(stage_name=self.stage_name)
        manifest = self.engine.manifest
        if manifest is None:
            raise RuntimeError("ManifestEngine has no loaded manifest. Call load_or_create() first.")

        import re

        def _sort_key_natural(page: PageManifest) -> tuple:
            page_num = getattr(page, "page_number", None)
            if page_num is not None and isinstance(page_num, int) and page_num > 0:
                return (0, page_num, page.image_filename)
            m = re.search(r'(\d+)', page.image_filename)
            if m:
                return (1, int(m.group(1)), page.image_filename)
            return (2, 0, page.image_filename)

        pages = sorted(list(manifest.pages.values()), key=_sort_key_natural)
        pages_to_run = [p for p in pages if self._should_process(p)]
        pages_to_skip = [p for p in pages if not self._should_process(p)]

        # Record skipped pages immediately
        for page in pages_to_skip:
            report.page_results[page.image_filename] = PageResult(
                image_filename=page.image_filename,
                success=True,
                skipped=True,
            )
            logger.debug("[%s] Skipping %s (already complete)", self.stage_name, page.image_filename)

        logger.info("[%s] Starting — %d pages to process, %d skipped.",
                    self.stage_name, len(pages_to_run), len(pages_to_skip))

        if not pages_to_run:
            logger.info("[%s] Nothing to do.", self.stage_name)
            return report

        # IMPORTANT: Load worker ONCE for the whole batch, not per-page
        with VramManager(self.worker, flush_cache=self.flush_vram) as active_worker:
            for page in pages_to_run:
                logger.info("[%s] Processing page: %s", self.stage_name, page.image_filename)
                try:
                    # 1. Run the worker
                    updated_page = active_worker.process_page(page, manifest)

                    # 2. Optionally run quality checker
                    qreport = None
                    if self.run_quality_check:
                        qreport = self.quality_checker.check_page(updated_page)
                        logger.info("[%s] Quality: %s", self.stage_name, qreport.summary())

                    # 3. Mark completion and update engine
                    self._mark_complete(updated_page)
                    self.engine.update_page(updated_page.image_filename, updated_page)

                    # 4. Save after EVERY page (incremental crash recovery)
                    self.engine.save_manifest(backup=False)

                    report.page_results[page.image_filename] = PageResult(
                        image_filename=page.image_filename,
                        success=True,
                        quality_report=qreport,
                    )
                    logger.info("[%s] ✓ Page done: %s", self.stage_name, page.image_filename)
                    if self.on_page_done:
                        self.on_page_done(page.image_filename, True)

                except Exception as e:
                    logger.error("[%s] ✗ Failed on %s: %s",
                                 self.stage_name, page.image_filename, e, exc_info=True)
                    report.page_results[page.image_filename] = PageResult(
                        image_filename=page.image_filename,
                        success=False,
                        error_message=str(e),
                    )
                    if self.on_page_done:
                        self.on_page_done(page.image_filename, False)
                    # Continue to the next page — do NOT break the loop

        logger.info("[%s] Complete — %s", self.stage_name, report.summary())
        return report

    def _should_process(self, page: PageManifest) -> bool:
        """Return True if this page should be processed in this run."""
        if self.page_filter and page.image_filename not in self.page_filter:
            return False
        if self.force:
            return True
        if self.checks_phase == "a" and page.phase_a_completed:
            return False
        if self.checks_phase == "b" and page.phase_b_completed:
            return False
        return True

    def _mark_complete(self, page: PageManifest) -> None:
        """Set the appropriate phase completion flag."""
        if self.marks_phase == "a":
            page.phase_a_completed = True
        elif self.marks_phase == "b":
            page.phase_b_completed = True
