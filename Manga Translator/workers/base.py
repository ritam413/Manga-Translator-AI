import os
# Force redirect model/package downloads to E: drive to prevent C: drive exhaustion
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(project_dir, ".hf_cache"))
os.environ.setdefault("EASYOCR_MODULE_PATH", os.path.join(project_dir, ".easyocr_cache"))
os.environ.setdefault("ARGOS_PACKAGES_DIR", os.path.join(project_dir, ".argos_cache"))

from abc import ABC, abstractmethod
from core.models import PageManifest, ProjectManifest

class BaseWorker(ABC):
    """
    Abstract contract for all pipeline workers.

    Lifecycle:
        worker.load()              # load model into VRAM/RAM
        results = worker.process_page(page, manifest)  # do work, return mutated page
        worker.unload()            # release VRAM/RAM

    The StageController always calls them in this order and never skips unload().
    """

    @property
    @abstractmethod
    def worker_name(self) -> str:
        """Human-readable identifier used in logs and reports."""
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load all required models/resources into memory.
        Must be idempotent (calling twice must not crash or double-load).
        Should log at INFO level when loading starts and completes.
        """
        pass

    @abstractmethod
    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Process one page. Receives the current PageManifest, returns the
        updated PageManifest (may be the same object mutated in-place).

        Must NOT write to disk. Must NOT call load() or unload().
        The StageController handles persistence.

        Args:
            page:     PageManifest to process.
            manifest: Full ProjectManifest (read-only context: source/target lang etc.)

        Returns:
            The updated PageManifest.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Release all model resources and free VRAM.
        Must be idempotent.
        Should call torch.cuda.empty_cache() if PyTorch was used.
        Should log at INFO level confirming release.
        """
        pass

    # ── Optional override ──────────────────────────────────────────────
    def is_loaded(self) -> bool:
        """
        Returns True if load() has been called and unload() has not yet
        been called. Default implementation returns False (override in subclass).
        """
        return False
