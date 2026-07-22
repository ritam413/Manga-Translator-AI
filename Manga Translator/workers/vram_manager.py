import logging
from workers.base import BaseWorker

logger = logging.getLogger("workers.vram_manager")

class VramManager:
    """
    Context manager that handles the load → use → unload lifecycle of a
    BaseWorker, guaranteeing unload() is always called.

    Usage:
        with VramManager(worker) as w:
            result = w.process_page(page, manifest)
        # worker is unloaded here, even if process_page raised

    Args:
        worker:       A BaseWorker instance.
        flush_cache:  If True (default), calls torch.cuda.empty_cache()
                      after unload() to aggressively reclaim VRAM.
    """

    def __init__(self, worker: BaseWorker, flush_cache: bool = True):
        self.worker = worker
        self.flush_cache = flush_cache

    def __enter__(self) -> BaseWorker:
        logger.info("[VramManager] Loading worker: %s", self.worker.worker_name)
        self.worker.load()
        return self.worker

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("[VramManager] Unloading worker: %s", self.worker.worker_name)
        try:
            self.worker.unload()
        except Exception as e:
            logger.error("[VramManager] Error during unload of %s: %s",
                         self.worker.worker_name, e, exc_info=True)
        if self.flush_cache:
            self._flush_vram()
        return False   # do not suppress exceptions from the body

    @staticmethod
    def _flush_vram() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("[VramManager] torch.cuda.empty_cache() called.")
        except ImportError:
            pass   # torch not installed; ignore silently
