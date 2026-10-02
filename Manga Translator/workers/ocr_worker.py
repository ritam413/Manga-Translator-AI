import logging
import os
from typing import Optional

# manga_ocr is a heavy dependency — import at module level with guard
# so that patch("workers.ocr_worker.MangaOcr") works in tests
try:
    from manga_ocr import MangaOcr
except ImportError:          # pragma: no cover
    MangaOcr = None          # type: ignore[assignment,misc]

from PIL import Image

from core.models import PageManifest, ProjectManifest
from workers.base import BaseWorker

logger = logging.getLogger("workers.ocr_worker")


class OcrWorker(BaseWorker):
    """
    Stage 7 worker — re-OCRs every TextRegion detected in Stage 6 using
    MangaOCR, overwriting raw_text with higher-quality transcription.

    Lifecycle (managed by VramManager / StageController):
        worker.load()                       # downloads / loads MangaOcr model
        updated_page = worker.process_page(page, manifest)
        worker.unload()                     # releases model reference

    Args:
        input_dir:  Absolute path to input image directory (same as Stage 6).
        use_gpu:    Passed implicitly — MangaOcr auto-detects CUDA. Keep True.
    """

    def __init__(
        self,
        input_dir: str,
        tm_db: Optional[object] = None,
        enable_tm_correction: bool = False,
    ) -> None:
        self._input_dir = input_dir
        self._tm_db = tm_db
        self._enable_tm_correction = enable_tm_correction
        self._mocr = None
        self._loaded = False

    @property
    def worker_name(self) -> str:
        return "MangaOCR Cross-Check"

    def load(self) -> None:
        """
        Initialise MangaOcr. Idempotent — safe to call multiple times.
        MangaOcr auto-selects CUDA/CPU; no explicit gpu flag needed.
        """
        if self._loaded:
            logger.info("[OcrWorker] Already loaded — skipping.")
            return
        logger.info("[OcrWorker] Loading MangaOcr model…")
        if MangaOcr is None:
            raise ImportError(
                "manga-ocr is not installed. Run: pip install manga-ocr==0.1.11"
            )
        try:
            import manga_ocr.ocr
            import transformers
            if hasattr(transformers, "AutoImageProcessor"):
                # If manga_ocr.ocr.AutoFeatureExtractor is still the old class, patch from_pretrained fallback
                orig_afp = getattr(manga_ocr.ocr, "AutoFeatureExtractor", None)
                if orig_afp is not None and not hasattr(orig_afp, "_patched_for_manga_ocr"):
                    orig_from_pretrained = orig_afp.from_pretrained
                    def _safe_from_pretrained(*args, **kwargs):
                        try:
                            return orig_from_pretrained(*args, **kwargs)
                        except Exception:
                            return transformers.AutoImageProcessor.from_pretrained(*args, **kwargs)
                    orig_afp.from_pretrained = _safe_from_pretrained
                    orig_afp._patched_for_manga_ocr = True
        except Exception as patch_err:
            logger.debug("[OcrWorker] AutoFeatureExtractor patch skipped/failed: %s", patch_err)

        self._mocr = MangaOcr()
        self._loaded = True
        logger.info("[OcrWorker] MangaOcr model ready.")

    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Re-OCR every TextRegion on the page using MangaOcr.

        For each region:
          1. Open the full page image with Pillow.
          2. Crop to region.box: img.crop((x_min, y_min, x_max, y_max)).
          3. Call self._mocr(crop) → returns the recognised Japanese string.
          4. Overwrite region.raw_text with the new string.
          5. Set region.ocr_confidence = 1.0 (MangaOCR does not return scores;
             use 1.0 as a sentinel meaning "processed by MangaOCR").

        Regions with an empty bounding box (zero area) are skipped gracefully.
        Exceptions per-region are caught, logged, and the region is left unchanged
        (do NOT raise — one bad crop must not abort the whole page).

        Must NOT write to disk. Returns the mutated PageManifest.
        """
        if not self._loaded or self._mocr is None:
            raise RuntimeError(
                "OcrWorker.load() must be called before process_page()."
            )

        img_path = os.path.join(self._input_dir, page.image_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"[OcrWorker] Image not found: {img_path}"
            )

        logger.info("[OcrWorker] OCR-ing %d region(s) in: %s",
                    len(page.text_regions), page.image_filename)

        # Open once — crop per region (memory-efficient)
        img = Image.open(img_path).convert("RGB")

        manga_id = getattr(manifest, "manga_id", "") or getattr(manifest, "project_name", "") or "default"

        for region in page.text_regions:
            box = region.box
            if box.area() == 0:
                logger.debug("[OcrWorker] Skipping zero-area region %s", region.id)
                continue
            try:
                crop = img.crop((box.x_min, box.y_min, box.x_max, box.y_max))
                text = str(self._mocr(crop))
                
                if self._enable_tm_correction and self._tm_db is not None:
                    from finetuning.pipeline_integration import correct_ocr_with_vocab
                    corrected = correct_ocr_with_vocab(self._tm_db, manga_id, text)
                    if corrected != text:
                        logger.info("[OcrWorker] Corrected OCR spelling for %s: %r -> %r", region.id, text, corrected)
                    region.raw_text = corrected
                else:
                    region.raw_text = text
                    
                region.ocr_confidence = 1.0    # sentinel: processed by MangaOCR
                logger.debug("[OcrWorker] %s -> %r", region.id, region.raw_text)
            except Exception as e:
                logger.warning(
                    "[OcrWorker] Failed to OCR region %s in %s: %s",
                    region.id, page.image_filename, e
                )
                # Leave raw_text unchanged — do NOT re-raise

        logger.info("[OcrWorker] Finished page: %s", page.image_filename)
        return page

    def unload(self) -> None:
        """
        Release the MangaOcr model reference. Idempotent.
        VRAM cache flushing is handled by VramManager.
        """
        if not self._loaded:
            logger.info("[OcrWorker] Already unloaded — skipping.")
            return
        logger.info("[OcrWorker] Releasing MangaOcr model…")
        self._mocr = None
        self._loaded = False
        logger.info("[OcrWorker] MangaOcr model released.")

    def is_loaded(self) -> bool:
        return self._loaded
