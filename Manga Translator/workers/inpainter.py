import logging
import os
import numpy as np
from typing import Optional

try:
    # pyrefly: ignore [missing-import]
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

from core.models import PageManifest, ProjectManifest
from workers.base import BaseWorker

logger = logging.getLogger("workers.inpainter")


class InpainterWorker(BaseWorker):
    """
    Stage 9 worker — inpaints (erases) the original Japanese text from each
    page image using OpenCV's INPAINT_TELEA algorithm.

    Lifecycle (managed by VramManager / StageController):
        worker.load()                        # validates cv2 importable
        updated_page = worker.process_page(page, manifest)
        worker.unload()                      # releases references (minimal)

    Args:
        input_dir:       Absolute path to input image directory.
        output_dir:      Absolute path to output directory (created if missing).
        inpaint_radius:  Radius used by cv2.inpaint() (default 3).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        inpaint_radius: int = 3,
        use_contour_inpaint: bool = False,
        layout_model_min_confidence: float = 0.5,
    ) -> None:
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._inpaint_radius = inpaint_radius
        self._use_contour_inpaint = use_contour_inpaint
        self._layout_model_min_confidence = layout_model_min_confidence
        self._loaded = False

    @property
    def worker_name(self) -> str:
        return "OpenCV Inpainter"

    def load(self) -> None:
        """
        Validate that cv2 is importable. Creates output_dir if needed.
        Idempotent — safe to call multiple times.
        """
        if self._loaded:
            logger.info("[InpainterWorker] Already loaded — skipping.")
            return
        if cv2 is None:
            raise ImportError(
                "opencv-python-headless is not installed. "
                "Run: pip install opencv-python-headless==4.8.1.78"
            )
        os.makedirs(self._output_dir, exist_ok=True)
        self._loaded = True
        logger.info("[InpainterWorker] Ready (output_dir=%s).", self._output_dir)

    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Inpaint all text regions on the page.

        Steps:
          1. Load the source image with cv2.imread() from input_dir.
          2. Build a binary mask: for each TextRegion.box, draw a white
             filled rectangle on a zeros mask (same HxW as image, uint8).
          3. Call cv2.inpaint(img, mask, inpaintRadius, cv2.INPAINT_TELEA).
          4. Write the result to output_dir/<page.image_filename> using
             cv2.imwrite(). Use a tempfile + os.replace for crash safety.
          5. Return the unchanged PageManifest (no mutation).

        Pages with zero text regions are still processed (empty mask → copy).
        Per-region drawing errors are caught and logged; the mask is written
        as-is. cv2 errors from inpaint() or imwrite() propagate up to
        StageController (which handles per-page failures).
        """
        if not self._loaded:
            raise RuntimeError(
                "InpainterWorker.load() must be called before process_page()."
            )

        img_path = os.path.join(self._input_dir, page.image_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"[InpainterWorker] Image not found: {img_path}"
            )

        logger.info(
            "[InpainterWorker] Inpainting %d region(s) in: %s",
            len(page.text_regions),
            page.image_filename,
        )

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(
                f"[InpainterWorker] cv2.imread failed for: {img_path}"
            )

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        page_gray = None
        if self._use_contour_inpaint:
            page_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for region in page.text_regions:
            box = region.box
            try:
                # Clamp coordinates to image bounds
                x1 = max(0, box.x_min)
                y1 = max(0, box.y_min)
                x2 = min(w, box.x_max)
                y2 = min(h, box.y_max)
                
                if x2 > x1 and y2 > y1:
                    fallback = True
                    if self._use_contour_inpaint and page_gray is not None:
                        from finetuning.pipeline_integration import build_jp_ink_inpaint_mask
                        ink_mask, confidence = build_jp_ink_inpaint_mask(page_gray, box)
                        if confidence >= self._layout_model_min_confidence and np.any(ink_mask):
                            mask = cv2.bitwise_or(mask, ink_mask)
                            fallback = False
                            logger.debug("[InpainterWorker] Contour mask used for %s (confidence=%.2f).", region.id, confidence)
                            
                    if fallback:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        logger.debug("[InpainterWorker] Rectangle fallback mask used for %s.", region.id)
            except Exception as exc:
                logger.warning(
                    "[InpainterWorker] Failed to mask region %s: %s", region.id, exc
                )

        inpainted = cv2.inpaint(img, mask, self._inpaint_radius, cv2.INPAINT_TELEA)

        # Crash-safe write: write to temp file, then atomic rename
        out_path = os.path.join(self._output_dir, page.image_filename)
        tmp_path = out_path + ".tmp" + os.path.splitext(page.image_filename)[1]
        cv2.imwrite(tmp_path, inpainted)
        os.replace(tmp_path, out_path)

        logger.info("[InpainterWorker] Saved inpainted page: %s", out_path)
        return page

    def unload(self) -> None:
        """Release worker state. Idempotent."""
        if not self._loaded:
            logger.info("[InpainterWorker] Already unloaded — skipping.")
            return
        self._loaded = False
        logger.info("[InpainterWorker] Unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded
