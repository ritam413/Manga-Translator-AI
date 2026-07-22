"""MangaOCR extraction for the Japanese raw bubbles."""

from __future__ import annotations

import logging
import unicodedata
from PIL import Image

from core.models import BoundingBox
from finetuning.models import JpOcrResult

logger = logging.getLogger(__name__)

# manga_ocr is a heavy dependency — import at module level with guard
try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None


class MangaOcrExtractor:
    """Lazy-loaded wrapper around manga-ocr MangaOcr model.

    Mirrors the OcrWorker lifecycle pattern from workers/ocr_worker.py:
      extractor.load()   # downloads/loads model once per batch
      result = extractor.extract_text(raw_pil, jp_box)
      extractor.unload() # releases model before EasyOCR reloads
    """

    def __init__(self, crop_margin_px: int = 4) -> None:
        self._mocr = None
        self._loaded = False
        self.crop_margin_px = crop_margin_px

    def load(self) -> None:
        """Initialise MangaOcr. Idempotent. Raises ImportError if not installed."""
        if self._loaded:
            logger.info("MangaOcrExtractor already loaded.")
            return

        logger.info("Loading MangaOcr model...")
        if MangaOcr is None:
            raise ImportError(
                "manga-ocr is not installed. Run: pip install manga-ocr"
            )
        self._mocr = MangaOcr()
        self._loaded = True
        logger.info("MangaOcr model ready.")

    def unload(self) -> None:
        """Release model reference. Idempotent."""
        if not self._loaded:
            logger.info("MangaOcrExtractor already unloaded.")
            return

        logger.info("Releasing MangaOcr model...")
        self._mocr = None
        self._loaded = False
        logger.info("MangaOcr model released.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def extract_text(
        self,
        raw_pil: Image.Image,
        jp_box: BoundingBox | dict,
    ) -> JpOcrResult:
        """Crop jp_box (+ margin), call MangaOcr, return JpOcrResult."""
        if not self._loaded or self._mocr is None:
            raise RuntimeError(
                "MangaOcrExtractor must be loaded before calling extract_text()."
            )

        # Normalize box
        if isinstance(jp_box, dict):
            box = BoundingBox(
                x_min=jp_box["x_min"],
                y_min=jp_box["y_min"],
                x_max=jp_box["x_max"],
                y_max=jp_box["y_max"],
            )
        else:
            box = jp_box

        if box.width() <= 0 or box.height() <= 0:
            raise ValueError(f"Invalid crop area for box: {box}")

        margin = self.crop_margin_px
        left = max(0, box.x_min - margin)
        upper = max(0, box.y_min - margin)
        right = min(raw_pil.width, box.x_max + margin)
        lower = min(raw_pil.height, box.y_max + margin)

        if (right - left) <= 0 or (lower - upper) <= 0:
            raise ValueError(f"Invalid crop area for box: {box}")

        crop = raw_pil.crop((left, upper, right, lower))
        crop_rgb = crop.convert("RGB")

        text: str = self._mocr(crop_rgb)
        text = text.strip()

        if text == "":
            return JpOcrResult(jp_ocr_text="", jp_ocr_confidence=0.5)

        text = unicodedata.normalize("NFKC", text)
        return JpOcrResult(jp_ocr_text=text, jp_ocr_confidence=1.0)


def extract_jp_text_from_page(
    raw_pil: Image.Image,
    jp_box: BoundingBox | dict,
    extractor: MangaOcrExtractor,
) -> JpOcrResult:
    """Thin functional wrapper; extractor must already be loaded."""
    return extractor.extract_text(raw_pil, jp_box)
