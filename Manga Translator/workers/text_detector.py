"""
workers/text_detector.py — Stage 6: Phase A Text Detection

Uses EasyOCR to detect text bounding boxes in manga page images.
Subclasses BaseWorker so it plugs directly into StageController.
"""

import logging
import os
from typing import List, Optional

# easyocr is a heavy dependency that may not be installed in all environments.
# We import it at module level so that `patch("workers.text_detector.easyocr")`
# works correctly in tests.  If it is absent, the ImportError surfaces only when
# TextDetectorWorker.load() is actually called.
try:
    import easyocr
except ImportError:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]

from core.models import BoundingBox, PageManifest, ProjectManifest, TextRegion
from workers.base import BaseWorker

logger = logging.getLogger("workers.text_detector")



class TextDetectorWorker(BaseWorker):
    """
    Detects text regions in manga pages using EasyOCR.

    Lifecycle (managed by VramManager / StageController):
        worker.load()                          # initialises easyocr.Reader once
        updated_page = worker.process_page(page, manifest)
        worker.unload()                        # releases Reader, frees memory

    Args:
        input_dir:  Absolute path to the directory containing input images
                    (typically ``ManifestEngine.input_dir``).  Every page's
                    image is looked up as ``os.path.join(input_dir, page.image_filename)``.
        gpu:        Pass ``True`` to enable CUDA acceleration in EasyOCR.
                    Defaults to ``True``; set ``False`` for CPU-only environments.
        languages:  Language list passed to ``easyocr.Reader``.
                    Defaults to ``['ja', 'en']``.
    """

    def __init__(
        self,
        input_dir: str,
        gpu: bool = True,
        languages: Optional[List[str]] = None,
        enable_box_merge: bool = False,
    ) -> None:
        self._input_dir = input_dir
        self._gpu = gpu
        self._languages = languages if languages is not None else ["ja", "en"]
        self._reader = None
        self._loaded = False
        self._enable_box_merge = enable_box_merge

    # ── BaseWorker interface ──────────────────────────────────────────────────

    @property
    def worker_name(self) -> str:
        return "EasyOCR Text Detector"

    def load(self) -> None:
        """
        Initialise the EasyOCR Reader.  Idempotent — safe to call multiple times.
        """
        if self._loaded:
            logger.info("[TextDetectorWorker] Already loaded — skipping.")
            return
        logger.info(
            "[TextDetectorWorker] Loading EasyOCR Reader (languages=%s, gpu=%s)…",
            self._languages,
            self._gpu,
        )
        if easyocr is None:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr==1.7.1"
            )
        self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
        self._loaded = True
        logger.info("[TextDetectorWorker] EasyOCR Reader ready.")


    def process_page(
        self,
        page: PageManifest,
        manifest: ProjectManifest,
    ) -> PageManifest:
        """
        Run text detection on one page and populate ``page.text_regions``.

        EasyOCR returns results in the format::

            [(bbox_quad, text, confidence), …]

        where ``bbox_quad`` is::

            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]   # top-left → clockwise

        Each quad is converted to an axis-aligned ``BoundingBox`` using only the
        top-left and bottom-right corners (x1, y1, x2, y2).

        Existing ``page.text_regions`` is **replaced** (not appended to) so that
        re-running the stage from scratch produces a clean result.

        Returns:
            The mutated ``PageManifest`` (same object).
        """
        if not self._loaded or self._reader is None:
            raise RuntimeError(
                "TextDetectorWorker.load() must be called before process_page()."
            )

        img_path = os.path.join(self._input_dir, page.image_filename)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"[TextDetectorWorker] Image not found: {img_path}"
            )

        logger.info(
            "[TextDetectorWorker] Detecting text in: %s", page.image_filename
        )

        results = self._reader.readtext(img_path)

        regions: List[TextRegion] = []
        for region_idx, (bbox_quad, text, confidence) in enumerate(results):
            # bbox_quad: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            top_left = bbox_quad[0]
            bottom_right = bbox_quad[2]
            x1, y1 = top_left
            x2, y2 = bottom_right

            box = BoundingBox(
                x_min=int(x1),
                y_min=int(y1),
                x_max=int(x2),
                y_max=int(y2),
            )
            region = TextRegion(
                id=f"r{region_idx:04d}",
                box=box,
                raw_text=str(text),
                ocr_confidence=float(confidence),
                region_index=region_idx,
            )
            regions.append(region)

        logger.info(
            "[TextDetectorWorker] Detected %d text region(s) in %s.",
            len(regions),
            page.image_filename,
        )

        if self._enable_box_merge:
            merged = self._merge_close_regions(regions)
            logger.info(
                "[TextDetectorWorker] Merged close regions: %d -> %d on %s.",
                len(regions),
                len(merged),
                page.image_filename,
            )
            regions = merged

        # Replace — not append — so re-runs start clean
        page.text_regions = regions
        return page

    def _merge_close_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        if not regions:
            return []

        n = len(regions)
        adj = {i: set() for i in range(n)}
        
        # Horizontal gap threshold: 35px
        # Vertical overlap threshold: 30% of shorter region's height
        for i in range(n):
            for j in range(i + 1, n):
                r1 = regions[i]
                r2 = regions[j]
                
                if r1.box.x_min <= r2.box.x_min:
                    gap_x = r2.box.x_min - r1.box.x_max
                else:
                    gap_x = r1.box.x_min - r2.box.x_max
                    
                if gap_x > 35:
                    continue
                    
                overlap_y = max(0, min(r1.box.y_max, r2.box.y_max) - max(r1.box.y_min, r2.box.y_min))
                min_h = min(r1.box.height(), r2.box.height())
                overlap_ratio = overlap_y / min_h if min_h > 0 else 0.0
                
                if overlap_ratio >= 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Connected components (BFS)
        visited = set()
        components = []
        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)
                
        merged_regions = []
        for idx, comp in enumerate(components):
            comp_regions = [regions[i] for i in comp]
            if len(comp_regions) == 1:
                merged_regions.append(comp_regions[0])
                continue
                
            x_min = min(r.box.x_min for r in comp_regions)
            y_min = min(r.box.y_min for r in comp_regions)
            x_max = max(r.box.x_max for r in comp_regions)
            y_max = max(r.box.y_max for r in comp_regions)
            
            # Sort right-to-left (reverse by x_max)
            sorted_comp = sorted(comp_regions, key=lambda r: r.box.x_max, reverse=True)
            
            # Concatenate raw Japanese text
            merged_raw_text = "".join(r.raw_text for r in sorted_comp)
            
            # Average OCR confidence
            merged_confidence = sum(r.ocr_confidence for r in sorted_comp) / len(sorted_comp)
            
            from core.models import BoundingBox
            merged_box = BoundingBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
            
            merged_region = TextRegion(
                id="",  # Re-indexed later
                box=merged_box,
                raw_text=merged_raw_text,
                ocr_confidence=merged_confidence,
            )
            merged_regions.append(merged_region)
            
        # Re-sort all final regions by top-left for stable ordering
        merged_regions.sort(key=lambda r: (r.box.y_min, r.box.x_min))
        
        # Re-index
        for idx, r in enumerate(merged_regions):
            r.id = f"r{idx:04d}"
            r.region_index = idx
            
        return merged_regions

    def unload(self) -> None:
        """
        Release the EasyOCR Reader.  Idempotent — safe to call multiple times.
        VRAM cache flushing is handled by VramManager after unload().
        """
        if not self._loaded:
            logger.info("[TextDetectorWorker] Already unloaded — skipping.")
            return
        logger.info("[TextDetectorWorker] Unloading EasyOCR Reader…")
        self._reader = None
        self._loaded = False
        logger.info("[TextDetectorWorker] EasyOCR Reader released.")

    def is_loaded(self) -> bool:
        return self._loaded
