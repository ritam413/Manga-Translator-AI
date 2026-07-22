"""Incremental dataset extractor for finetuning."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from finetuning.checkpoint import load_checkpoint, reset_checkpoint, save_checkpoint
from finetuning.detection import detect_text_regions
from finetuning.geometry import match_regions
from finetuning.models import (
    ExtractionState,
    TypesettingFeatures,
    page_from_filename,
    JpOcrResult,
    StoryContextMetadata,
)
from finetuning.typesetting_features import extract_typesetting_features
from finetuning.mask_extraction import extract_inpaint_mask_from_pages, save_region_mask
from finetuning.manga_ocr_extraction import MangaOcrExtractor
from finetuning.page_order import PageOrderTracker
import cv2
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetExtractor:
    def __init__(
        self,
        raw_dir: str | Path,
        eng_dir: str | Path,
        output_file: str | Path,
        checkpoint_file: str | Path,
        checkpoint_interval: int = 50,
        reset: bool = False,
        gpu: bool = True,
        iou_threshold: float = 0.3,
        masks_dir: str | Path = "training_data/masks",
        use_manga_ocr: bool = True,
        ocr_crop_margin_px: int = 4,
        model_version: str = "v2",
        reset_checkpoint_reset: bool = False,
    ):
        self.raw_dir = Path(raw_dir)
        self.eng_dir = Path(eng_dir)
        self.output_file = Path(output_file)
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_interval = checkpoint_interval
        self.gpu = gpu
        self.iou_threshold = iou_threshold
        self.masks_dir = Path(masks_dir)
        self.use_manga_ocr = use_manga_ocr
        self.ocr_crop_margin_px = ocr_crop_margin_px
        self.model_version = model_version
        self.reset_checkpoint_reset = reset_checkpoint_reset

        self._manga_ocr_extractor = (
            MangaOcrExtractor(crop_margin_px=ocr_crop_margin_px)
            if use_manga_ocr
            else None
        )
        self._page_tracker = None

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        if reset or reset_checkpoint_reset:
            reset_checkpoint(self.checkpoint_file)
            
        if reset_checkpoint_reset and self.output_file.exists():
            self.output_file.unlink()

        self.state = load_checkpoint(self.checkpoint_file)

    def extract(self) -> dict[str, int]:
        logger.info("Starting dataset extraction")
        logger.info("Raw dir: %s", self.raw_dir)
        logger.info("Eng dir: %s", self.eng_dir)
        logger.info("Checkpoint interval: %s", self.checkpoint_interval)
        logger.info("GPU: %s, IoU threshold: %s", self.gpu, self.iou_threshold)

        if not self.raw_dir.exists():
            logger.warning("Raw directory %s does not exist.", self.raw_dir)
            return {"processed": 0, "skipped": 0, "failed": 0}

        if not self.eng_dir.exists():
            logger.warning("Eng directory %s does not exist.", self.eng_dir)

        pairs = self.find_image_pairs()
        logger.info("Found %d potential image pairs.", len(pairs))
        self._page_tracker = PageOrderTracker(pairs)

        from core.progress import MangaProgress

        processed_count = 0
        skipped_count = 0
        failed_count = 0
        reader: Any = None
        pairs_since_load = 0

        with MangaProgress() as mp:
            task_id = mp.add_task("Extracting dataset", total=len(pairs))
            try:
                for raw_path, eng_path, manga_name, img_name in pairs:
                    pair_id = f"{manga_name}/{img_name}"

                    if pair_id in self.state.processed_pairs:
                        skipped_count += 1
                        mp.advance(task_id)
                        continue

                    mp.print(f"Processing pair: {raw_path.name} -> {eng_path.name}")
                    logger.info("Processing pair: %s -> %s", raw_path, eng_path)

                    try:
                        if reader is None or pairs_since_load >= self.checkpoint_interval:
                            if reader is not None:
                                del reader
                                reader = None
                            if self._manga_ocr_extractor and self._manga_ocr_extractor.is_loaded:
                                self._manga_ocr_extractor.unload()

                            reader = self._create_reader()
                            if self._manga_ocr_extractor:
                                self._manga_ocr_extractor.load()
                            pairs_since_load = 0

                        records = self.process_pair(
                            raw_path, eng_path, manga_name, img_name, reader=reader
                        )
                        for record in records:
                            self.append_jsonl(record)
                        self.state.processed_pairs.add(pair_id)
                        self.state.failed_pairs.pop(pair_id, None)
                        self.state.total_processed += 1
                        processed_count += 1
                        pairs_since_load += 1

                        if processed_count % self.checkpoint_interval == 0:
                            logger.info(
                                "Saving checkpoint at %d pairs processed in this run...",
                                processed_count,
                            )
                            save_checkpoint(self.state, self.checkpoint_file)
                    except Exception as exc:
                        logger.error("Failed to process %s: %s", pair_id, exc)
                        self.state.failed_pairs[pair_id] = str(exc)
                        failed_count += 1
                    finally:
                        mp.advance(task_id)
            finally:
                if reader is not None:
                    del reader
                if self._manga_ocr_extractor and self._manga_ocr_extractor.is_loaded:
                    self._manga_ocr_extractor.unload()

        save_checkpoint(self.state, self.checkpoint_file)
        logger.info(
            "Extraction complete. Processed %d new, skipped %d, failed %d.",
            processed_count,
            skipped_count,
            failed_count,
        )
        return {
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
        }

    def _create_reader(self) -> Any:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError(
                "easyocr is not installed. Run: pip install easyocr==1.7.1"
            ) from exc
        logger.info("Loading EasyOCR Reader (languages=['ja','en'], gpu=%s)...", self.gpu)
        return easyocr.Reader(["ja", "en"], gpu=self.gpu)

    def process_pair(
        self,
        raw_path: Path,
        eng_path: Path,
        manga_name: str,
        img_name: str,
        reader: Any = None,
    ) -> list[dict]:
        self._validate_dimensions(raw_path, eng_path)

        jp_regions = detect_text_regions(
            raw_path, ["ja"], reader=reader, gpu=self.gpu,
            enable_box_merge=(self.model_version == "v2")
        )
        en_regions = detect_text_regions(
            eng_path, ["en"], reader=reader, gpu=self.gpu,
            enable_box_merge=(self.model_version == "v2")
        )

        matches = match_regions(jp_regions, en_regions, self.iou_threshold)

        unmatched_jp = len(jp_regions) - len(matches)
        unmatched_en = len(en_regions) - len(matches)
        if unmatched_jp or unmatched_en:
            logger.warning(
                "Pair %s/%s: %d unmatched JP, %d unmatched EN regions",
                manga_name,
                img_name,
                unmatched_jp,
                unmatched_en,
            )

        pair_id = f"{manga_name}/{img_name}"
        page = page_from_filename(img_name)
        rows: list[dict] = []

        raw_gray = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if raw_gray is None:
            raw_gray = np.zeros((10, 10), dtype=np.uint8)

        page_gray = cv2.imread(str(eng_path), cv2.IMREAD_GRAYSCALE)
        if page_gray is None:
            logger.warning("Failed to load eng_path image for typesetting features: %s", eng_path)
            # We'll just pass a dummy array or let fallback handle it
            page_gray = np.zeros((10, 10), dtype=np.uint8)

        raw_pil: PILImage.Image | None = None
        if self._manga_ocr_extractor and self._manga_ocr_extractor.is_loaded:
            try:
                raw_pil = PILImage.open(raw_path).convert("RGB")
            except Exception as exc:
                logger.warning("Failed to open raw_path for manga-ocr: %s: %s", raw_path, exc)

        for idx, (jp, en, iou) in enumerate(matches):
            try:
                features = extract_typesetting_features(page_gray, en.box)
            except Exception as exc:
                logger.warning("Failed to extract typesetting features for %s region %d: %s", pair_id, idx, exc)
                features = TypesettingFeatures.fallback()

            try:
                mask_result = extract_inpaint_mask_from_pages(raw_gray, page_gray, jp.box, en.box)
                mask_path = save_region_mask(
                    mask_result.mask, self.masks_dir, manga_name, page, idx
                )
                mask_roi_box_dict = asdict(mask_result.mask_roi_box)
                mask_confidence = round(mask_result.mask_confidence, 4)
                mask_pixel_count = mask_result.mask_pixel_count
            except Exception as exc:
                logger.warning("Failed to extract inpaint mask for %s region %d: %s", pair_id, idx, exc)
                mask_path = ""
                mask_roi_box_dict = {}
                mask_confidence = 0.0
                mask_pixel_count = 0

            # Phase 6: manga-ocr JP transcription
            if raw_pil is not None and self._manga_ocr_extractor and self._manga_ocr_extractor.is_loaded:
                try:
                    jp_ocr_result = self._manga_ocr_extractor.extract_text(raw_pil, jp.box)
                except Exception as exc:
                    logger.warning(
                        "Failed to extract JP OCR for %s region %d: %s", pair_id, idx, exc
                    )
                    jp_ocr_result = JpOcrResult.fallback()
            else:
                jp_ocr_result = JpOcrResult.fallback()

            # Phase 6: page ordering
            page_number = self._page_tracker.get_page_number(manga_name, img_name) if self._page_tracker else 0
            story_position = self._page_tracker.get_story_position(manga_name, img_name, idx) if self._page_tracker else ""

            story_metadata = StoryContextMetadata(
                jp_ocr_text=jp_ocr_result.jp_ocr_text,
                jp_ocr_confidence=jp_ocr_result.jp_ocr_confidence,
                page_number=page_number,
                story_position=story_position,
            )

            row = {
                "pair_id": pair_id,
                "manga_id": manga_name,
                "page": page,
                "region_index": idx,
                "status": "matched",
                "iou": round(iou, 4),
                "jp_box": asdict(jp.box),
                "en_box": asdict(en.box),
                "jp_text_detected": jp.text,
                "en_text_detected": en.text,
                "raw_path": str(raw_path),
                "eng_path": str(eng_path),
                "mask_path": mask_path,
                "mask_roi_box": mask_roi_box_dict,
                "mask_confidence": mask_confidence,
                "mask_pixel_count": mask_pixel_count,
            }
            row.update(features.to_dict())
            row.update(story_metadata.to_dict())
            rows.append(row)

        if not rows:
            raise ValueError("No region matches above IoU threshold")

        return rows

    @staticmethod
    def _validate_dimensions(raw_path: Path, eng_path: Path) -> None:
        from PIL import Image

        with Image.open(raw_path) as raw, Image.open(eng_path) as eng:
            if raw.size != eng.size:
                raise ValueError(
                    f"Dimension mismatch: {raw_path} {raw.size} vs {eng_path} {eng.size}"
                )

    def append_jsonl(self, record: dict) -> None:
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def find_image_pairs(self) -> list[tuple[Path, Path, str, str]]:
        pairs: list[tuple[Path, Path, str, str]] = []
        seen: set[str] = set()

        if not self.raw_dir.exists():
            return pairs

        for ext in IMAGE_EXTENSIONS:
            for raw_path in self.raw_dir.glob(f"*{ext}"):
                if not raw_path.is_file():
                    continue
                manga_name = "default"
                img_name = raw_path.name
                eng_path = self.eng_dir / img_name
                pair_id = f"{manga_name}/{img_name}"
                if eng_path.exists() and pair_id not in seen:
                    pairs.append((raw_path, eng_path, manga_name, img_name))
                    seen.add(pair_id)

            for raw_path in self.raw_dir.glob(f"*/*{ext}"):
                if not raw_path.is_file():
                    continue
                manga_name = raw_path.parent.name
                img_name = raw_path.name
                eng_path = self.eng_dir / manga_name / img_name
                pair_id = f"{manga_name}/{img_name}"
                if eng_path.exists() and pair_id not in seen:
                    pairs.append((raw_path, eng_path, manga_name, img_name))
                    seen.add(pair_id)

        pairs.sort(key=lambda item: (item[2], item[3]))
        return pairs
