"""Data models for the finetuning dataset extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from core.models import BoundingBox


@dataclass
class DetectedRegion:
    """Text region detected by EasyOCR during finetuning extraction."""

    box: BoundingBox
    text: str
    confidence: float


@dataclass
class ExtractionState:
    """Checkpoint state for resumable extraction runs."""

    processed_pairs: set[str] = field(default_factory=set)
    failed_pairs: dict[str, str] = field(default_factory=dict)
    total_processed: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionState:
        processed = data.get("processed_pairs", [])
        if not isinstance(processed, list):
            raise ValueError("checkpoint 'processed_pairs' must be a list")

        failed = data.get("failed_pairs", {})
        if not isinstance(failed, dict):
            raise ValueError("checkpoint 'failed_pairs' must be a dict")

        return cls(
            processed_pairs=set(processed),
            failed_pairs={str(k): str(v) for k, v in failed.items()},
            total_processed=int(data.get("total_processed", len(processed))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed_pairs": sorted(self.processed_pairs),
            "failed_pairs": dict(sorted(self.failed_pairs.items())),
            "total_processed": self.total_processed,
        }


@dataclass
class ExtractionRecord:
    """One JSONL row emitted per successfully processed image pair."""

    pair_id: str
    manga_id: str
    page: str
    status: str
    raw_path: str
    eng_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "manga_id": self.manga_id,
            "page": self.page,
            "status": self.status,
            "raw_path": self.raw_path,
            "eng_path": self.eng_path,
        }

    @classmethod
    def stub(
        cls,
        raw_path: str,
        eng_path: str,
        manga_name: str,
        img_name: str,
    ) -> ExtractionRecord:
        pair_id = f"{manga_name}/{img_name}"
        page = page_from_filename(img_name)
        return cls(
            pair_id=pair_id,
            manga_id=manga_name,
            page=page,
            status="pending_extraction",
            raw_path=raw_path,
            eng_path=eng_path,
        )


def page_from_filename(img_name: str) -> str:
    stem = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
    return stem


@dataclass
class TypesettingFeatures:
    font_size: int
    padding: int
    line_spacing: float
    wrap_shape: Literal["rectangular", "elliptical"]
    line_count: int = 1
    typesetting_confidence: float = 1.0

    @classmethod
    def fallback(cls) -> "TypesettingFeatures":
        # mirrors workers/typesetter.py _BOX_PADDING=4, _MIN_FONT_SIZE=8
        return cls(
            font_size=12,
            padding=4,
            line_spacing=1.0,
            wrap_shape="rectangular",
            line_count=0,
            typesetting_confidence=0.0
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass
class JpOcrResult:
    """OCR result for a single Japanese text region."""

    jp_ocr_text: str
    jp_ocr_confidence: float

    @classmethod
    def fallback(cls) -> "JpOcrResult":
        return cls(jp_ocr_text="", jp_ocr_confidence=0.0)

    def to_dict(self) -> dict[str, str | float]:
        return {
            "jp_ocr_text": self.jp_ocr_text,
            "jp_ocr_confidence": self.jp_ocr_confidence,
        }


@dataclass
class StoryContextMetadata:
    """Story context metadata for a text region."""

    jp_ocr_text: str
    jp_ocr_confidence: float
    page_number: int
    story_position: str

    @classmethod
    def fallback(cls, page_number: int = 0, story_position: str = "") -> "StoryContextMetadata":
        return cls(
            jp_ocr_text="",
            jp_ocr_confidence=0.0,
            page_number=page_number,
            story_position=story_position,
        )

    def to_dict(self) -> dict[str, str | float | int]:
        return {
            "jp_ocr_text": self.jp_ocr_text,
            "jp_ocr_confidence": self.jp_ocr_confidence,
            "page_number": self.page_number,
            "story_position": self.story_position,
        }
