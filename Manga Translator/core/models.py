import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

@dataclass
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def width(self) -> int:
        return self.x_max - self.x_min

    def height(self) -> int:
        return self.y_max - self.y_min

    def area(self) -> int:
        return self.width() * self.height()

    def center(self) -> tuple[int, int]:
        return (self.x_min + self.width() // 2, self.y_min + self.height() // 2)

    @classmethod
    def from_dict(cls, data: dict) -> 'BoundingBox':
        return cls(**data)


@dataclass
class TextRegion:
    id: str
    box: BoundingBox
    raw_text: str = ""
    translated_text: str = ""
    ocr_confidence: float = 0.0
    is_vertical: bool = True
    needs_review: bool = False
    region_index: int = 0
    debug_tone_reasons: List[str] = field(default_factory=list)
    speaker_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> 'TextRegion':
        data_copy = data.copy()
        if 'box' in data_copy and isinstance(data_copy['box'], dict):
            data_copy['box'] = BoundingBox.from_dict(data_copy['box'])
        return cls(**data_copy)


@dataclass
class PageManifest:
    image_filename: str
    width: int
    height: int
    text_regions: List[TextRegion] = field(default_factory=list)
    phase_a_completed: bool = False
    phase_b_completed: bool = False
    page_number: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> 'PageManifest':
        data_copy = data.copy()
        if 'text_regions' in data_copy:
            data_copy['text_regions'] = [
                TextRegion.from_dict(tr) for tr in data_copy['text_regions']
            ]
        return cls(**data_copy)


@dataclass
class ProjectManifest:
    project_name: str
    pages: Dict[str, PageManifest] = field(default_factory=dict)
    source_language: str = "ja"
    target_language: str = "en"
    manga_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ProjectManifest':
        data_copy = data.copy()
        if 'pages' in data_copy:
            data_copy['pages'] = {
                k: PageManifest.from_dict(v) for k, v in data_copy['pages'].items()
            }
        return cls(**data_copy)

    def save_to_file(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectManifest':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
