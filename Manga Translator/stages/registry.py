from typing import List, Tuple, Type
from workers import (
    TextDetectorWorker,
    OcrWorker,
    TranslationWorker,
    InpainterWorker,
    TypesetterWorker,
)
from workers.base import BaseWorker


class StageRegistry:
    """
    Central registry for all executable pipeline stages.
    Provides mappings from stage names/IDs to worker classes and metadata.
    """

    def __init__(self) -> None:
        # Define the pipeline stages in execution order.
        # Each tuple is: (stage_id, name, description, worker_class, phase)
        self._stages: List[Tuple[int, str, str, Type[BaseWorker], str]] = [
            (
                6,
                "detect-text",
                "Text detection via EasyOCR (Phase A)",
                TextDetectorWorker,
                "a",
            ),
            (
                7,
                "ocr",
                "OCR via MangaOCR (Phase A)",
                OcrWorker,
                "a",
            ),
            (
                8,
                "translate",
                "Translation to English (Phase A)",
                TranslationWorker,
                "a",
            ),
            (
                9,
                "inpaint",
                "Remove original text (Phase B)",
                InpainterWorker,
                "b",
            ),
            (
                10,
                "typeset",
                "Render English text (Phase B)",
                TypesetterWorker,
                "b",
            ),
        ]

    def list_all(self) -> List[Tuple[int, str, str, Type[BaseWorker], str]]:
        """Return all stages in the registry."""
        return list(self._stages)

    def list_names(self) -> List[str]:
        """Return names of all stages in order."""
        return [stage[1] for stage in self._stages]

    def get_by_name(self, name: str) -> Tuple[int, Type[BaseWorker], str]:
        """
        Find stage by its name.
        Returns: (stage_id, worker_class, phase)
        Raises ValueError if name is not found.
        """
        for stage_id, stage_name, _, worker_class, phase in self._stages:
            if stage_name == name:
                return stage_id, worker_class, phase
        raise ValueError(f"Unknown stage name: '{name}'")

    def validate_range(self, from_name: str, to_name: str) -> None:
        """
        Validate a range of stage names.
        Raises ValueError if:
          - Either name is unknown
          - from_name appears after to_name in the execution order
        """
        names = self.list_names()
        if from_name not in names:
            raise ValueError(f"Unknown start stage name: '{from_name}'")
        if to_name not in names:
            raise ValueError(f"Unknown end stage name: '{to_name}'")

        from_index = names.index(from_name)
        to_index = names.index(to_name)

        if from_index > to_index:
            raise ValueError(
                f"Start stage '{from_name}' cannot be after end stage '{to_name}'"
            )

    def get_range(
        self, from_name: str, to_name: str
    ) -> List[Tuple[int, str, str, Type[BaseWorker], str]]:
        """
        Get all stages in the range from_name to to_name (inclusive).
        Raises ValueError if invalid.
        """
        self.validate_range(from_name, to_name)
        names = self.list_names()
        from_index = names.index(from_name)
        to_index = names.index(to_name)
        return self._stages[from_index : to_index + 1]