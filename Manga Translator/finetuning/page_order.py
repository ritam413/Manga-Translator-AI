"""Page order tracking and sorting for story context."""

from __future__ import annotations

import re
from pathlib import Path


def _sort_key(img_name: str) -> tuple[int, str]:
    stem = img_name.rsplit(".", 1)[0] if "." in img_name else img_name
    m = re.search(r"\d+", stem)
    return (int(m.group()), stem) if m else (0, stem)


class PageOrderTracker:
    """Map (manga_id, img_name) → 1-based page_number for story ordering.

    Built once before extraction; read-only during process_pair calls.
    """

    def __init__(self, pairs: list[tuple[Path, Path, str, str]]) -> None:
        """
        Args:
            pairs: Result of DatasetExtractor.find_image_pairs() —
                   list of (raw_path, eng_path, manga_name, img_name).
        """
        manga_pages: dict[str, set[str]] = {}
        for _, _, manga_id, img_name in pairs:
            manga_pages.setdefault(manga_id, set()).add(img_name)

        self._page_map: dict[tuple[str, str], int] = {}
        for manga_id, img_names in manga_pages.items():
            sorted_names = sorted(img_names, key=_sort_key)
            for idx, img_name in enumerate(sorted_names, 1):
                self._page_map[(manga_id, img_name)] = idx

    def get_page_number(self, manga_id: str, img_name: str) -> int:
        """Return 1-based page number. Returns 0 if not found (safety)."""
        return self._page_map.get((manga_id, img_name), 0)

    def get_story_position(self, manga_id: str, img_name: str, region_index: int) -> str:
        """Return sortable key: '{manga_id}::{page_number:04d}::{region_index:02d}'."""
        page_num = self.get_page_number(manga_id, img_name)
        if page_num == 0:
            return ""
        return f"{manga_id}::{page_num:04d}::{region_index:02d}"
