"""Tests for the PageOrderTracker."""

from __future__ import annotations

from pathlib import Path

from finetuning.page_order import PageOrderTracker


def test_sequential_numeric() -> None:
    # pairs: (raw_path, eng_path, manga_name, img_name)
    pairs = [
        (Path("r/m1/003.jpg"), Path("e/m1/003.jpg"), "manga1", "003.jpg"),
        (Path("r/m1/001.jpg"), Path("e/m1/001.jpg"), "manga1", "001.jpg"),
        (Path("r/m1/002.jpg"), Path("e/m1/002.jpg"), "manga1", "002.jpg"),
    ]
    tracker = PageOrderTracker(pairs)
    assert tracker.get_page_number("manga1", "001.jpg") == 1
    assert tracker.get_page_number("manga1", "002.jpg") == 2
    assert tracker.get_page_number("manga1", "003.jpg") == 3


def test_mixed_format_stems() -> None:
    pairs = [
        (Path("r/m1/page_02.png"), Path("e/m1/page_02.png"), "manga1", "page_02.png"),
        (Path("r/m1/001.png"), Path("e/m1/001.png"), "manga1", "001.png"),
    ]
    tracker = PageOrderTracker(pairs)
    assert tracker.get_page_number("manga1", "001.png") == 1
    assert tracker.get_page_number("manga1", "page_02.png") == 2


def test_alphabetic_fallback() -> None:
    pairs = [
        (Path("r/m1/intro.png"), Path("e/m1/intro.png"), "manga1", "intro.png"),
        (Path("r/m1/epilogue.png"), Path("e/m1/epilogue.png"), "manga1", "epilogue.png"),
    ]
    tracker = PageOrderTracker(pairs)
    # alphabetical: epilogue -> 1, intro -> 2
    assert tracker.get_page_number("manga1", "epilogue.png") == 1
    assert tracker.get_page_number("manga1", "intro.png") == 2


def test_multi_manga_independent() -> None:
    pairs = [
        (Path("r/m1/002.jpg"), Path("e/m1/002.jpg"), "manga1", "002.jpg"),
        (Path("r/m1/001.jpg"), Path("e/m1/001.jpg"), "manga1", "001.jpg"),
        (Path("r/m2/002.jpg"), Path("e/m2/002.jpg"), "manga2", "002.jpg"),
        (Path("r/m2/001.jpg"), Path("e/m2/001.jpg"), "manga2", "001.jpg"),
    ]
    tracker = PageOrderTracker(pairs)
    assert tracker.get_page_number("manga1", "001.jpg") == 1
    assert tracker.get_page_number("manga1", "002.jpg") == 2
    assert tracker.get_page_number("manga2", "001.jpg") == 1
    assert tracker.get_page_number("manga2", "002.jpg") == 2


def test_story_position_format() -> None:
    pairs = [
        (Path("r/m1/003.jpg"), Path("e/m1/003.jpg"), "manga1", "003.jpg"),
    ]
    tracker = PageOrderTracker(pairs)
    assert tracker.get_story_position("manga1", "003.jpg", 0) == "manga1::0001::00"
    assert tracker.get_story_position("manga1", "003.jpg", 5) == "manga1::0001::05"


def test_get_page_number_unknown() -> None:
    tracker = PageOrderTracker([])
    assert tracker.get_page_number("manga1", "unknown.png") == 0
    assert tracker.get_story_position("manga1", "unknown.png", 0) == ""
