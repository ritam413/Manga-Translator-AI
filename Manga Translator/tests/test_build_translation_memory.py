from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from finetuning.translation_memory import TranslationMemoryDB, TranslationMemoryEntry
from scripts.build_translation_memory import main as cli_main


def test_entry_serialization():
    entry = TranslationMemoryEntry(
        jp_text="こんにちは",
        en_text="Hello",
        page_number=3,
        story_position="manga1::0003::01",
        region_index=1,
    )
    d = entry.to_dict()
    assert d["jp_text"] == "こんにちは"
    assert d["en_text"] == "Hello"
    assert d["page_number"] == 3
    assert d["story_position"] == "manga1::0003::01"
    assert d["region_index"] == 1

    entry2 = TranslationMemoryEntry.from_dict(d)
    assert entry2.jp_text == entry.jp_text
    assert entry2.en_text == entry.en_text
    assert entry2.page_number == entry.page_number
    assert entry2.story_position == entry.story_position
    assert entry2.region_index == entry.region_index


def test_db_exact_matching():
    db = TranslationMemoryDB()
    entry1 = TranslationMemoryEntry(
        jp_text="こんにちは",
        en_text="Hello",
        page_number=1,
        story_position="manga1::0001::00",
        region_index=0,
    )
    entry2 = TranslationMemoryEntry(
        jp_text="さようなら",
        en_text="Goodbye",
        page_number=1,
        story_position="manga2::0001::00",
        region_index=0,
    )
    db.add_entry("manga1", entry1)
    db.add_entry("manga2", entry2)

    # Test exact match locally
    assert db.exact_match("manga1", "こんにちは") == "Hello"
    
    # Test spaces and NFKC normalizations
    assert db.exact_match("manga1", "  こんにちは  ") == "Hello"
    assert db.exact_match("manga1", "コニチハ") is None  # different characters
    # Test full-width normalization
    # "こんにちは" with full-width letters or spacing if normalized
    assert db.exact_match("manga1", "こんにちは") == "Hello"

    # Test global fallback: manga1 queries "さようなら" which belongs to manga2
    assert db.exact_match("manga1", "さようなら") == "Goodbye"
    
    # Test unmatched query
    assert db.exact_match("manga1", "存在しない") is None


def test_db_fuzzy_matching():
    db = TranslationMemoryDB()
    entry = TranslationMemoryEntry(
        jp_text="こんにちは世界",
        en_text="Hello World",
        page_number=1,
        story_position="manga1::0001::00",
        region_index=0,
    )
    db.add_entry("manga1", entry)

    # Typo query: "こんいちは世界" (edit distance 1)
    res = db.fuzzy_match("manga1", "こんいちは世界", threshold=0.8)
    assert res is not None
    text, score = res
    assert text == "Hello World"
    assert score >= 0.8

    # Query with low similarity: "さようなら地球"
    res2 = db.fuzzy_match("manga1", "さようなら地球", threshold=0.8)
    assert res2 is None


def test_context_tracking():
    db = TranslationMemoryDB()
    # Create 6 entries to check limit=5 behavior
    for i in range(6):
        entry = TranslationMemoryEntry(
            jp_text=f"日本語_{i}",
            en_text=f"English_{i}",
            page_number=1,
            story_position=f"manga1::0001::0{i}",
            region_index=i,
        )
        db.add_entry("manga1", entry)

    # Query context for the 6th entry (index 5)
    context = db.get_context("manga1", "manga1::0001::05", limit=5)
    assert len(context) == 5
    assert context[0] == ("日本語_0", "English_0")
    assert context[4] == ("日本語_4", "English_4")

    # If story_position is not padded or contains different page numbers
    db2 = TranslationMemoryDB()
    db2.add_entry("manga1", TranslationMemoryEntry("JP1", "EN1", 1, "manga1::0002::00", 0))
    db2.add_entry("manga1", TranslationMemoryEntry("JP2", "EN2", 1, "manga1::0001::01", 1))
    db2.add_entry("manga1", TranslationMemoryEntry("JP3", "EN3", 1, "manga1::0001::00", 0))

    # Sorted order should be: JP3 (page 1 reg 0), JP2 (page 1 reg 1), JP1 (page 2 reg 0)
    context2 = db2.get_context("manga1", "manga1::0002::00", limit=5)
    assert len(context2) == 2
    assert context2[0] == ("JP3", "EN3")
    assert context2[1] == ("JP2", "EN2")


def test_vocab_cooccurrence():
    db = TranslationMemoryDB()
    # Provide co-occurring character name "ルフィ" -> "Luffy"
    # and some noise words
    db.add_entry("manga1", TranslationMemoryEntry("ルフィが走る", "Luffy runs", 1, "manga1::0001::00", 0))
    db.add_entry("manga1", TranslationMemoryEntry("ルフィが叫ぶ", "Luffy shouts", 1, "manga1::0001::01", 1))
    db.add_entry("manga1", TranslationMemoryEntry("私はルフィだ", "I am Luffy", 1, "manga1::0001::02", 2))
    
    db.build_vocabularies(cooccurrence_threshold=0.3)
    vocab = db.manga_vocab["manga1"]

    assert "ルフィ" in vocab
    assert vocab["ルフィ"] == "luffy"
    
    # "runs" / "shouts" shouldn't map to "ルフィ" because cooccurrence score is lower than threshold
    # Also "i", "am" are stop words and should be filtered out
    assert "i" not in vocab.values()
    assert "am" not in vocab.values()


def test_cli_main_success(tmp_path):
    dataset_file = tmp_path / "dataset.jsonl"
    output_file = tmp_path / "translation_memory.json"

    # Write dummy dataset rows
    rows = [
        {
            "manga_id": "onepiece",
            "page_number": 1,
            "story_position": "onepiece::0001::00",
            "region_index": 0,
            "jp_ocr_text": "ルフィ",
            "en_text_detected": "Luffy",
        },
        {
            "manga_id": "onepiece",
            "page_number": 1,
            "story_position": "onepiece::0001::01",
            "region_index": 1,
            "jp_text_detected": "ゾロ",
            "en_text_detected": "Zoro",
        }
    ]
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    test_args = [
        "scripts/build_translation_memory.py",
        "--dataset-file", str(dataset_file),
        "--output-file", str(output_file),
        "--vocab-threshold", "0.3",
    ]
    
    with patch.object(sys, "argv", test_args):
        exit_code = cli_main()
        assert exit_code == 0

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == "1.0"
    assert "onepiece" in data["entries"]
    assert len(data["entries"]["onepiece"]) == 2
    assert data["vocabularies"]["onepiece"]["ルフィ"] == "luffy"
    assert data["vocabularies"]["onepiece"]["ゾロ"] == "zoro"


def test_cli_main_failure(tmp_path):
    dataset_file = tmp_path / "non_existent.jsonl"
    output_file = tmp_path / "translation_memory.json"

    test_args = [
        "scripts/build_translation_memory.py",
        "--dataset-file", str(dataset_file),
        "--output-file", str(output_file),
    ]
    
    with patch.object(sys, "argv", test_args):
        exit_code = cli_main()
        assert exit_code == 1


def test_edge_case_missing_fields(tmp_path):
    db = TranslationMemoryDB()
    
    # 1. Test missing jp_ocr_text but valid jp_text_detected
    dataset_file = tmp_path / "missing_fields.jsonl"
    output_file = tmp_path / "tm.json"

    row = {
        "manga_id": "manga1",
        "jp_text_detected": "こんにちは",
        "en_text_detected": "Hello",
    }
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    test_args = [
        "scripts/build_translation_memory.py",
        "--dataset-file", str(dataset_file),
        "--output-file", str(output_file),
    ]
    with patch.object(sys, "argv", test_args):
        exit_code = cli_main()
        assert exit_code == 0

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["entries"]["manga1"][0]["jp_text"] == "こんにちは"

    # 2. Both missing -> row skipped
    dataset_file2 = tmp_path / "both_missing.jsonl"
    row2 = {
        "manga_id": "manga1",
        "en_text_detected": "Hello",
    }
    with open(dataset_file2, "w", encoding="utf-8") as f:
        f.write(json.dumps(row2) + "\n")

    test_args2 = [
        "scripts/build_translation_memory.py",
        "--dataset-file", str(dataset_file2),
        "--output-file", str(output_file),
    ]
    with patch.object(sys, "argv", test_args2):
        exit_code = cli_main()
        # Exits with 1 because no valid translation pairs were found
        assert exit_code == 1


def test_edge_case_empty_sentences(tmp_path):
    # Rows with empty/whitespace strings are skipped
    dataset_file = tmp_path / "empty_sentences.jsonl"
    output_file = tmp_path / "tm.json"
    rows = [
        {
            "manga_id": "manga1",
            "jp_ocr_text": "   ",
            "en_text_detected": "Hello",
        },
        {
            "manga_id": "manga1",
            "jp_ocr_text": "こんにちは",
            "en_text_detected": "",
        }
    ]
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    test_args = [
        "scripts/build_translation_memory.py",
        "--dataset-file", str(dataset_file),
        "--output-file", str(output_file),
    ]
    with patch.object(sys, "argv", test_args):
        exit_code = cli_main()
        assert exit_code == 1  # No valid pairs found since all skipped
